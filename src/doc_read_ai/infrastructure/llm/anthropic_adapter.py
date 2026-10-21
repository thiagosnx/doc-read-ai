"""Adapter Anthropic (Claude) para LLMPort, usando structured outputs do SDK oficial."""

import base64
import logging
import os
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from doc_read_ai.domain.config import AgentConfig, ProviderConfig
from doc_read_ai.domain.document import Document, DocumentKind
from doc_read_ai.domain.errors import LLMError
from doc_read_ai.domain.ports import LLMPort

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Limites da API da Anthropic para arquivos enviados em base64.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 32 * 1024 * 1024

# Header beta exigido pelo server-side fallback: forma escalar ("default") x lista de modelos.
FALLBACK_BETA_SCALAR = "server-side-fallback-2026-07-01"
FALLBACK_BETA_LIST = "server-side-fallback-2026-06-01"


class AnthropicLLM(LLMPort):
    """Envia documento + instrução ao Claude e devolve o schema Pydantic já validado."""

    def __init__(
        self,
        agent: AgentConfig,
        provider: ProviderConfig,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._agent = agent
        self._client = client or self._build_client(provider)
        self._extra = self._map_params(agent.params)

    @staticmethod
    def _build_client(provider: ProviderConfig) -> anthropic.AsyncAnthropic:
        api_key: str | None = None
        if provider.api_key_env:
            api_key = os.environ.get(provider.api_key_env, "").strip()
            if not api_key:
                raise LLMError(
                    f"A variável de ambiente '{provider.api_key_env}' não está definida ou está "
                    "vazia. Configure a chave da API da Anthropic (ex: no arquivo .env)."
                )
        return anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=provider.timeout_seconds,
            max_retries=provider.max_retries,
        )

    def _map_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Traduz `params` do YAML para kwargs do SDK. Chaves desconhecidas são ignoradas."""
        extra: dict[str, Any] = {}
        for key, value in params.items():
            if key == "effort":
                extra["output_config"] = {"effort": value}
            elif key == "fallbacks":
                extra["fallbacks"] = value
                beta = FALLBACK_BETA_LIST if isinstance(value, list) else FALLBACK_BETA_SCALAR
                extra["betas"] = [beta]
            else:
                logger.warning(
                    "Parâmetro '%s' do agente '%s' não é suportado pelo provider anthropic; "
                    "ignorando.",
                    key,
                    self._agent.name,
                )
        return extra

    @staticmethod
    def _document_block(document: Document) -> dict[str, Any]:
        if document.kind is DocumentKind.PDF:
            limit, block_type = MAX_PDF_BYTES, "document"
        else:
            limit, block_type = MAX_IMAGE_BYTES, "image"
        if document.size_bytes > limit:
            size_mb = document.size_bytes / 1024 / 1024
            raise LLMError(
                f"'{document.filename}' tem {size_mb:.1f} MB; a Anthropic aceita no máximo "
                f"{limit // 1024 // 1024} MB para {document.kind.value}."
            )
        data = base64.standard_b64encode(document.content).decode("ascii")
        return {
            "type": block_type,
            "source": {"type": "base64", "media_type": document.media_type, "data": data},
        }

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        instruction: str,
        output_schema: type[T],
        document: Document | None = None,
    ) -> T:
        # Documento antes do texto: o modelo lê o arquivo e depois a instrução.
        blocks: list[dict[str, Any]] = []
        if document is not None:
            blocks.append(self._document_block(document))
        blocks.append({"type": "text", "text": instruction})

        agent, model = self._agent.name, self._agent.model
        try:
            response = await self._client.beta.messages.parse(
                model=model,
                max_tokens=self._agent.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": blocks}],
                output_format=output_schema,
                **self._extra,
            )
        except anthropic.AuthenticationError as e:
            raise LLMError(
                f"[{agent}] Chave da API da Anthropic inválida ou sem permissão{_rid(e)}."
            ) from e
        except anthropic.BadRequestError as e:
            raise LLMError(
                f"[{agent}] Requisição rejeitada pela Anthropic (modelo {model}): "
                f"{e.message}{_rid(e)}"
            ) from e
        except anthropic.RateLimitError as e:
            raise LLMError(
                f"[{agent}] Limite de requisições da Anthropic atingido mesmo após retries. "
                f"Tente novamente em instantes{_rid(e)}."
            ) from e
        except anthropic.APIStatusError as e:
            raise LLMError(
                f"[{agent}] Erro da API da Anthropic (HTTP {e.status_code}): {e.message}{_rid(e)}"
            ) from e
        except anthropic.APIConnectionError as e:
            raise LLMError(
                f"[{agent}] Falha de conexão com a API da Anthropic (rede ou timeout): {e}"
            ) from e
        except ValidationError as e:
            # O SDK valida o JSON ao receber; truncamento ou recusa também caem aqui.
            raise LLMError(
                f"[{agent}] A resposta do modelo não segue o schema {output_schema.__name__} "
                f"(pode ter sido truncada ou recusada): {e.error_count()} erro(s) de validação."
            ) from e

        rid = getattr(response, "_request_id", None)
        rid_txt = f" (request_id={rid})" if rid else ""
        usage = response.usage
        logger.info(
            "LLM ok: agente=%s modelo=%s input_tokens=%s output_tokens=%s%s",
            agent,
            response.model,
            usage.input_tokens,
            usage.output_tokens,
            rid_txt,
        )

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details and details.category else "não informada"
            explanation = f" {details.explanation}" if details and details.explanation else ""
            raise LLMError(
                f"[{agent}] O modelo recusou processar o documento (categoria: {category})."
                f"{explanation}{rid_txt}"
            )
        if response.stop_reason == "max_tokens":
            raise LLMError(
                f"[{agent}] Resposta truncada em max_tokens={self._agent.max_tokens}. Aumente "
                f"`max_tokens` do agente '{agent}' no YAML de configuração{rid_txt}."
            )
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(
                f"[{agent}] O modelo não devolveu saída estruturada "
                f"(stop_reason={response.stop_reason}){rid_txt}."
            )
        return parsed


def _rid(error: anthropic.APIStatusError) -> str:
    return f" (request_id={error.request_id})" if error.request_id else ""
