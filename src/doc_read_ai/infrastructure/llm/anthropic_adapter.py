"""Adapter Anthropic (Claude) para LLMPort, usando structured outputs do SDK oficial.

Structured outputs (decodificação restrita) tem limites de complexidade de schema: no máximo
16 campos com união/nulo e 24 campos opcionais por requisição, além de um limite interno de
gramática. Schemas dinâmicos de documentos grandes (ex: NF-e) passam disso, então o adapter
usa dois modos:
- strict:  `messages.parse` com output_format (garantia de schema pelo servidor);
- lenient: schema vai no prompt, a resposta JSON é validada com Pydantic e, se falhar,
           o modelo recebe os erros e tem uma chance de corrigir.

Telemetria: o SDK da Anthropic já emite o span `anthropic.messages.create` (com gen_ai.usage.*)
quando há um TracerProvider global. Aqui só gravamos no span corrente (`invoke_agent`) o modo,
as tentativas e os tokens somados em doc_read.usage.* (contrato do LLMPort), sem conteúdo.
"""

import base64
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from opentelemetry import metrics, trace
from pydantic import BaseModel, ValidationError

from doc_read_ai.domain.config import AgentConfig, ProviderConfig
from doc_read_ai.domain.document import Document, DocumentKind
from doc_read_ai.domain.errors import LLMError
from doc_read_ai.domain.ports import LLMPort

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_tokens = metrics.get_meter("doc_read_ai").create_counter(
    "doc_read.llm.tokens", unit="{token}", description="Tokens consumidos por agente/modelo"
)

# Limites da API da Anthropic para arquivos enviados em base64.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 32 * 1024 * 1024

# Limites de complexidade de structured outputs (somados em todos os schemas da requisição).
MAX_UNION_PARAMS = 16
MAX_OPTIONAL_PARAMS = 24

# Trechos da mensagem de erro 400 quando o schema estoura a complexidade permitida.
_COMPLEXITY_ERROR_HINTS = ("union types", "optional parameters", "too complex")

# Header beta exigido pelo server-side fallback: forma escalar ("default") x lista de modelos.
FALLBACK_BETA_SCALAR = "server-side-fallback-2026-07-01"
FALLBACK_BETA_LIST = "server-side-fallback-2026-06-01"

LENIENT_INSTRUCTION = (
    "Responda somente com um único objeto JSON válido que siga exatamente o JSON Schema abaixo, "
    "sem texto antes ou depois e sem blocos de código. Use null para campos sem valor no "
    "documento.\n\nJSON Schema:\n{schema}"
)
REPAIR_INSTRUCTION = (
    "O JSON que você devolveu não passou na validação do schema:\n{errors}\n\n"
    "Devolva o objeto JSON completo e corrigido, somente o JSON."
)


@dataclass
class _CallStats:
    """Acumulado de uma chamada a generate_structured (inclui a retentativa do modo lenient)."""

    mode: str = "strict"
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


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
            # Sem o nome do arquivo: a mensagem vai para spans/logs e pode conter dados pessoais.
            raise LLMError(
                f"O documento tem {size_mb:.1f} MB; a Anthropic aceita no máximo "
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
        stats = _CallStats()

        unions, optional = schema_complexity(output_schema.model_json_schema())
        if unions > MAX_UNION_PARAMS or optional > MAX_OPTIONAL_PARAMS:
            logger.info(
                "Schema %s grande demais para structured outputs (%d uniões, %d opcionais); "
                "usando modo lenient.",
                output_schema.__name__,
                unions,
                optional,
            )
            return await self._generate_lenient(system_prompt, blocks, output_schema, stats)

        try:
            return await self._generate_strict(system_prompt, blocks, output_schema, stats)
        except LLMError as e:
            # O limite interno de gramática não é publicado: se o servidor recusar o schema
            # por complexidade, tenta de novo no modo lenient.
            cause = e.__cause__
            if isinstance(cause, anthropic.BadRequestError) and _is_complexity_error(cause):
                logger.info("Schema recusado por complexidade; repetindo no modo lenient.")
                return await self._generate_lenient(system_prompt, blocks, output_schema, stats)
            raise

    async def _generate_strict(
        self,
        system_prompt: str,
        blocks: list[dict[str, Any]],
        output_schema: type[T],
        stats: _CallStats,
    ) -> T:
        stats.mode = "strict"
        response = await self._call(
            self._client.beta.messages.parse,
            output_schema,
            stats,
            system=system_prompt,
            messages=[{"role": "user", "content": blocks}],
            output_format=output_schema,
        )
        self._check_stop_reason(response)
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(
                f"[{self._agent.name}] O modelo não devolveu saída estruturada "
                f"(stop_reason={response.stop_reason}){_response_rid(response)}."
            )
        return parsed

    async def _generate_lenient(
        self,
        system_prompt: str,
        blocks: list[dict[str, Any]],
        output_schema: type[T],
        stats: _CallStats,
    ) -> T:
        stats.mode = "lenient"
        schema = json.dumps(output_schema.model_json_schema(), ensure_ascii=False)
        content = [*blocks, {"type": "text", "text": LENIENT_INSTRUCTION.format(schema=schema)}]
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        for attempt in (1, 2):
            response = await self._call(
                self._client.beta.messages.create,
                output_schema,
                stats,
                system=system_prompt,
                messages=messages,
            )
            self._check_stop_reason(response)
            text = "".join(b.text for b in response.content if b.type == "text")
            try:
                return output_schema.model_validate_json(_extract_json(text))
            except ValidationError as e:
                if attempt == 2:
                    raise LLMError(
                        f"[{self._agent.name}] A resposta não segue o schema "
                        f"{output_schema.__name__} mesmo após correção: "
                        f"{e.error_count()} erro(s) de validação{_response_rid(response)}."
                    ) from e
                logger.info(
                    "Resposta fora do schema %s (%d erros); pedindo correção ao modelo.",
                    output_schema.__name__,
                    e.error_count(),
                )
                # response.content inteiro (com blocos de thinking) mantém o histórico válido.
                messages += [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": REPAIR_INSTRUCTION.format(errors=e)},
                ]
        raise AssertionError("inalcançável")

    async def _call(
        self,
        method: Callable[..., Awaitable[Any]],
        output_schema: type[BaseModel],
        stats: _CallStats,
        **kwargs: Any,
    ) -> Any:
        """Chama o SDK e traduz erros para LLMError, do mais específico ao mais genérico."""
        agent, model = self._agent.name, self._agent.model
        stats.attempts += 1
        try:
            response = await method(
                model=model, max_tokens=self._agent.max_tokens, **kwargs, **self._extra
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

        usage = response.usage
        logger.info(
            "LLM ok: agente=%s modelo=%s input_tokens=%s output_tokens=%s%s",
            agent,
            response.model,
            usage.input_tokens,
            usage.output_tokens,
            _response_rid(response),
        )
        self._record_usage(stats, response)
        return response

    def _record_usage(self, stats: _CallStats, response: Any) -> None:
        """Grava no span corrente (invoke_agent) o acumulado da chamada e conta os tokens."""
        usage = response.usage
        input_tokens, output_tokens = usage.input_tokens or 0, usage.output_tokens or 0
        stats.input_tokens += input_tokens
        stats.output_tokens += output_tokens
        model = getattr(response, "model", None) or self._agent.model

        span = trace.get_current_span()
        span.set_attributes(
            {
                "doc_read.llm.mode": stats.mode,
                "doc_read.llm.attempts": stats.attempts,
                "doc_read.usage.input_tokens": stats.input_tokens,
                "doc_read.usage.output_tokens": stats.output_tokens,
                "gen_ai.response.model": model,
            }
        )
        if response.stop_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [str(response.stop_reason)])
        if request_id := getattr(response, "_request_id", None):
            span.set_attribute("doc_read.llm.request_id", str(request_id))

        labels = {"agent": self._agent.name, "provider": self._agent.provider, "model": model}
        _tokens.add(input_tokens, {**labels, "token_type": "input"})
        _tokens.add(output_tokens, {**labels, "token_type": "output"})

    def _check_stop_reason(self, response: Any) -> None:
        agent, rid = self._agent.name, _response_rid(response)
        if response.stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details and details.category else "não informada"
            explanation = f" {details.explanation}" if details and details.explanation else ""
            raise LLMError(
                f"[{agent}] O modelo recusou processar o documento (categoria: {category})."
                f"{explanation}{rid}"
            )
        if response.stop_reason == "max_tokens":
            raise LLMError(
                f"[{agent}] Resposta truncada em max_tokens={self._agent.max_tokens}. Aumente "
                f"`max_tokens` do agente '{agent}' no YAML de configuração{rid}."
            )


def schema_complexity(schema: dict[str, Any]) -> tuple[int, int]:
    """Conta (campos com união/nulo, campos opcionais) como a API conta, resolvendo $refs."""
    defs = schema.get("$defs", {})
    unions = optional = 0

    def walk(node: Any, stack: tuple[str, ...]) -> None:
        nonlocal unions, optional
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name in defs and name not in stack:  # proteção contra schema recursivo
                walk(defs[name], (*stack, name))
            return
        required = set(node.get("required", []))
        for prop_name, prop in node.get("properties", {}).items():
            if prop_name not in required:
                optional += 1
            if isinstance(prop, dict) and ("anyOf" in prop or isinstance(prop.get("type"), list)):
                unions += 1
            walk(prop, stack)
        walk(node.get("items"), stack)
        for key in ("anyOf", "allOf", "oneOf"):
            for sub in node.get(key, []):
                walk(sub, stack)

    walk(schema, ())
    return unions, optional


def _extract_json(text: str) -> str:
    """Isola o objeto JSON caso o modelo tenha colocado texto ou cercas ``` em volta."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _is_complexity_error(error: anthropic.BadRequestError) -> bool:
    message = str(error.message).lower()
    return any(hint in message for hint in _COMPLEXITY_ERROR_HINTS)


def _response_rid(response: Any) -> str:
    rid = getattr(response, "_request_id", None)
    return f" (request_id={rid})" if rid else ""


def _rid(error: anthropic.APIStatusError) -> str:
    return f" (request_id={error.request_id})" if error.request_id else ""
