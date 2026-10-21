"""Base dos agentes: cada agente = prompt de sistema + LLMPort + AgentConfig.

O agente não sabe qual provedor/modelo está por trás; isso vem do YAML (Strategy).
Cada chamada vira um span `invoke_agent <agente>` (convenção GenAI); o span da chamada HTTP do
SDK do provedor, quando existir, fica como filho dele.
"""

from typing import TypeVar

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from doc_read_ai.domain.config import AgentConfig
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.ports import LLMPort

T = TypeVar("T", bound=BaseModel)

tracer = trace.get_tracer("doc_read_ai")


class BaseAgent:
    default_system_prompt: str = ""

    def __init__(self, llm: LLMPort, config: AgentConfig) -> None:
        self._llm = llm
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def system_prompt(self) -> str:
        """O prompt do YAML, quando definido, tem prioridade sobre o padrão do agente."""
        return self._config.system_prompt or self.default_system_prompt

    async def _generate(self, instruction: str, output_schema: type[T], document: Document) -> T:
        # Tokens não entram aqui como gen_ai.usage.*: quem os reporta é o span do SDK (contaria
        # duas vezes no backend). O adapter grava doc_read.usage.* neste span.
        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": self.name,
            "gen_ai.provider.name": self._config.provider,
            "gen_ai.request.model": self._config.model,
            "gen_ai.request.max_tokens": self._config.max_tokens,
        }
        # Exceções saem do `with` com status ERROR e evento de exceção (padrão do OTel).
        with tracer.start_as_current_span(
            f"invoke_agent {self.name}", kind=SpanKind.INTERNAL, attributes=attributes
        ):
            return await self._llm.generate_structured(
                system_prompt=self.system_prompt,
                instruction=instruction,
                output_schema=output_schema,
                document=document,
            )
