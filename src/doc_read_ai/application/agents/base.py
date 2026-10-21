"""Base dos agentes: cada agente = prompt de sistema + LLMPort + AgentConfig.

O agente não sabe qual provedor/modelo está por trás; isso vem do YAML (Strategy).
"""

from typing import TypeVar

from pydantic import BaseModel

from doc_read_ai.domain.config import AgentConfig
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.ports import LLMPort

T = TypeVar("T", bound=BaseModel)


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
        return await self._llm.generate_structured(
            system_prompt=self.system_prompt,
            instruction=instruction,
            output_schema=output_schema,
            document=document,
        )
