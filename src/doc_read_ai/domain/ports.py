"""Portas (hexagonal). O domínio e a aplicação só conhecem estas interfaces.

Trocar Anthropic por OpenAI/Vertex = escrever um novo adapter de LLMPort e registrá-lo.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from doc_read_ai.domain.analysis import RegisteredSchema
from doc_read_ai.domain.config import AgentConfig
from doc_read_ai.domain.document import Document

T = TypeVar("T", bound=BaseModel)


class LLMPort(ABC):
    """Porta de saída para um modelo multimodal com saída estruturada."""

    @abstractmethod
    async def generate_structured(
        self,
        *,
        system_prompt: str,
        instruction: str,
        output_schema: type[T],
        document: Document | None = None,
    ) -> T:
        """Envia documento (se houver) + instrução; devolve instância validada de output_schema.

        Deve levantar LLMError em falhas do provedor, recusa, truncamento ou saída inválida.
        """


class LLMFactoryPort(ABC):
    """Cria o LLMPort certo a partir da config de um agente (Strategy por provedor)."""

    @abstractmethod
    def create(self, config: AgentConfig) -> LLMPort: ...


class SchemaRegistryPort(ABC):
    """Guarda um schema por tipo de documento, para que extrações do mesmo tipo sejam estáveis."""

    @abstractmethod
    def get(self, document_type: str) -> RegisteredSchema | None: ...

    @abstractmethod
    def save(self, schema: RegisteredSchema) -> None: ...

    @abstractmethod
    def list_all(self) -> list[RegisteredSchema]: ...

    @abstractmethod
    def delete(self, document_type: str) -> bool:
        """Remove o schema (o próximo documento do tipo terá o schema redesenhado)."""


class DocumentLoaderPort(ABC):
    """Porta de entrada de arquivos."""

    @abstractmethod
    def load(self, source: Path) -> Document:
        """Lê e valida o arquivo. Levanta UnsupportedDocumentError se o formato não for aceito."""
