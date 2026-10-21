"""Portas (hexagonal). O domínio e a aplicação só conhecem estas interfaces.

Trocar Anthropic por OpenAI/Vertex = escrever um novo adapter de LLMPort e registrá-lo.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from doc_read_ai.domain.analysis import RegisteredSchema
from doc_read_ai.domain.config import AgentConfig
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.runs import Run, RunStats, RunStatus
from doc_read_ai.domain.telemetry import SpanRecord, TelemetrySummary

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
        Telemetria: o adapter deve gravar no span corrente os tokens consumidos (somando
        retentativas) em `doc_read.usage.input_tokens` e `doc_read.usage.output_tokens`.
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


class RunRepositoryPort(ABC):
    """Histórico de execuções (auditoria). Guarda metadados e resultado, nunca o arquivo."""

    @abstractmethod
    def save(self, run: Run) -> None:
        """Cria ou atualiza (upsert por id)."""

    @abstractmethod
    def get(self, run_id: str) -> Run | None: ...

    @abstractmethod
    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: RunStatus | None = None,
        document_type: str | None = None,
    ) -> list[Run]:
        """Mais recentes primeiro."""

    @abstractmethod
    def stats(self, since: datetime | None = None, until: datetime | None = None) -> RunStats: ...

    @abstractmethod
    def delete(self, run_id: str) -> bool:
        """Apaga a execução e o resultado (direito de eliminação, LGPD)."""


class TelemetryStorePort(ABC):
    """Consulta local dos spans OTel gravados (alimenta os endpoints de observabilidade)."""

    @abstractmethod
    def get_trace(self, trace_id: str) -> list[SpanRecord]:
        """Spans do trace ordenados por início; lista vazia se não existir."""

    @abstractmethod
    def summary(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> TelemetrySummary: ...


class DocumentLoaderPort(ABC):
    """Porta de entrada de arquivos."""

    @abstractmethod
    def load(self, source: Path) -> Document:
        """Lê e valida o arquivo. Levanta UnsupportedDocumentError se o formato não for aceito."""
