"""Execuções (runs): o histórico auditável de cada documento processado."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from doc_read_ai.domain.analysis import ExtractionResult, PipelineStatus, SchemaSource


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    EXTRACTED = "extracted"
    REJECTED = "rejected"
    FAILED = "failed"

    @classmethod
    def from_pipeline(cls, status: PipelineStatus) -> "RunStatus":
        return cls(status.value)

    @property
    def is_final(self) -> bool:
        return self in (RunStatus.EXTRACTED, RunStatus.REJECTED, RunStatus.FAILED)


class PipelineEvent(BaseModel):
    """Progresso emitido pelo caso de uso a cada nó do grafo (alimenta SSE/frontend)."""

    step: str = Field(description="Nó do grafo: validate, classify, ..., ou 'done' no final")
    status: Literal["completed", "failed", "done"]
    detail: dict[str, Any] = Field(default_factory=dict)
    result: ExtractionResult | None = Field(default=None, description="Só no evento 'done'")


class Run(BaseModel):
    """Um documento enviado para processamento. O arquivo em si NÃO é guardado, só o hash."""

    id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    trace_id: str | None = Field(default=None, description="Trace OTel (32 hex) da execução")
    document_type: str | None = None
    schema_source: SchemaSource | None = None
    result: ExtractionResult | None = None
    errors: list[str] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None


class RunStats(BaseModel):
    """Agregados das execuções num período (governança)."""

    total: int
    by_status: dict[str, int]
    by_document_type: dict[str, int]
    by_schema_source: dict[str, int]
    avg_duration_ms: float | None
    p95_duration_ms: float | None
