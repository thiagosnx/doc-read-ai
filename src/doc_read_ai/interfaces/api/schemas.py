"""DTOs da API (contrato com o frontend). Reaproveitam os modelos de domínio que já são
contrato público (ExtractionResult, FieldSpec, RunStats, TelemetrySummary, SpanRecord) e
escondem o resto (config de provedor, caminhos, segredos)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from doc_read_ai.domain.analysis import ExtractionResult, FieldSpec, RegisteredSchema, SchemaSource
from doc_read_ai.domain.config import AgentConfig
from doc_read_ai.domain.runs import Run, RunStats, RunStatus
from doc_read_ai.domain.telemetry import SpanRecord, TelemetrySummary


class ErrorOut(BaseModel):
    """Corpo de erro padrão de toda a API."""

    detail: str


# --- runs ---------------------------------------------------------------------------------


class RunSummaryOut(BaseModel):
    """Execução sem o resultado (listagens)."""

    id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None
    trace_id: str | None
    document_type: str | None
    schema_source: SchemaSource | None
    errors: list[str]

    @classmethod
    def from_run(cls, run: Run) -> "RunSummaryOut":
        return cls(**run.model_dump(exclude={"result"}), duration_ms=run.duration_ms)


class RunOut(RunSummaryOut):
    """Execução completa, com o resultado da extração quando houver."""

    result: ExtractionResult | None = Field(
        description="Ausente enquanto roda. Com DOC_READ_STORE_RESULTS=false, `data` só vem "
        "na resposta/evento ao vivo e não é guardado."
    )

    @classmethod
    def from_run(cls, run: Run) -> "RunOut":
        return cls(**run.model_dump(), duration_ms=run.duration_ms)


class RunPage(BaseModel):
    items: list[RunSummaryOut]
    limit: int
    offset: int


class SubmissionOut(BaseModel):
    """Resposta do upload: uma execução por arquivo, na ordem enviada."""

    runs: list[RunOut]


class StepEventOut(BaseModel):
    """Dado do evento SSE `step` (um por nó do grafo concluído)."""

    run_id: str
    step: str = Field(description="validate, classify, resolve_schema, design_schema, ...")
    status: Literal["completed", "failed"]
    detail: dict[str, Any]
    at: datetime


class TraceOut(BaseModel):
    trace_id: str
    spans: list[SpanRecord]


# --- schemas ------------------------------------------------------------------------------


class SchemaSummaryOut(BaseModel):
    document_type: str
    document_type_label: str
    category: str
    field_count: int
    created_at: datetime

    @classmethod
    def from_schema(cls, schema: RegisteredSchema) -> "SchemaSummaryOut":
        return cls(
            document_type=schema.document_type,
            document_type_label=schema.document_type_label,
            category=schema.category,
            field_count=len(schema.fields),
            created_at=schema.created_at,
        )


class SchemaOut(RegisteredSchema):
    json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema do modelo de extração gerado a partir dos campos"
    )


class SchemaIn(BaseModel):
    """Substitui o schema de um tipo de documento."""

    document_type_label: str = Field(min_length=1)
    category: str = Field(min_length=1)
    fields: list[FieldSpec] = Field(description="Ao menos um campo")


# --- agentes / config ---------------------------------------------------------------------


class AgentOut(BaseModel):
    name: str
    provider: str
    model: str
    max_tokens: int
    params: dict[str, Any]
    custom_system_prompt: bool = Field(description="True se o YAML sobrescreve o prompt padrão")

    @classmethod
    def from_config(cls, config: AgentConfig) -> "AgentOut":
        return cls(
            name=config.name,
            provider=config.provider,
            model=config.model,
            max_tokens=config.max_tokens,
            params=_redact(config.params),
            custom_system_prompt=bool(config.system_prompt),
        )


class ProviderOut(BaseModel):
    name: str
    timeout_seconds: float
    max_retries: int
    credentials_configured: bool = Field(description="Só presença da chave, nunca o valor")


class SchemaRegistryOut(BaseModel):
    enabled: bool
    count: int


class AgentsOut(BaseModel):
    agents: list[AgentOut]
    providers: list[ProviderOut]
    schema_registry: SchemaRegistryOut


_SECRET_HINTS = ("key", "secret", "token", "password", "credential", "auth")


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    """params são opções do provedor; por garantia, nada com cara de segredo sai na API."""
    return {
        k: "***" if any(h in k.lower() for h in _SECRET_HINTS) else v for k, v in params.items()
    }


# --- observabilidade / governança ---------------------------------------------------------


class PeriodOut(BaseModel):
    since: datetime | None
    until: datetime | None


class OtlpExportOut(BaseModel):
    traces: bool
    metrics: bool
    logs: bool


class GovernanceOut(BaseModel):
    raw_files_stored: Literal[False] = Field(
        default=False, description="O arquivo enviado nunca é gravado (só o sha256)"
    )
    results_stored: bool
    llm_content_capture: Literal["metadata_only", "content"]
    otlp_export: OtlpExportOut
    api_auth_enabled: bool


class ObservabilitySummaryOut(BaseModel):
    period: PeriodOut
    runs: RunStats
    telemetry: TelemetrySummary | None = Field(
        description="null quando a telemetria local não está configurada"
    )
    agents: list[AgentOut]
    schema_registry: SchemaRegistryOut
    governance: GovernanceOut


# --- saúde --------------------------------------------------------------------------------


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"


class CheckOut(BaseModel):
    ok: bool
    detail: str | None = None


class ReadyOut(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckOut]
