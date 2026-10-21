"""Telemetria consultável localmente (governança). A fonte é o OpenTelemetry: os mesmos spans
que vão para Langfuse/qualquer backend OTLP também são gravados localmente para a API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SpanRecord(BaseModel):
    trace_id: str  # 32 hex
    span_id: str  # 16 hex
    parent_span_id: str | None
    name: str
    kind: str  # INTERNAL, CLIENT, SERVER...
    status: str  # UNSET, OK, ERROR
    start_time: datetime
    end_time: datetime
    duration_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class AgentUsage(BaseModel):
    """Uso de LLM por agente + provedor + modelo."""

    agent: str
    provider: str
    model: str
    calls: int
    errors: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    avg_latency_ms: float
    p95_latency_ms: float


class StepLatency(BaseModel):
    """Latência por nó do grafo."""

    step: str
    count: int
    errors: int
    avg_ms: float
    p95_ms: float


class TelemetrySummary(BaseModel):
    since: datetime | None
    until: datetime | None
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    by_agent: list[AgentUsage]
    by_step: list[StepLatency]
