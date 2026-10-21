"""Configura o SDK do OpenTelemetry (traces, métricas, logs) uma vez por processo.

- traces: OTLP (se configurado) + SQLite local (governança) + console (opcional);
- métricas: OTLP (se configurado);
- logs: OTLP (se configurado); trace_id/span_id sempre injetados nos LogRecords
  (atributos otelTraceID/otelSpanID, use no formato do logging se quiser vê-los).

Precisa rodar ANTES de criar os clientes dos provedores: o SDK da Anthropic só emite spans
`anthropic.messages.create` quando já existe um TracerProvider global. Por padrão o SDK grava só
metadados (sem prompts, respostas ou arquivos); `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=
content` liga o conteúdo, o que manda dados pessoais dos documentos para o backend.
"""

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from doc_read_ai.domain.config import ModelPricing
from doc_read_ai.domain.ports import TelemetryStorePort
from doc_read_ai.domain.telemetry import SpanRecord, TelemetrySummary
from doc_read_ai.infrastructure.config import load_env_file
from doc_read_ai.infrastructure.observability.propagation import AttributePropagationProcessor
from doc_read_ai.infrastructure.observability.settings import ObservabilitySettings
from doc_read_ai.infrastructure.observability.sqlite_store import (
    SqliteSpanExporter,
    SqliteTelemetryStore,
)

logger = logging.getLogger(__name__)

# Chamadas de LLM levam segundos: buckets do histograma até 5 minutos (em ms).
_DURATION_BUCKETS_MS = (
    50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000, 60000, 120000, 300000
)  # fmt: skip

_handles: "ObservabilityHandles | None" = None
_lock = threading.Lock()


class EmptyTelemetryStore(TelemetryStorePort):
    """Store vazio: telemetria desligada ou sem gravação local."""

    def get_trace(self, trace_id: str) -> list[SpanRecord]:
        return []

    def summary(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> TelemetrySummary:
        return TelemetrySummary(
            since=since,
            until=until,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            by_agent=[],
            by_step=[],
        )


@dataclass
class ObservabilityHandles:
    tracer_provider: Any  # TracerProvider do SDK (ou NoOp quando desligado)
    meter_provider: Any
    logger_provider: LoggerProvider | None
    telemetry_store: TelemetryStorePort

    def shutdown(self) -> None:
        """Descarrega tudo o que está pendente (traces, métricas, logs).

        Os providers continuam ativos e o próprio SDK os encerra no atexit: assim reabrir o app
        no mesmo processo (ex.: TestClient em vários testes) não perde telemetria.
        """
        for provider in (self.tracer_provider, self.meter_provider, self.logger_provider):
            flush = getattr(provider, "force_flush", None)
            if flush is None:
                continue
            try:
                flush()
            except Exception:  # telemetria nunca derruba a aplicação
                logger.warning("Falha ao descarregar telemetria", exc_info=True)


def configure_observability(
    *,
    service_name: str = "doc-read-ai",
    pricing: dict[str, ModelPricing] | None = None,
    settings: ObservabilitySettings | None = None,
) -> ObservabilityHandles:
    """Registra os providers globais. Idempotente: chamadas seguintes devolvem os mesmos handles
    (providers globais só podem ser definidos uma vez por processo)."""
    global _handles
    with _lock:
        if _handles is None:
            load_env_file()  # as variáveis OTEL_* do .env precisam estar em os.environ
            settings = settings or ObservabilitySettings()
            if settings.otel_sdk_disabled:
                _handles = _disabled_handles()
            else:
                _handles = _build_handles(service_name, pricing or {}, settings)
        return _handles


def instrument_fastapi(app: Any) -> None:
    """Spans HTTP de servidor para o app FastAPI (o health check fica de fora)."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")


def _disabled_handles() -> ObservabilityHandles:
    return ObservabilityHandles(
        tracer_provider=trace.NoOpTracerProvider(),
        meter_provider=metrics.NoOpMeterProvider(),
        logger_provider=None,
        telemetry_store=EmptyTelemetryStore(),
    )


def _build_handles(
    service_name: str, pricing: dict[str, ModelPricing], settings: ObservabilitySettings
) -> ObservabilityHandles:
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name or service_name,
            "service.version": _package_version(),
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(AttributePropagationProcessor())
    store: TelemetryStorePort = EmptyTelemetryStore()
    if settings.doc_read_telemetry_local:
        # Síncrono: um INSERT local por span, e a API enxerga o trace assim que ele termina.
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(SqliteSpanExporter(settings.doc_read_telemetry_db))
        )
        store = SqliteTelemetryStore(settings.doc_read_telemetry_db, pricing)
    if otlp_enabled("traces"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if settings.doc_read_otel_console:
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    readers = []
    if otlp_enabled("metrics"):
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    meter_provider = MeterProvider(
        metric_readers=readers,
        resource=resource,
        views=[
            View(
                instrument_name="doc_read.*duration",
                aggregation=ExplicitBucketHistogramAggregation(_DURATION_BUCKETS_MS),
            )
        ],
    )
    metrics.set_meter_provider(meter_provider)

    logger_provider: LoggerProvider | None = None
    if otlp_enabled("logs"):
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
        set_logger_provider(logger_provider)
    # Correlação trace_id <-> log sempre; o handler OTLP só quando há logger provider.
    LoggingInstrumentor().instrument(
        tracer_provider=tracer_provider,
        inject_trace_context=True,
        enable_log_auto_instrumentation=logger_provider is not None,
    )

    if trace.get_tracer_provider() is not tracer_provider:
        logger.warning("Já havia um TracerProvider global; os spans da aplicação vão para ele.")
    return ObservabilityHandles(tracer_provider, meter_provider, logger_provider, store)


def otlp_enabled(signal: str) -> bool:
    """Sinal (traces|metrics|logs) com endpoint OTLP definido e exportador diferente de none."""
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    exporters = os.environ.get(f"OTEL_{signal.upper()}_EXPORTER", "otlp").lower().split(",")
    if "otlp" not in {e.strip() for e in exporters}:
        return False
    endpoint = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    return bool(endpoint and endpoint.strip())


def _package_version() -> str:
    try:
        return version("doc-read-ai")
    except PackageNotFoundError:
        return "unknown"
