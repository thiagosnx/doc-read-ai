"""OpenTelemetry: árvore de spans do pipeline, privacidade, store SQLite e setup (sem rede)."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from doc_read_ai.application.use_cases import ROOT_SPAN
from doc_read_ai.bootstrap import build_use_case
from doc_read_ai.domain.analysis import DocumentAnalysis, PipelineStatus, ValidationVerdict
from doc_read_ai.domain.config import ModelPricing
from doc_read_ai.domain.errors import LLMError
from doc_read_ai.infrastructure.loaders import LocalFileLoader
from doc_read_ai.infrastructure.observability import (
    ObservabilitySettings,
    SqliteSpanExporter,
    SqliteTelemetryStore,
    configure_observability,
)
from doc_read_ai.infrastructure.observability import setup as otel_setup
from doc_read_ai.infrastructure.registry import InMemorySchemaRegistry
from tests.conftest import agent_llms
from tests.fakes import FakeLLMFactory, png_bytes, write_file
from tests.test_anthropic_adapter import big_model, make_llm, text_response

PRICING = {"claude-haiku-5-5": ModelPricing(input_per_mtok=0.10, output_per_mtok=0.50)}
STEPS = ["validate", "classify", "resolve_schema", "design_schema", "build_schema", "extract"]
# Nome de arquivo com dados pessoais: não pode aparecer em nenhum span.
SENSITIVE_NAME = "rg_maria_souza_123.456.789-00.png"


@pytest.fixture(scope="module")
def memory(tmp_path_factory: pytest.TempPathFactory) -> InMemorySpanExporter:
    """Exportador em memória no provider global (configurado uma vez por processo)."""
    db = tmp_path_factory.mktemp("otel") / "telemetry.db"
    configure_observability(
        settings=ObservabilitySettings(otel_sdk_disabled=False, doc_read_telemetry_db=db)
    )
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture
def spans(memory: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    memory.clear()
    yield memory
    memory.clear()


def _use_case(config_path: Path, factory: FakeLLMFactory):
    return build_use_case(
        config_path=config_path,
        llm_factory=factory,
        loader=LocalFileLoader(),
        schema_registry=InMemorySchemaRegistry(),
    )


def _by_name(finished: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    return {s.name: s for s in finished}


def _parent(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent else None


async def test_pipeline_span_tree(
    config_path: Path, png_file: Path, nfe_factory: FakeLLMFactory, spans: InMemorySpanExporter
) -> None:
    result = await _use_case(config_path, nfe_factory).execute(png_file, run_id="run-42")

    assert result.status == PipelineStatus.EXTRACTED, result.errors
    named = _by_name(spans.get_finished_spans())
    root = named[ROOT_SPAN]
    assert root.parent is None
    assert root.attributes["doc_read.run_id"] == "run-42"
    assert root.attributes["doc_read.status"] == "extracted"
    assert result.trace_id == format(root.context.trace_id, "032x")

    for step in STEPS:
        span = named[f"doc_read.step {step}"]
        assert span.attributes["doc_read.step"] == step
        assert _parent(span) == root.context.span_id
        assert span.context.trace_id == root.context.trace_id

    agent_steps = {
        "orchestrator": "validate",
        "classifier": "classify",
        "analyst": "design_schema",
        "extractor": "extract",
    }
    for agent, step in agent_steps.items():
        span = named[f"invoke_agent {agent}"]
        assert span.kind == trace.SpanKind.INTERNAL
        assert _parent(span) == named[f"doc_read.step {step}"].context.span_id
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert span.attributes["gen_ai.agent.name"] == agent
        assert span.attributes["gen_ai.provider.name"] == "anthropic"
        assert span.attributes["gen_ai.request.model"]
        assert span.attributes["gen_ai.request.max_tokens"] > 0
        # tokens canônicos ficam no span do SDK; aqui contaria duas vezes
        assert not any(k.startswith("gen_ai.usage.") for k in span.attributes)

    # etapas sem LLM não abrem invoke_agent
    assert len([n for n in named if n.startswith("invoke_agent ")]) == 4


async def test_filename_never_reaches_spans(
    config_path: Path, tmp_path: Path, nfe_factory: FakeLLMFactory, spans: InMemorySpanExporter
) -> None:
    path = write_file(tmp_path, SENSITIVE_NAME, png_bytes())

    result = await _use_case(config_path, nfe_factory).execute(path)

    assert result.status == PipelineStatus.EXTRACTED
    finished = spans.get_finished_spans()
    assert finished
    for span in finished:
        texts = [str(v) for v in span.attributes.values()] + [span.status.description or ""]
        texts += [str(v) for e in span.events for v in e.attributes.values()]
        assert not any("maria" in t or SENSITIVE_NAME in t for t in texts), span.name


async def test_failing_step_has_error_status(
    config_path: Path,
    png_file: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
    spans: InMemorySpanExporter,
) -> None:
    factory = FakeLLMFactory(agent_llms(valid_verdict, nfe_analysis, LLMError("truncada")))

    result = await _use_case(config_path, factory).execute(png_file)

    assert result.status == PipelineStatus.FAILED
    named = _by_name(spans.get_finished_spans())
    extract = named["doc_read.step extract"]
    assert extract.status.status_code == StatusCode.ERROR
    assert "LLMError" in extract.status.description
    assert any(e.name == "exception" for e in extract.events)
    assert named["invoke_agent extractor"].status.status_code == StatusCode.ERROR
    assert named[ROOT_SPAN].status.status_code == StatusCode.ERROR
    assert named["doc_read.step classify"].status.status_code == StatusCode.UNSET


async def test_anthropic_adapter_records_usage_on_agent_span(
    spans: InMemorySpanExporter,
) -> None:
    """Lenient com correção: duas chamadas, tokens somados, sem gen_ai.usage.* no span."""
    model = big_model(total=(float | None, None))
    create = AsyncMock(
        side_effect=[text_response('{"total": "x"}'), text_response('{"total": 10.5}')]
    )
    llm, _ = make_llm(create=create)

    with trace.get_tracer("doc_read_ai").start_as_current_span("invoke_agent extractor"):
        await llm.generate_structured(system_prompt="s", instruction="i", output_schema=model)

    attrs = _by_name(spans.get_finished_spans())["invoke_agent extractor"].attributes
    assert attrs["doc_read.llm.mode"] == "lenient"
    assert attrs["doc_read.llm.attempts"] == 2
    assert attrs["doc_read.usage.input_tokens"] == 20
    assert attrs["doc_read.usage.output_tokens"] == 10
    assert attrs["gen_ai.response.model"] == "claude-haiku-5-5"
    assert attrs["gen_ai.response.finish_reasons"] == ("end_turn",)
    assert not any(k.startswith("gen_ai.usage.") for k in attrs)


async def test_pipeline_spans_are_queryable_in_sqlite_store(
    config_path: Path,
    png_file: Path,
    nfe_factory: FakeLLMFactory,
    tmp_path: Path,
    memory: InMemorySpanExporter,
) -> None:
    db = tmp_path / "pipeline.db"
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.add_span_processor(SimpleSpanProcessor(SqliteSpanExporter(db)))

    result = await _use_case(config_path, nfe_factory).execute(png_file)

    records = SqliteTelemetryStore(db).get_trace(result.trace_id)
    assert records[0].name == ROOT_SPAN and records[0].parent_span_id is None
    assert [r.start_time for r in records] == sorted(r.start_time for r in records)
    by_name = {r.name: r for r in records}
    assert by_name["doc_read.step extract"].parent_span_id == records[0].span_id
    assert {f"doc_read.step {s}" for s in STEPS} <= set(by_name)


def test_sqlite_store_summary_uses_pricing(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    provider = TracerProvider()  # local: não mexe no global
    provider.add_span_processor(SimpleSpanProcessor(SqliteSpanExporter(db)))
    app, sdk = provider.get_tracer("doc_read_ai"), provider.get_tracer("anthropic")

    def agent(name: str, model: str, tokens_in: int, tokens_out: int) -> dict:
        return {
            "gen_ai.agent.name": name,
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": model,
            "doc_read.usage.input_tokens": tokens_in,
            "doc_read.usage.output_tokens": tokens_out,
        }

    with app.start_as_current_span(ROOT_SPAN) as root:
        with app.start_as_current_span(
            "doc_read.step classify", attributes={"doc_read.step": "classify"}
        ):
            with app.start_as_current_span(
                "invoke_agent classifier",
                attributes=agent("classifier", "claude-haiku-5-5", 1000, 500),
            ):
                # span do SDK (outro escopo): não entra no resumo, nem com gen_ai.usage.*
                with sdk.start_as_current_span(
                    "anthropic.messages.create",
                    attributes={
                        "gen_ai.usage.input_tokens": 1000,
                        "gen_ai.input.messages": "conteúdo sensível",
                    },
                ):
                    pass
        with app.start_as_current_span(
            "doc_read.step design_schema", attributes={"doc_read.step": "design_schema"}
        ):
            with app.start_as_current_span(
                "invoke_agent analyst", attributes=agent("analyst", "modelo-sem-preco", 2000, 300)
            ):
                pass
        with app.start_as_current_span(
            "doc_read.step extract", attributes={"doc_read.step": "extract"}
        ) as failed:
            failed.set_status(trace.Status(StatusCode.ERROR, "falhou"))

    store = SqliteTelemetryStore(db, PRICING)
    trace_id = format(root.get_span_context().trace_id, "032x")
    records = store.get_trace(trace_id)
    assert [r.name for r in records][:3] == [
        ROOT_SPAN,
        "doc_read.step classify",
        "invoke_agent classifier",
    ]
    sdk_span = next(r for r in records if r.name == "anthropic.messages.create")
    assert sdk_span.parent_span_id == records[2].span_id
    assert "gen_ai.input.messages" not in sdk_span.attributes  # conteúdo nunca é gravado
    assert store.get_trace("0" * 32) == []

    summary = store.summary()
    assert summary.llm_calls == 2
    assert (summary.input_tokens, summary.output_tokens) == (3000, 800)
    usage = {a.agent: a for a in summary.by_agent}
    assert usage["classifier"].cost_usd == pytest.approx(1000 * 0.10 / 1e6 + 500 * 0.50 / 1e6)
    assert usage["analyst"].cost_usd == 0.0  # modelo sem preço
    assert summary.cost_usd == pytest.approx(usage["classifier"].cost_usd)
    steps = {s.step: s for s in summary.by_step}
    assert set(steps) == {"classify", "design_schema", "extract"}
    assert steps["extract"].errors == 1 and steps["classify"].errors == 0
    assert steps["classify"].p95_ms >= 0

    future = datetime.now(UTC) + timedelta(hours=1)
    assert store.summary(since=future).llm_calls == 0
    assert store.summary(until=future).llm_calls == 2


def test_configure_observability_is_idempotent(memory: InMemorySpanExporter) -> None:
    first = configure_observability()
    second = configure_observability(service_name="outro")

    assert first is second
    assert trace.get_tracer_provider() is first.tracer_provider
    first.shutdown()  # só descarrega: o provider segue ativo
    with trace.get_tracer("doc_read_ai").start_as_current_span("depois-do-shutdown"):
        pass
    assert "depois-do-shutdown" in _by_name(memory.get_finished_spans())


def test_disabled_mode_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_setup, "_handles", None)
    before = trace.get_tracer_provider()

    handles = configure_observability(settings=ObservabilitySettings(otel_sdk_disabled=True))

    assert configure_observability() is handles
    assert trace.get_tracer_provider() is before  # nada global foi registrado
    assert isinstance(handles.tracer_provider, trace.NoOpTracerProvider)
    assert handles.logger_provider is None
    assert handles.telemetry_store.get_trace("0" * 32) == []
    assert handles.telemetry_store.summary().llm_calls == 0
    handles.shutdown()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, {"traces": False, "metrics": False, "logs": False}),
        (
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"},
            {"traces": True, "metrics": True, "logs": True},
        ),
        (  # Langfuse: só traces
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://cloud.langfuse.com/api/public/otel/v1/traces",
                "OTEL_METRICS_EXPORTER": "none",
                "OTEL_LOGS_EXPORTER": "none",
            },
            {"traces": True, "metrics": False, "logs": False},
        ),
        (
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
                "OTEL_TRACES_EXPORTER": "none",
            },
            {"traces": False, "metrics": True, "logs": True},
        ),
    ],
)
def test_otlp_exporters_follow_standard_env_vars(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: dict[str, bool]
) -> None:
    for signal in ("TRACES", "METRICS", "LOGS"):
        monkeypatch.delenv(f"OTEL_{signal}_EXPORTER", raising=False)
        monkeypatch.delenv(f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert {s: otel_setup.otlp_enabled(s) for s in expected} == expected
