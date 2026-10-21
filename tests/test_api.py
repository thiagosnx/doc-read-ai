"""API HTTP com container injetado: LLM falso, repositórios locais, telemetria falsa, sem rede."""

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from doc_read_ai.bootstrap import build_use_case, load_config
from doc_read_ai.domain.analysis import FieldType
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.ports import RunRepositoryPort, TelemetryStorePort
from doc_read_ai.domain.runs import PipelineEvent, Run, RunStatus
from doc_read_ai.domain.telemetry import AgentUsage, SpanRecord, StepLatency, TelemetrySummary
from doc_read_ai.infrastructure.loaders import LocalFileLoader
from doc_read_ai.infrastructure.persistence import InMemoryRunRepository, SqliteRunRepository
from doc_read_ai.infrastructure.registry import InMemorySchemaRegistry
from doc_read_ai.interfaces.api import ApiContainer, ApiSettings, create_app
from doc_read_ai.interfaces.api.jobs import RESTART_ERROR
from doc_read_ai.interfaces.api.routes import runs as runs_routes
from tests.conftest import field
from tests.fakes import FakeLLMFactory, pdf_bytes, png_bytes

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
STEPS = ["validate", "classify", "resolve_schema", "design_schema", "build_schema", "extract"]
OTEL_VARS = (
    "OTEL_SDK_DISABLED",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_TRACES_EXPORTER",
    "OTEL_METRICS_EXPORTER",
    "OTEL_LOGS_EXPORTER",
    "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE",
)


class FakeTelemetryStore(TelemetryStorePort):
    def __init__(self) -> None:
        self.spans: dict[str, list[SpanRecord]] = {}

    def get_trace(self, trace_id: str) -> list[SpanRecord]:
        return self.spans.get(trace_id, [])

    def summary(self, since: datetime | None = None, until: datetime | None = None):
        return TelemetrySummary(
            since=since,
            until=until,
            llm_calls=4,
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.0002,
            by_agent=[
                AgentUsage(
                    agent="extractor",
                    provider="anthropic",
                    model="claude-haiku-5-5",
                    calls=1,
                    errors=0,
                    input_tokens=400,
                    output_tokens=100,
                    cost_usd=0.0001,
                    avg_latency_ms=900,
                    p95_latency_ms=900,
                )
            ],
            by_step=[StepLatency(step="extract", count=1, errors=0, avg_ms=900, p95_ms=900)],
        )


class GatedUseCase:
    """Emite o primeiro evento e espera o `gate` (threading.Event) para seguir."""

    def __init__(self, inner: Any, gate: threading.Event) -> None:
        self._inner = inner
        self._gate = gate

    async def stream_document(
        self, document: Document, run_id: str | None = None
    ) -> AsyncIterator[PipelineEvent]:
        first = True
        async for event in self._inner.stream_document(document, run_id):
            yield event
            if first:
                first = False
                while not self._gate.is_set():
                    await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OTEL_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def telemetry() -> FakeTelemetryStore:
    return FakeTelemetryStore()


@pytest.fixture
def make_client(
    config_path: Path, nfe_factory: FakeLLMFactory, telemetry: FakeTelemetryStore
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(
        *,
        runs: RunRepositoryPort | None = None,
        use_case: Any = "default",
        startup_error: str | None = None,
        **settings: Any,
    ) -> TestClient:
        registry = InMemorySchemaRegistry()
        if use_case == "default":
            use_case = build_use_case(config_path, nfe_factory, LocalFileLoader(), registry)
        values = {
            "doc_read_api_key": None,
            "doc_read_cors_origins": ["http://localhost:5173"],
            "doc_read_runs_db": Path("unused.db"),
            "doc_read_store_results": True,
            "doc_read_max_upload_mb": 1,
            "doc_read_concurrency": 2,
            **settings,
        }
        container = ApiContainer(
            settings=ApiSettings(_env_file=None, **values),
            app_config=load_config(config_path),
            runs=runs or InMemoryRunRepository(),
            schema_registry=registry,
            use_case=use_case,
            telemetry=telemetry,
            startup_error=startup_error,
        )
        client = TestClient(create_app(container))
        client.__enter__()  # roda o lifespan (runner + recuperação)
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def upload(client: TestClient, *files: tuple[str, bytes], **params: Any):
    return client.post(
        "/v1/documents",
        files=[("files", (name, content, "application/octet-stream")) for name, content in files],
        params=params,
    )


def wait_final(client: TestClient, run_id: str, timeout: float = 5) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] in ("extracted", "rejected", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} não terminou")


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        if not lines:
            continue
        kind = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((kind, json.loads(data)))
    return events


# --- upload e execução --------------------------------------------------------------------


def test_upload_with_wait_returns_extracted_run(client: TestClient) -> None:
    response = upload(client, ("nota.png", png_bytes()), wait="true")

    assert response.status_code == 200, response.text
    [run] = response.json()["runs"]
    assert run["status"] == "extracted"
    assert run["filename"] == "nota.png" and run["media_type"] == "image/png"
    assert run["document_type"] == "nota_fiscal_eletronica"
    assert run["schema_source"] == "designed"
    assert run["result"]["data"]["numero"] == "000123"
    assert len(run["sha256"]) == 64 and run["duration_ms"] is not None

    stored = client.get(f"/v1/runs/{run['id']}").json()
    assert stored["result"]["data"]["emitente"]["cnpj"] == "12.345.678/0001-90"


def test_upload_returns_202_and_polling_reaches_extracted(client: TestClient) -> None:
    response = upload(client, ("nota.png", png_bytes()))

    assert response.status_code == 202, response.text
    [run] = response.json()["runs"]
    assert run["status"] == "queued" and run["result"] is None

    final = wait_final(client, run["id"])
    assert final["status"] == "extracted"
    assert final["result"]["data"]["valor_total"] == 150.5

    page = client.get("/v1/runs").json()
    assert page["limit"] == 50 and page["offset"] == 0
    assert [r["id"] for r in page["items"]] == [run["id"]]
    assert "result" not in page["items"][0]


def test_multiple_files(client: TestClient) -> None:
    response = upload(client, ("nota.png", png_bytes()), ("outra.pdf", pdf_bytes()), wait="true")

    assert response.status_code == 200, response.text
    runs = response.json()["runs"]
    assert [r["filename"] for r in runs] == ["nota.png", "outra.pdf"]
    assert [r["media_type"] for r in runs] == ["image/png", "application/pdf"]
    assert all(r["status"] == "extracted" for r in runs)
    # o segundo do mesmo tipo pode reaproveitar o schema do primeiro: ambos válidos
    assert {r["schema_source"] for r in runs} <= {"designed", "registry"}


def test_unsupported_file_is_rejected_without_creating_runs(client: TestClient) -> None:
    response = upload(client, ("nota.txt", b"so texto"))
    assert response.status_code == 415
    assert "não é um formato suportado" in response.json()["detail"]

    mixed = upload(client, ("nota.png", png_bytes()), ("nota.txt", b"so texto"))
    assert mixed.status_code == 415
    assert client.get("/v1/runs").json()["items"] == []


def test_upload_limits(client: TestClient) -> None:
    too_big = png_bytes() + b"\x00" * (1024 * 1024)
    response = upload(client, ("grande.png", too_big))
    assert response.status_code == 413
    assert upload(client, ("vazio.png", b"")).status_code == 422
    assert client.post("/v1/documents").status_code == 422
    assert client.get("/v1/runs").json()["items"] == []


def test_filename_from_client_has_no_directories(client: TestClient) -> None:
    response = upload(client, ("../../etc/nota.png", png_bytes()), wait="true")
    assert response.json()["runs"][0]["filename"] == "nota.png"


def test_store_results_false_keeps_only_metadata(make_client) -> None:
    client = make_client(doc_read_store_results=False)
    response = upload(client, ("nota.png", png_bytes()), wait="true")

    [run] = response.json()["runs"]
    assert run["result"]["data"]["numero"] == "000123"  # entregue a quem pediu...
    stored = client.get(f"/v1/runs/{run['id']}").json()
    assert stored["status"] == "extracted"
    assert stored["result"]["data"] is None  # ...mas não guardado
    assert stored["result"]["analysis"]["summary"] == ""
    assert stored["document_type"] == "nota_fiscal_eletronica"


def test_pipeline_unavailable_returns_503(make_client) -> None:
    client = make_client(use_case=None, startup_error="Chave ANTHROPIC_API_KEY ausente")
    response = upload(client, ("nota.png", png_bytes()))
    assert response.status_code == 503
    assert response.json() == {"detail": "Chave ANTHROPIC_API_KEY ausente"}

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["checks"]["pipeline"]["ok"] is False


# --- SSE ----------------------------------------------------------------------------------


def test_sse_streams_steps_in_order_then_done(
    make_client, config_path: Path, nfe_factory: FakeLLMFactory
) -> None:
    gate = threading.Event()
    inner = build_use_case(config_path, nfe_factory, LocalFileLoader(), InMemorySchemaRegistry())
    client = make_client(use_case=GatedUseCase(inner, gate))

    [run] = upload(client, ("nota.png", png_bytes())).json()["runs"]
    timer = threading.Timer(0.3, gate.set)  # libera depois que o stream já assinou
    timer.start()
    try:
        with client.stream("GET", f"/v1/runs/{run['id']}/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            text = response.read().decode()
    finally:
        timer.cancel()
        gate.set()

    events = parse_sse(text)
    assert [kind for kind, _ in events] == ["step"] * len(STEPS) + ["done"]
    assert [data["step"] for _, data in events[:-1]] == STEPS  # 'validate' veio do replay
    assert all(data["run_id"] == run["id"] for _, data in events[:-1])
    assert events[1][1]["detail"]["document_type"] == "nota_fiscal_eletronica"
    done = events[-1][1]
    assert done["status"] == "extracted" and done["result"]["data"]["numero"] == "000123"


def test_sse_for_finished_run_sends_single_done(client: TestClient) -> None:
    [run] = upload(client, ("nota.png", png_bytes()), wait="true").json()["runs"]
    response = client.get(f"/v1/runs/{run['id']}/events")
    events = parse_sse(response.text)
    assert [kind for kind, _ in events] == ["done"]
    assert events[0][1]["id"] == run["id"]
    assert client.get("/v1/runs/nao-existe/events").status_code == 404


async def test_sse_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem eventos por HEARTBEAT_SECONDS, o stream manda um comentário `: ping`."""
    from doc_read_ai.interfaces.api.jobs import JobEvent, JobRunner

    monkeypatch.setattr(runs_routes, "HEARTBEAT_SECONDS", 0.01)
    runner = JobRunner(lambda: None, InMemoryRunRepository())
    runner._subscribers["r1"] = set()
    runner._history["r1"] = []

    class FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    response = await runs_routes.stream_run_events(FakeRequest(), None, runner, "r1")
    chunks = response.body_iterator
    assert await anext(chunks) == ": ping\n\n"
    runner._publish("r1", JobEvent("done", "{}"))
    assert await anext(chunks) == "event: done\ndata: {}\n\n"


# --- runs ---------------------------------------------------------------------------------


def test_delete_run(make_client) -> None:
    repo = InMemoryRunRepository()
    client = make_client(runs=repo)
    [run] = upload(client, ("nota.png", png_bytes()), wait="true").json()["runs"]

    assert client.delete(f"/v1/runs/{run['id']}").status_code == 204
    assert client.get(f"/v1/runs/{run['id']}").status_code == 404
    assert client.delete(f"/v1/runs/{run['id']}").status_code == 404

    # em andamento (salvo depois do startup, então não é recuperado)
    running = Run(
        id="em-andamento",
        filename="x.png",
        media_type="image/png",
        size_bytes=1,
        sha256="0" * 64,
        status=RunStatus.RUNNING,
        created_at=datetime.now(UTC),
    )
    repo.save(running)
    conflict = client.delete("/v1/runs/em-andamento")
    assert conflict.status_code == 409 and "andamento" in conflict.json()["detail"]


def test_list_filters_and_validation(client: TestClient) -> None:
    upload(client, ("nota.png", png_bytes()), wait="true")
    assert len(client.get("/v1/runs", params={"status": "extracted"}).json()["items"]) == 1
    assert client.get("/v1/runs", params={"status": "failed"}).json()["items"] == []
    by_type = client.get("/v1/runs", params={"document_type": "curriculo"}).json()
    assert by_type["items"] == []

    invalid = client.get("/v1/runs", params={"limit": 0})
    assert invalid.status_code == 422
    assert isinstance(invalid.json()["detail"], str) and "limit" in invalid.json()["detail"]


def test_trace_endpoints(make_client, telemetry: FakeTelemetryStore) -> None:
    repo = InMemoryRunRepository()
    client = make_client(runs=repo)
    now = datetime.now(UTC)
    span = SpanRecord(
        trace_id=TRACE_ID,
        span_id="b7ad6b7169203331",
        parent_span_id=None,
        name="doc_read.process_document",
        kind="INTERNAL",
        status="OK",
        start_time=now,
        end_time=now + timedelta(seconds=1),
        duration_ms=1000,
    )
    telemetry.spans[TRACE_ID] = [span]
    base = {
        "filename": "x.png",
        "media_type": "image/png",
        "size_bytes": 1,
        "sha256": "0" * 64,
        "status": RunStatus.EXTRACTED,
        "created_at": now,
    }
    repo.save(Run(id="com-trace", trace_id=TRACE_ID, **base))
    repo.save(Run(id="sem-trace", **base))

    traced = client.get("/v1/runs/com-trace/trace")
    assert traced.status_code == 200
    assert traced.json()["spans"][0]["name"] == "doc_read.process_document"
    assert client.get("/v1/runs/sem-trace/trace").status_code == 404
    assert client.get(f"/v1/observability/traces/{TRACE_ID}").status_code == 200
    assert client.get(f"/v1/observability/traces/{'1' * 32}").status_code == 404
    assert client.get("/v1/observability/traces/nao-hex").status_code == 422


def test_restart_recovery_marks_stale_runs_failed(make_client, tmp_path: Path) -> None:
    db = tmp_path / "runs.db"
    previous = SqliteRunRepository(db)
    base = {
        "filename": "x.png",
        "media_type": "image/png",
        "size_bytes": 1,
        "sha256": "0" * 64,
        "created_at": datetime.now(UTC),
    }
    previous.save(Run(id="queued", status=RunStatus.QUEUED, **base))
    previous.save(
        Run(id="running", status=RunStatus.RUNNING, started_at=base["created_at"], **base)
    )
    previous.save(Run(id="done", status=RunStatus.EXTRACTED, **base))
    previous.close()

    repo = SqliteRunRepository(db)
    try:
        client = make_client(runs=repo)
        for run_id in ("queued", "running"):
            body = client.get(f"/v1/runs/{run_id}").json()
            assert body["status"] == "failed"
            assert body["errors"] == [RESTART_ERROR]
            assert body["finished_at"] is not None
        assert client.get("/v1/runs/done").json()["status"] == "extracted"
    finally:
        repo.close()


# --- schemas ------------------------------------------------------------------------------


def test_schemas_crud(client: TestClient) -> None:
    assert client.get("/v1/schemas").json() == []
    body = {
        "document_type_label": "Recibo",
        "category": "financeiro",
        "fields": [
            field("valor", FieldType.NUMBER).model_dump(mode="json"),
            field("pagador", FieldType.STRING, required=False).model_dump(mode="json"),
        ],
    }
    created = client.put("/v1/schemas/recibo", json=body)
    assert created.status_code == 200, created.text
    assert created.json()["json_schema"]["properties"]["valor"]
    created_at = created.json()["created_at"]

    [summary] = client.get("/v1/schemas").json()
    assert summary == {
        "document_type": "recibo",
        "document_type_label": "Recibo",
        "category": "financeiro",
        "field_count": 2,
        "created_at": created_at,
    }

    replaced = client.put("/v1/schemas/recibo", json={**body, "document_type_label": "Recibo v2"})
    assert replaced.json()["document_type_label"] == "Recibo v2"
    assert replaced.json()["created_at"] == created_at  # substituir mantém a data de criação
    assert client.get("/v1/schemas/recibo").json()["document_type_label"] == "Recibo v2"

    assert client.delete("/v1/schemas/recibo").status_code == 204
    assert client.get("/v1/schemas/recibo").status_code == 404
    assert client.delete("/v1/schemas/recibo").status_code == 404


def test_invalid_schema_put_is_422(client: TestClient) -> None:
    empty = client.put(
        "/v1/schemas/recibo",
        json={"document_type_label": "Recibo", "category": "financeiro", "fields": []},
    )
    assert empty.status_code == 422
    assert "nenhum campo" in empty.json()["detail"]  # mensagem do SchemaBuildError
    assert client.get("/v1/schemas/recibo").status_code == 404

    bad_type = client.put(
        "/v1/schemas/recibo",
        json={"document_type_label": "R", "category": "c", "fields": [{"name": "x"}]},
    )
    assert bad_type.status_code == 422 and isinstance(bad_type.json()["detail"], str)
    bad_id = client.put(
        "/v1/schemas/Nota-Fiscal",
        json={"document_type_label": "R", "category": "c", "fields": []},
    )
    assert bad_id.status_code == 422


# --- config e governança ------------------------------------------------------------------


def test_agents_endpoint_has_no_secrets(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-segredo-123")
    response = client.get("/v1/agents")

    assert response.status_code == 200
    body = response.json()
    assert [a["name"] for a in body["agents"]] == [
        "orchestrator",
        "classifier",
        "analyst",
        "extractor",
    ]
    assert body["agents"][0]["provider"] == "anthropic"
    assert body["agents"][0]["params"] == {"effort": "low"}
    assert body["providers"] == [
        {
            "name": "anthropic",
            "timeout_seconds": 300.0,
            "max_retries": 3,
            "credentials_configured": True,
        }
    ]
    assert body["schema_registry"] == {"enabled": True, "count": 0}
    assert "sk-segredo-123" not in response.text
    assert "ANTHROPIC_API_KEY" not in response.text and "api_key" not in response.text


def test_observability_summary(client: TestClient, monkeypatch) -> None:
    upload(client, ("nota.png", png_bytes()), wait="true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")

    response = client.get("/v1/observability/summary", params={"since": "2020-01-01T00:00:00Z"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"period", "runs", "telemetry", "agents", "schema_registry", "governance"}
    assert body["period"]["since"].startswith("2020-01-01") and body["period"]["until"] is None
    assert body["runs"]["total"] == 1 and body["runs"]["by_status"] == {"extracted": 1}
    assert body["runs"]["by_schema_source"] == {"designed": 1}
    assert body["telemetry"]["llm_calls"] == 4
    assert body["schema_registry"] == {"enabled": True, "count": 1}
    assert len(body["agents"]) == 4
    assert body["governance"] == {
        "raw_files_stored": False,
        "results_stored": True,
        "llm_content_capture": "metadata_only",
        "otlp_export": {"traces": True, "metrics": False, "logs": True},
        "api_auth_enabled": False,
    }


# --- segurança e saúde --------------------------------------------------------------------


def test_api_key_enforced_on_v1_but_not_on_health(make_client) -> None:
    client = make_client(doc_read_api_key="s3cret")

    denied = client.get("/v1/runs")
    assert denied.status_code == 401
    assert denied.json() == {"detail": "X-API-Key ausente ou inválida"}
    assert client.get("/v1/runs", headers={"X-API-Key": "errada"}).status_code == 401
    assert client.get("/v1/runs", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert upload(client, ("nota.png", png_bytes())).status_code == 401

    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    summary = client.get("/v1/observability/summary", headers={"X-API-Key": "s3cret"})
    assert summary.json()["governance"]["api_auth_enabled"] is True


def test_cors_preflight(make_client) -> None:
    client = make_client(doc_read_api_key="s3cret")
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key",
    }
    allowed = client.options("/v1/documents", headers=headers)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "x-api-key" in allowed.headers["access-control-allow-headers"].lower()

    blocked = client.options("/v1/documents", headers={**headers, "Origin": "https://evil.test"})
    assert "access-control-allow-origin" not in blocked.headers


def test_openapi_is_typed(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    operation = spec["paths"]["/v1/runs/{run_id}"]["get"]
    assert operation["operationId"] == "get_run"
    assert operation["tags"] == ["runs"]
    assert {"RunOut", "RunPage", "StepEventOut", "ErrorOut"} <= set(spec["components"]["schemas"])
    assert "ApiKey" in spec["components"]["securitySchemes"]
    assert (
        "text/event-stream"
        in (spec["paths"]["/v1/runs/{run_id}/events"]["get"]["responses"]["200"]["content"])
    )


def test_settings_parse_cors_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DOC_READ_CORS_ORIGINS", "https://a.test, https://b.test")
    monkeypatch.setenv("DOC_READ_API_KEY", "  ")
    settings = ApiSettings(_env_file=None)
    assert settings.doc_read_cors_origins == ["https://a.test", "https://b.test"]
    assert settings.api_auth_enabled is False
    monkeypatch.setenv("DOC_READ_CORS_ORIGINS", '["https://c.test"]')
    assert ApiSettings(_env_file=None).doc_read_cors_origins == ["https://c.test"]
