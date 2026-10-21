"""Spans OTel em SQLite: o exportador grava, o store consulta (TelemetryStorePort).

Fica ao lado do exportador OTLP: os mesmos spans que vão para Langfuse/qualquer backend também
alimentam a governança local. Atributos de conteúdo (prompts/respostas, só existem quando o
modo `content` do SDK é ligado) nunca são gravados aqui.
"""

import json
import logging
import math
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from doc_read_ai.domain.config import ModelPricing
from doc_read_ai.domain.ports import TelemetryStorePort
from doc_read_ai.domain.telemetry import AgentUsage, SpanRecord, StepLatency, TelemetrySummary

logger = logging.getLogger(__name__)

# Escopo de instrumentação dos spans da aplicação (trace.get_tracer("doc_read_ai")).
APP_SCOPE = "doc_read_ai"

_CONTENT_ATTRIBUTES = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    }
)

# A PK (trace_id, span_id) já serve de índice por trace_id.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    scope TEXT,
    start_ns INTEGER NOT NULL,
    end_ns INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    attributes TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS ix_spans_start ON spans (start_ns);
"""


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA synchronous=NORMAL")  # com WAL: sem fsync por commit
    return conn


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Conexão curta, para leituras e para criar o schema."""
    with closing(_open(path)) as conn, conn:
        yield conn


def _init_db(path: Path) -> Path:
    path = path.resolve()  # o cwd pode mudar depois (testes, servidores)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
    return path


class SqliteSpanExporter(SpanExporter):
    """Grava spans finalizados no SQLite (WAL).

    Uma conexão persistente protegida por lock (os processors exportam de outras threads):
    abrir e fechar por export força checkpoint do WAL e custa dezenas de ms por span.
    """

    def __init__(self, path: Path) -> None:
        self._path = _init_db(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = _open(self._path)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = [_row(span) for span in spans if span.context is not None]
        with self._lock:
            if self._conn is None:
                return SpanExportResult.FAILURE
            try:
                with self._conn:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
            except sqlite3.Error:
                logger.exception("Falha ao gravar spans em %s", self._path)
                return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _row(span: ReadableSpan) -> tuple[Any, ...]:
    ctx = span.context
    start, end = span.start_time or 0, span.end_time or span.start_time or 0
    attributes = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in (span.attributes or {}).items()
        if k not in _CONTENT_ATTRIBUTES
    }
    return (
        format(ctx.trace_id, "032x"),
        format(ctx.span_id, "016x"),
        format(span.parent.span_id, "016x") if span.parent else None,
        span.name,
        span.kind.name,
        span.status.status_code.name,
        span.instrumentation_scope.name if span.instrumentation_scope else None,
        start,
        end,
        (end - start) / 1e6,
        json.dumps(attributes, ensure_ascii=False, default=str),
    )


class SqliteTelemetryStore(TelemetryStorePort):
    """Consulta os spans gravados pelo SqliteSpanExporter (mesmo arquivo)."""

    def __init__(self, path: Path, pricing: dict[str, ModelPricing] | None = None) -> None:
        self._path = _init_db(path)
        self._pricing = pricing or {}

    def get_trace(self, trace_id: str) -> list[SpanRecord]:
        # Empate de início (relógio grosso no Windows): pai antes do filho. O pai termina depois
        # e é gravado depois (rowid maior).
        with _connect(self._path) as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ns, end_ns DESC, rowid DESC",
                (trace_id.lower(),),
            ).fetchall()
        return [
            SpanRecord(
                trace_id=r["trace_id"],
                span_id=r["span_id"],
                parent_span_id=r["parent_span_id"],
                name=r["name"],
                kind=r["kind"],
                status=r["status"],
                start_time=_to_datetime(r["start_ns"]),
                end_time=_to_datetime(r["end_ns"]),
                duration_ms=r["duration_ms"],
                attributes=json.loads(r["attributes"]),
            )
            for r in rows
        ]

    def summary(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> TelemetrySummary:
        """Agrega só os spans da aplicação: `invoke_agent *` (LLM) e `doc_read.step *` (nós)."""
        query = (
            "SELECT name, status, duration_ms, attributes FROM spans WHERE scope = ? "
            "AND (name LIKE 'invoke_agent %' OR name LIKE 'doc_read.step %')"
        )
        params: list[Any] = [APP_SCOPE]
        if since is not None:
            query += " AND start_ns >= ?"
            params.append(_to_ns(since))
        if until is not None:
            query += " AND start_ns <= ?"
            params.append(_to_ns(until))
        with _connect(self._path) as conn:
            rows = conn.execute(query, params).fetchall()

        agents: dict[tuple[str, str, str], list[tuple[float, bool, dict[str, Any]]]]
        agents = defaultdict(list)
        steps: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for r in rows:
            attrs, failed = json.loads(r["attributes"]), r["status"] == "ERROR"
            if r["name"].startswith("invoke_agent "):
                key = (
                    str(attrs.get("gen_ai.agent.name", r["name"].removeprefix("invoke_agent "))),
                    str(attrs.get("gen_ai.provider.name", "")),
                    str(attrs.get("gen_ai.request.model", "")),
                )
                agents[key].append((r["duration_ms"], failed, attrs))
            else:
                step = str(attrs.get("doc_read.step", r["name"].removeprefix("doc_read.step ")))
                steps[step].append((r["duration_ms"], failed))

        by_agent = [self._agent_usage(key, spans) for key, spans in sorted(agents.items())]
        by_step = [
            StepLatency(
                step=step,
                count=len(spans),
                errors=sum(failed for _, failed in spans),
                avg_ms=_avg([d for d, _ in spans]),
                p95_ms=_p95([d for d, _ in spans]),
            )
            for step, spans in sorted(steps.items())
        ]
        return TelemetrySummary(
            since=since,
            until=until,
            llm_calls=sum(a.calls for a in by_agent),
            input_tokens=sum(a.input_tokens for a in by_agent),
            output_tokens=sum(a.output_tokens for a in by_agent),
            cost_usd=round(sum(a.cost_usd for a in by_agent), 6),
            by_agent=by_agent,
            by_step=by_step,
        )

    def _agent_usage(
        self, key: tuple[str, str, str], spans: list[tuple[float, bool, dict[str, Any]]]
    ) -> AgentUsage:
        agent, provider, model = key
        input_tokens = output_tokens = 0
        cost = 0.0
        for _, _, attrs in spans:
            tokens_in = int(attrs.get("doc_read.usage.input_tokens", 0))
            tokens_out = int(attrs.get("doc_read.usage.output_tokens", 0))
            input_tokens += tokens_in
            output_tokens += tokens_out
            # O modelo que respondeu (fallback do servidor) vence o pedido, se tiver preço.
            price = self._pricing.get(attrs.get("gen_ai.response.model", ""))
            price = price or self._pricing.get(model)
            if price:
                cost += tokens_in * price.input_per_mtok / 1e6
                cost += tokens_out * price.output_per_mtok / 1e6
        durations = [d for d, _, _ in spans]
        return AgentUsage(
            agent=agent,
            provider=provider,
            model=model,
            calls=len(spans),
            errors=sum(failed for _, failed, _ in spans),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            avg_latency_ms=_avg(durations),
            p95_latency_ms=_p95(durations),
        )


def _to_datetime(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC)


def _to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)  # datetime ingênuo = UTC
    return int(value.timestamp() * 1e9)


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _p95(values: list[float]) -> float:
    """Percentil 95 pelo método nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 3)
