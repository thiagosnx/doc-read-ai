"""Adapters de RunRepositoryPort: SQLite (padrão da API) e memória (testes/efêmero).

Guardam metadados e resultado da execução, nunca o arquivo enviado (só o sha256).
"""

import json
import math
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from doc_read_ai.domain.analysis import ExtractionResult
from doc_read_ai.domain.ports import RunRepositoryPort
from doc_read_ai.domain.runs import Run, RunStats, RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    media_type    TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    sha256        TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    duration_ms   REAL,
    trace_id      TEXT,
    document_type TEXT,
    schema_source TEXT,
    result_json   TEXT,
    errors_json   TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_document_type ON runs (document_type);
"""

_COLUMNS = (
    "id, filename, media_type, size_bytes, sha256, status, created_at, started_at, "
    "finished_at, duration_ms, trace_id, document_type, schema_source, result_json, errors_json"
)


def _iso(value: datetime | None) -> str | None:
    """UTC com microssegundos fixos: a comparação de texto no SQLite vira cronológica."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _in_period(run: Run, since: datetime | None, until: datetime | None) -> bool:
    created = _iso(run.created_at)
    return (since is None or created >= _iso(since)) and (until is None or created < _iso(until))


def _percentile(values: list[float], pct: float) -> float | None:
    """Percentil por nearest-rank (sem interpolação)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def compute_stats(runs: Iterable[Run]) -> RunStats:
    """Agregados de governança; compartilhado pelos dois adapters para darem o mesmo número."""
    runs = list(runs)
    durations = [d for r in runs if (d := r.duration_ms) is not None]
    return RunStats(
        total=len(runs),
        by_status=dict(Counter(r.status.value for r in runs)),
        by_document_type=dict(Counter(r.document_type for r in runs if r.document_type)),
        by_schema_source=dict(Counter(r.schema_source.value for r in runs if r.schema_source)),
        avg_duration_ms=sum(durations) / len(durations) if durations else None,
        p95_duration_ms=_percentile(durations, 95),
    )


class SqliteRunRepository(RunRepositoryPort):
    """Um arquivo SQLite (WAL). Uma conexão protegida por lock: seguro entre threads."""

    def __init__(self, path: Path = Path("data/runs.db")) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)

    def save(self, run: Run) -> None:
        result = run.result.model_dump_json() if run.result else None
        values = (
            run.id,
            run.filename,
            run.media_type,
            run.size_bytes,
            run.sha256,
            run.status.value,
            _iso(run.created_at),
            _iso(run.started_at),
            _iso(run.finished_at),
            run.duration_ms,
            run.trace_id,
            run.document_type,
            run.schema_source.value if run.schema_source else None,
            result,
            json.dumps(run.errors, ensure_ascii=False),
        )
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO runs ({_COLUMNS}) VALUES ({', '.join('?' * 15)})",
                values,
            )

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._to_run(row) if row else None

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: RunStatus | None = None,
        document_type: str | None = None,
    ) -> list[Run]:
        where, params = [], []
        if status is not None:
            where.append("status = ?")
            params.append(RunStatus(status).value)
        if document_type is not None:
            where.append("document_type = ?")
            params.append(document_type)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {_COLUMNS} FROM runs {clause} "
            "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (*params, limit, offset)).fetchall()
        return [self._to_run(r) for r in rows]

    def stats(self, since: datetime | None = None, until: datetime | None = None) -> RunStats:
        where, params = [], []
        if since is not None:
            where.append("created_at >= ?")
            params.append(_iso(since))
        if until is not None:
            where.append("created_at < ?")
            params.append(_iso(until))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        # Sem result_json: agregados não precisam do resultado (nem dos dados extraídos).
        sql = (
            "SELECT id, filename, media_type, size_bytes, sha256, status, created_at, started_at, "
            "finished_at, duration_ms, trace_id, document_type, schema_source, "
            f"NULL AS result_json, '[]' AS errors_json FROM runs {clause}"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return compute_stats(self._to_run(r) for r in rows)

    def delete(self, run_id: str) -> bool:
        """Apaga a execução (direito de eliminação, LGPD). Não está na porta ainda."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0

    def ping(self) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1").fetchone()[0] == 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _to_run(row: sqlite3.Row) -> Run:
        return Run(
            id=row["id"],
            filename=row["filename"],
            media_type=row["media_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            status=RunStatus(row["status"]),
            created_at=_parse(row["created_at"]),
            started_at=_parse(row["started_at"]),
            finished_at=_parse(row["finished_at"]),
            trace_id=row["trace_id"],
            document_type=row["document_type"],
            schema_source=row["schema_source"],
            result=(
                ExtractionResult.model_validate_json(row["result_json"])
                if row["result_json"]
                else None
            ),
            errors=json.loads(row["errors_json"]),
        )


class InMemoryRunRepository(RunRepositoryPort):
    """Histórico que dura só o processo (testes ou uso sem persistência)."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def save(self, run: Run) -> None:
        with self._lock:
            self._runs[run.id] = run.model_copy(deep=True)

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            run = self._runs.get(run_id)
        return run.model_copy(deep=True) if run else None

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: RunStatus | None = None,
        document_type: str | None = None,
    ) -> list[Run]:
        with self._lock:
            # dict preserva a ordem de inserção: o desempate é a ordem em que foram criadas.
            runs = list(self._runs.values())
        runs.reverse()
        runs.sort(key=lambda r: _iso(r.created_at), reverse=True)
        selected = [
            r
            for r in runs
            if (status is None or r.status == status)
            and (document_type is None or r.document_type == document_type)
        ]
        return [r.model_copy(deep=True) for r in selected[offset : offset + limit]]

    def stats(self, since: datetime | None = None, until: datetime | None = None) -> RunStats:
        with self._lock:
            runs = [r for r in self._runs.values() if _in_period(r, since, until)]
        return compute_stats(runs)

    def delete(self, run_id: str) -> bool:
        with self._lock:
            return self._runs.pop(run_id, None) is not None

    def ping(self) -> bool:
        return True
