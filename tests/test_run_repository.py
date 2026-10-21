"""Repositórios de execuções (SQLite e memória) com o mesmo contrato."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from doc_read_ai.domain.analysis import ExtractionResult, PipelineStatus, SchemaSource
from doc_read_ai.domain.ports import RunRepositoryPort
from doc_read_ai.domain.runs import Run, RunStatus
from doc_read_ai.infrastructure.persistence import InMemoryRunRepository, SqliteRunRepository

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_run(
    run_id: str,
    minutes: int = 0,
    status: RunStatus = RunStatus.EXTRACTED,
    document_type: str | None = "nota_fiscal_eletronica",
    duration_s: float | None = 1.0,
    source: SchemaSource | None = SchemaSource.REGISTRY,
) -> Run:
    created = T0 + timedelta(minutes=minutes)
    started = created if duration_s is not None else None
    finished = created + timedelta(seconds=duration_s) if duration_s is not None else None
    result = None
    if status == RunStatus.EXTRACTED:
        result = ExtractionResult(
            filename=f"{run_id}.png",
            media_type="image/png",
            status=PipelineStatus.EXTRACTED,
            data={"numero": run_id, "valor_total": 10.5},
            trace_id="a" * 32,
        )
    return Run(
        id=run_id,
        filename=f"{run_id}.png",
        media_type="image/png",
        size_bytes=123,
        sha256="f" * 64,
        status=status,
        created_at=created,
        started_at=started,
        finished_at=finished,
        trace_id="a" * 32 if result else None,
        document_type=document_type,
        schema_source=source,
        result=result,
        errors=[] if status != RunStatus.FAILED else ["[extract] LLMError: boom"],
    )


@pytest.fixture(params=["sqlite", "memory"])
def repo(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[RunRepositoryPort]:
    if request.param == "memory":
        yield InMemoryRunRepository()
        return
    sqlite_repo = SqliteRunRepository(tmp_path / "data" / "runs.db")
    yield sqlite_repo
    sqlite_repo.close()


def test_save_and_get_roundtrip(repo: RunRepositoryPort) -> None:
    run = make_run("r1")
    repo.save(run)
    loaded = repo.get("r1")
    assert loaded == run
    assert loaded.result.data == {"numero": "r1", "valor_total": 10.5}
    assert loaded.duration_ms == pytest.approx(1000)
    assert repo.get("nao-existe") is None


def test_save_is_upsert(repo: RunRepositoryPort) -> None:
    run = make_run("r1", status=RunStatus.QUEUED, duration_s=None, document_type=None)
    repo.save(run)
    repo.save(run.model_copy(update={"status": RunStatus.RUNNING}))
    assert repo.get("r1").status == RunStatus.RUNNING
    assert len(repo.list()) == 1


def test_list_newest_first_with_filters_and_pagination(repo: RunRepositoryPort) -> None:
    repo.save(make_run("a", minutes=0))
    repo.save(make_run("b", minutes=1, status=RunStatus.FAILED))
    repo.save(make_run("c", minutes=2, document_type="curriculo"))
    repo.save(make_run("d", minutes=3, status=RunStatus.REJECTED, document_type=None))

    assert [r.id for r in repo.list()] == ["d", "c", "b", "a"]
    assert [r.id for r in repo.list(limit=2, offset=1)] == ["c", "b"]
    assert [r.id for r in repo.list(status=RunStatus.FAILED)] == ["b"]
    assert [r.id for r in repo.list(document_type="nota_fiscal_eletronica")] == ["b", "a"]
    assert [r.id for r in repo.list(status=RunStatus.EXTRACTED, document_type="curriculo")] == ["c"]


def test_stats(repo: RunRepositoryPort) -> None:
    for i, duration in enumerate([1, 2, 3, 4, 10]):
        repo.save(make_run(f"ok{i}", minutes=i, duration_s=duration))
    repo.save(make_run("fail", minutes=5, status=RunStatus.FAILED, duration_s=20))
    repo.save(make_run("cv", minutes=6, document_type="curriculo", source=SchemaSource.DESIGNED))
    repo.save(
        make_run(
            "queued",
            minutes=7,
            status=RunStatus.QUEUED,
            duration_s=None,
            document_type=None,
            source=None,
        )
    )

    stats = repo.stats()
    assert stats.total == 8
    assert stats.by_status == {"extracted": 6, "failed": 1, "queued": 1}
    assert stats.by_document_type == {"nota_fiscal_eletronica": 6, "curriculo": 1}
    assert stats.by_schema_source == {"registry": 6, "designed": 1}
    # durações (s): 1,2,3,4,10,20,1 -> média 41/7; p95 nearest-rank = 20
    assert stats.avg_duration_ms == pytest.approx(41 / 7 * 1000)
    assert stats.p95_duration_ms == pytest.approx(20_000)

    window = repo.stats(since=T0 + timedelta(minutes=1), until=T0 + timedelta(minutes=5))
    assert window.total == 4  # ok1..ok4 (until é exclusivo)
    assert window.avg_duration_ms == pytest.approx((2 + 3 + 4 + 10) / 4 * 1000)

    empty = repo.stats(since=T0 + timedelta(days=1))
    assert empty.total == 0 and empty.avg_duration_ms is None and empty.p95_duration_ms is None


def test_delete(repo: RunRepositoryPort) -> None:
    repo.save(make_run("r1"))
    assert repo.delete("r1") is True
    assert repo.get("r1") is None
    assert repo.delete("r1") is False


def test_sqlite_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    first = SqliteRunRepository(path)
    first.save(make_run("r1"))
    first.close()

    second = SqliteRunRepository(path)
    try:
        assert second.get("r1").result.data["numero"] == "r1"
        assert second.ping()
    finally:
        second.close()
