"""Histórico de execuções, progresso ao vivo (SSE), trace e eliminação."""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from doc_read_ai.domain.runs import Run, RunStatus
from doc_read_ai.interfaces.api.deps import ContainerDep, RunnerDep
from doc_read_ai.interfaces.api.jobs import JobEvent
from doc_read_ai.interfaces.api.schemas import ErrorOut, RunOut, RunPage, RunSummaryOut, TraceOut

router = APIRouter(prefix="/runs", tags=["runs"])

HEARTBEAT_SECONDS = 15.0

SSE_DESCRIPTION = """Stream `text/event-stream` com o progresso da execução.

- `event: step` — `data` é um `StepEventOut` (um por nó do grafo concluído);
- `event: done` — `data` é o `RunOut` final; o servidor fecha o stream em seguida;
- linhas `: ping` a cada ~15 s mantêm a conexão viva (ignoradas pelo EventSource).

Quem conecta no meio recebe antes os `step` já emitidos. Se a execução já terminou, vem só
o `done`."""


async def _get_run(container: ContainerDep, run_id: str) -> Run:
    run = await asyncio.to_thread(container.runs.get, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Execução '{run_id}' não encontrada")
    return run


def _sse(event: JobEvent) -> str:
    return f"event: {event.kind}\ndata: {event.data}\n\n"


@router.get("", summary="Lista execuções (mais recentes primeiro)")
async def list_runs(
    container: ContainerDep,
    status_filter: Annotated[RunStatus | None, Query(alias="status")] = None,
    document_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunPage:
    runs = await asyncio.to_thread(
        container.runs.list,
        limit=limit,
        offset=offset,
        status=status_filter,
        document_type=document_type,
    )
    return RunPage(items=[RunSummaryOut.from_run(r) for r in runs], limit=limit, offset=offset)


@router.get("/{run_id}", summary="Detalhe de uma execução, com o resultado")
async def get_run(container: ContainerDep, run_id: str) -> RunOut:
    return RunOut.from_run(await _get_run(container, run_id))


@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Apaga a execução e o resultado (eliminação, LGPD)",
    responses={409: {"model": ErrorOut, "description": "Execução ainda em andamento"}},
)
async def delete_run(container: ContainerDep, runner: RunnerDep, run_id: str) -> None:
    run = await _get_run(container, run_id)
    if runner.is_active(run_id) or not run.status.is_final:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "A execução ainda está em andamento; tente quando terminar"
        )
    await asyncio.to_thread(container.runs.delete, run_id)


@router.get(
    "/{run_id}/events",
    summary="Progresso ao vivo (Server-Sent Events)",
    description=SSE_DESCRIPTION,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Stream SSE",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def stream_run_events(
    request: Request, container: ContainerDep, runner: RunnerDep, run_id: str
) -> StreamingResponse:
    # Assina antes de qualquer await: não há janela entre "está rodando?" e a inscrição.
    queue = runner.subscribe(run_id)
    if queue is None:
        run = await _get_run(container, run_id)
        final = JobEvent("done", RunOut.from_run(run).model_dump_json())

        async def single() -> AsyncIterator[str]:
            yield _sse(final)

        return _stream(single())

    async def live() -> AsyncIterator[str]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except TimeoutError:
                    if await request.is_disconnected():
                        break
                    yield ": ping\n\n"
                    continue
                yield _sse(event)
                if event.kind == "done":
                    break
        finally:
            runner.unsubscribe(run_id, queue)

    return _stream(live())


def _stream(body: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{run_id}/trace",
    summary="Spans OpenTelemetry da execução",
    responses={
        404: {"model": ErrorOut, "description": "Execução ou trace inexistente"},
        503: {"model": ErrorOut, "description": "Telemetria local não configurada"},
    },
)
async def get_run_trace(container: ContainerDep, run_id: str) -> TraceOut:
    run = await _get_run(container, run_id)
    if container.telemetry is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Telemetria não configurada")
    if not run.trace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Execução sem trace registrado")
    spans = await asyncio.to_thread(container.telemetry.get_trace, run.trace_id)
    if not spans:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Trace '{run.trace_id}' não encontrado")
    return TraceOut(trace_id=run.trace_id, spans=spans)
