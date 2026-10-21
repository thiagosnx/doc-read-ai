"""Execução em background dentro do processo (v1, sem fila externa).

Cada documento vira uma task asyncio limitada por semáforo. O progresso vai para o
repositório (estado durável) e para os assinantes SSE (fan-out por asyncio.Queue). Os bytes
enviados ficam só em memória durante a execução; nunca vão para o disco.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from doc_read_ai.application.use_cases import ProcessDocumentUseCase
from doc_read_ai.domain.analysis import ExtractionResult
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.ports import RunRepositoryPort
from doc_read_ai.domain.runs import PipelineEvent, Run, RunStatus
from doc_read_ai.interfaces.api.schemas import RunOut, StepEventOut

logger = logging.getLogger(__name__)

RESTART_ERROR = "interrompido: o servidor reiniciou"
SHUTDOWN_ERROR = "interrompido: o servidor foi desligado"


@dataclass(frozen=True)
class JobEvent:
    """Evento pronto para SSE: `data` já é JSON."""

    kind: Literal["step", "done"]
    data: str


def _now() -> datetime:
    return datetime.now(UTC)


def strip_for_storage(run: Run) -> Run:
    """Sem guarda de resultados: some `data` e o texto livre que o LLM escreveu sobre o
    conteúdo (resumo, notas, justificativa), que pode conter dados pessoais."""
    if run.result is None:
        return run
    result = run.result
    update: dict = {"data": None}
    if result.analysis:
        update["analysis"] = result.analysis.model_copy(
            update={"summary": "", "structure_notes": ""}
        )
    if result.verdict:
        update["verdict"] = result.verdict.model_copy(update={"reason": ""})
    return run.model_copy(update={"result": result.model_copy(update=update)})


class JobRunner:
    def __init__(
        self,
        use_case_provider: Callable[[], ProcessDocumentUseCase | None],
        runs: RunRepositoryPort,
        *,
        concurrency: int = 3,
        store_results: bool = True,
    ) -> None:
        self._use_case_provider = use_case_provider
        self._runs = runs
        self._semaphore = asyncio.Semaphore(concurrency)
        self._store_results = store_results
        self._tasks: dict[str, asyncio.Task[Run]] = {}
        self._history: dict[str, list[JobEvent]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[JobEvent]]] = {}

    # --- ciclo de vida ----------------------------------------------------------------------

    async def recover(self) -> int:
        """Execuções QUEUED/RUNNING de um processo anterior nunca vão terminar: viram FAILED."""
        recovered = 0
        for status in (RunStatus.QUEUED, RunStatus.RUNNING):
            while stale := await asyncio.to_thread(self._runs.list, limit=100, status=status):
                for run in stale:
                    failed = run.model_copy(
                        update={
                            "status": RunStatus.FAILED,
                            "finished_at": _now(),
                            "errors": [*run.errors, RESTART_ERROR],
                        }
                    )
                    await asyncio.to_thread(self._runs.save, failed)
                    recovered += 1
        if recovered:
            logger.warning("%d execução(ões) interrompida(s) marcada(s) como failed", recovered)
        return recovered

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # --- API --------------------------------------------------------------------------------

    def submit(self, document: Document, run: Run) -> str:
        """Agenda o processamento (o Run já deve estar salvo como QUEUED)."""
        self._history[run.id] = []
        self._subscribers[run.id] = set()
        self._tasks[run.id] = asyncio.create_task(self._execute(document, run), name=run.id)
        return run.id

    def is_active(self, run_id: str) -> bool:
        return run_id in self._tasks

    async def wait(self, run_ids: list[str], timeout: float) -> dict[str, Run]:
        """Espera as execuções terminarem (até `timeout`). Devolve as que terminaram, com o
        resultado completo mesmo quando ele não é guardado no repositório."""
        tasks = {rid: t for rid in run_ids if (t := self._tasks.get(rid))}
        if tasks:
            await asyncio.wait(tasks.values(), timeout=timeout)
        return {
            rid: task.result()
            for rid, task in tasks.items()
            if task.done() and not task.cancelled() and task.exception() is None
        }

    def subscribe(self, run_id: str) -> asyncio.Queue[JobEvent] | None:
        """Fila com os eventos já emitidos (replay) + os próximos. None se não está rodando."""
        if run_id not in self._subscribers:
            return None
        queue: asyncio.Queue[JobEvent] = asyncio.Queue()
        for event in self._history[run_id]:
            queue.put_nowait(event)
        self._subscribers[run_id].add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[JobEvent]) -> None:
        self._subscribers.get(run_id, set()).discard(queue)

    # --- execução ---------------------------------------------------------------------------

    def _publish(self, run_id: str, event: JobEvent) -> None:
        if run_id in self._history:
            self._history[run_id].append(event)
        for queue in self._subscribers.get(run_id, ()):
            queue.put_nowait(event)

    def _step_event(self, run_id: str, event: PipelineEvent) -> JobEvent:
        out = StepEventOut(
            run_id=run_id,
            step=event.step,
            status="failed" if event.status == "failed" else "completed",
            detail=event.detail,
            at=_now(),
        )
        return JobEvent("step", out.model_dump_json())

    async def _save(self, run: Run) -> None:
        stored = run if self._store_results else strip_for_storage(run)
        await asyncio.to_thread(self._runs.save, stored)

    async def _execute(self, document: Document, run: Run) -> Run:
        try:
            async with self._semaphore:
                run = run.model_copy(update={"status": RunStatus.RUNNING, "started_at": _now()})
                await self._save(run)
                run = await self._run_pipeline(document, run)
        except asyncio.CancelledError:
            run = self._failed(run, SHUTDOWN_ERROR)
            await self._finish(run)
            raise
        except Exception as exc:  # o caso de uso não deveria levantar; não deixa o run órfão
            logger.exception("Execução %s falhou", run.id)
            run = self._failed(run, f"{type(exc).__name__}: {exc}")
        await self._finish(run)
        return run

    async def _run_pipeline(self, document: Document, run: Run) -> Run:
        use_case = self._use_case_provider()
        if use_case is None:
            return self._failed(run, "pipeline indisponível: verifique a configuração da API")
        result: ExtractionResult | None = None
        async for event in use_case.stream_document(document, run.id):
            if event.step == "done":
                result = event.result
            else:
                self._publish(run.id, self._step_event(run.id, event))
        if result is None:
            return self._failed(run, "o pipeline terminou sem resultado")
        return run.model_copy(
            update={
                "status": RunStatus.from_pipeline(result.status),
                "finished_at": _now(),
                "trace_id": result.trace_id,
                "document_type": result.analysis.document_type if result.analysis else None,
                "schema_source": result.schema_source,
                "result": result,
                "errors": list(result.errors),
            }
        )

    @staticmethod
    def _failed(run: Run, error: str) -> Run:
        return run.model_copy(
            update={
                "status": RunStatus.FAILED,
                "started_at": run.started_at or _now(),
                "finished_at": _now(),
                "errors": [*run.errors, error],
            }
        )

    async def _finish(self, run: Run) -> None:
        """Grava o estado final ANTES de avisar e de sair da lista de ativos: quem assinar
        durante o save ainda recebe o `done`; quem chegar depois lê o repositório."""
        try:
            await self._save(run)
        except Exception:
            logger.exception("Não foi possível gravar o estado final da execução %s", run.id)
        self._publish(run.id, JobEvent("done", RunOut.from_run(run).model_dump_json()))
        self._history.pop(run.id, None)
        self._subscribers.pop(run.id, None)
        self._tasks.pop(run.id, None)
