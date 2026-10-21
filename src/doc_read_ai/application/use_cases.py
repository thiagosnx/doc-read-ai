"""Caso de uso principal (porta de entrada): processar um documento de ponta a ponta.

Usa só a API do OpenTelemetry (no-op sem SDK configurado): a aplicação abre o span raiz de
cada documento e quem configura exportadores é a infraestrutura.
"""

import hashlib
import mimetypes
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from doc_read_ai.domain.analysis import ExtractionResult, PipelineStatus
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.errors import UnsupportedDocumentError
from doc_read_ai.domain.ports import DocumentLoaderPort
from doc_read_ai.domain.runs import PipelineEvent

tracer = trace.get_tracer("doc_read_ai")

ROOT_SPAN = "doc_read.process_document"

_meter = metrics.get_meter("doc_read_ai")
_documents = _meter.create_counter(
    "doc_read.documents", unit="{document}", description="Documentos processados"
)
_document_duration = _meter.create_histogram(
    "doc_read.document.duration", unit="ms", description="Duração do processamento do documento"
)


class ProcessDocumentUseCase:
    """Carrega o arquivo e executa o grafo de agentes, devolvendo sempre um ExtractionResult."""

    def __init__(self, loader: DocumentLoaderPort, graph: Any) -> None:
        self._loader = loader
        self._graph = graph  # grafo LangGraph compilado (build_graph)

    async def execute(self, source: Path, run_id: str | None = None) -> ExtractionResult:
        started = time.perf_counter()
        try:
            document = self._loader.load(source)
        except UnsupportedDocumentError as exc:
            result = ExtractionResult(
                filename=source.name,
                media_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
                status=PipelineStatus.FAILED,
                errors=[f"[load] {type(exc).__name__}: {exc}"],
            )
            _record_metrics(result, started)
            return result
        return await self.execute_document(document, run_id)

    async def execute_document(
        self, document: Document, run_id: str | None = None
    ) -> ExtractionResult:
        result: ExtractionResult | None = None
        async for event in self.stream_document(document, run_id):
            result = event.result or result
        assert result is not None  # stream_document sempre termina com o evento 'done'
        return result

    async def stream_document(
        self, document: Document, run_id: str | None = None
    ) -> AsyncIterator[PipelineEvent]:
        """Executa o grafo emitindo um evento por nó concluído e um 'done' com o resultado.

        O nome do arquivo não vai para o span (pode conter dados pessoais); o hash sim.
        """
        started = time.perf_counter()
        attributes: dict[str, Any] = {
            "doc_read.document.media_type": document.media_type,
            "doc_read.document.size_bytes": document.size_bytes,
            "doc_read.document.sha256": hashlib.sha256(document.content).hexdigest(),
        }
        if run_id:
            attributes["doc_read.run_id"] = run_id

        with tracer.start_as_current_span(ROOT_SPAN, attributes=attributes) as span:
            state: dict[str, Any] = {}
            async for mode, chunk in self._graph.astream(
                {"document": document, "errors": []}, stream_mode=["updates", "values"]
            ):
                if mode == "values":
                    state = chunk
                    continue
                for step, update in chunk.items():
                    failed = (update or {}).get("status") == PipelineStatus.FAILED
                    yield PipelineEvent(
                        step=step,
                        status="failed" if failed else "completed",
                        detail=_event_detail(update or {}),
                    )

            result = self._to_result(document, state, span)
            span.set_attribute("doc_read.status", result.status.value)
            if result.analysis:
                span.set_attribute("doc_read.document_type", result.analysis.document_type)
            if result.schema_source:
                span.set_attribute("doc_read.schema_source", result.schema_source.value)
            if result.status == PipelineStatus.FAILED:
                span.set_status(Status(StatusCode.ERROR, "; ".join(result.errors)[:500]))
            _record_metrics(result, started)
            yield PipelineEvent(step="done", status="done", result=result)

    @staticmethod
    def _to_result(document: Document, state: dict[str, Any], span: Any) -> ExtractionResult:
        ctx = span.get_span_context()
        return ExtractionResult(
            filename=document.filename,
            media_type=document.media_type,
            status=state.get("status", PipelineStatus.FAILED),
            verdict=state.get("verdict"),
            analysis=state.get("analysis"),
            schema_source=state.get("schema_source"),
            data=state.get("data"),
            json_schema=state.get("json_schema"),
            errors=state.get("errors", []),
            trace_id=format(ctx.trace_id, "032x") if ctx.is_valid else None,
        )


def _record_metrics(result: ExtractionResult, started: float) -> None:
    attributes = {
        "status": result.status.value,
        "document_type": result.analysis.document_type if result.analysis else "none",
        "schema_source": result.schema_source.value if result.schema_source else "none",
    }
    _documents.add(1, attributes)
    _document_duration.record((time.perf_counter() - started) * 1000, attributes)


def _event_detail(update: dict[str, Any]) -> dict[str, Any]:
    """Resumo leve e serializável do que o nó produziu (sem dados extraídos)."""
    detail: dict[str, Any] = {}
    if verdict := update.get("verdict"):
        detail["is_valid_document"] = verdict.is_valid_document
    if classification := update.get("classification"):
        detail["document_type"] = classification.document_type
        detail["document_type_label"] = classification.document_type_label
    if source := update.get("schema_source"):
        detail["schema_source"] = source.value
    if errors := update.get("errors"):
        detail["errors"] = errors
    return detail
