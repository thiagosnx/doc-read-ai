"""Caso de uso principal (porta de entrada): processar um documento de ponta a ponta."""

import mimetypes
from pathlib import Path
from typing import Any

from doc_read_ai.domain.analysis import ExtractionResult, PipelineStatus
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.errors import UnsupportedDocumentError
from doc_read_ai.domain.ports import DocumentLoaderPort


class ProcessDocumentUseCase:
    """Carrega o arquivo e executa o grafo de agentes, devolvendo sempre um ExtractionResult."""

    def __init__(self, loader: DocumentLoaderPort, graph: Any) -> None:
        self._loader = loader
        self._graph = graph  # grafo LangGraph compilado (build_graph)

    async def execute(self, source: Path) -> ExtractionResult:
        try:
            document = self._loader.load(source)
        except UnsupportedDocumentError as exc:
            return ExtractionResult(
                filename=source.name,
                media_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
                status=PipelineStatus.FAILED,
                errors=[f"[load] {type(exc).__name__}: {exc}"],
            )
        return await self.execute_document(document)

    async def execute_document(self, document: Document) -> ExtractionResult:
        state = await self._graph.ainvoke({"document": document, "errors": []})
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
        )
