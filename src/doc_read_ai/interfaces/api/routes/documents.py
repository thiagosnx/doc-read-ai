"""Upload de documentos: valida em memória, cria as execuções e processa em background."""

import asyncio
import hashlib
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status

from doc_read_ai.domain.document import Document
from doc_read_ai.domain.runs import Run
from doc_read_ai.interfaces.api.deps import ContainerDep, RunnerDep
from doc_read_ai.interfaces.api.schemas import ErrorOut, RunOut, SubmissionOut
from doc_read_ai.interfaces.api.uploads import document_from_bytes

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILES_PER_REQUEST = 20


def _safe_filename(raw: str | None) -> str:
    """Só o nome (sem diretórios vindos do cliente), com tamanho limitado."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name[:255] or "documento"


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Envia um ou mais documentos para processamento",
    responses={
        200: {"model": SubmissionOut, "description": "wait=true e todas as execuções terminaram"},
        413: {"model": ErrorOut, "description": "Arquivo acima do limite"},
        415: {"model": ErrorOut, "description": "Formato não suportado (checado pelo conteúdo)"},
        503: {"model": ErrorOut, "description": "Pipeline indisponível (ex.: chave ausente)"},
    },
)
async def submit_documents(
    container: ContainerDep,
    runner: RunnerDep,
    response: Response,
    files: Annotated[list[UploadFile], File(description="PDF, PNG, JPEG, WEBP ou GIF")],
    wait: Annotated[
        bool, Query(description="Espera terminar (útil em scripts); senão responde 202 já")
    ] = False,
    timeout: Annotated[float, Query(ge=1, le=600, description="Limite do wait, em s")] = 120,
) -> SubmissionOut:
    """Cada arquivo vira uma execução (`queued`). Acompanhe por `GET /v1/runs/{id}` ou pelo
    stream SSE `GET /v1/runs/{id}/events`. Se algum arquivo for inválido, nada é processado.
    O arquivo nunca é gravado em disco: só o sha256 fica no histórico."""
    if container.use_case is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, container.startup_error)
    if not files:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Envie ao menos um arquivo")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"No máximo {MAX_FILES_PER_REQUEST} arquivos por requisição",
        )

    max_bytes = container.settings.max_upload_bytes
    documents: list[Document] = []
    for upload in files:
        content = await upload.read(max_bytes + 1)
        await upload.close()
        # Erros de validação viram 413/415/422 pelo handler de UnsupportedDocumentError.
        documents.append(document_from_bytes(_safe_filename(upload.filename), content, max_bytes))

    runs = [
        Run(
            id=str(uuid.uuid4()),
            filename=doc.filename,
            media_type=doc.media_type,
            size_bytes=doc.size_bytes,
            sha256=hashlib.sha256(doc.content).hexdigest(),
            created_at=datetime.now(UTC),
        )
        for doc in documents
    ]
    for run in runs:
        await asyncio.to_thread(container.runs.save, run)
    ids = [runner.submit(doc, run) for doc, run in zip(documents, runs, strict=True)]

    if not wait:
        return SubmissionOut(runs=[RunOut.from_run(r) for r in runs])

    finished = await runner.wait(ids, timeout)
    final: list[Run] = []
    for run in runs:
        current = finished.get(run.id) or await asyncio.to_thread(container.runs.get, run.id)
        final.append(current or run)
    if all(r.status.is_final for r in final):
        response.status_code = status.HTTP_200_OK
    return SubmissionOut(runs=[RunOut.from_run(r) for r in final])
