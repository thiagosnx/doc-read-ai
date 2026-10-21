"""App FastAPI (adapter de entrada HTTP). `create_app()` monta tudo; testes injetam o container."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.formparsers import MultiPartParser

from doc_read_ai.domain.errors import (
    DocReadError,
    LLMError,
    SchemaBuildError,
    UnsupportedDocumentError,
)
from doc_read_ai.interfaces.api.container import ApiContainer, build_api_container
from doc_read_ai.interfaces.api.deps import require_api_key
from doc_read_ai.interfaces.api.jobs import JobRunner
from doc_read_ai.interfaces.api.routes import V1_ROUTERS, health
from doc_read_ai.interfaces.api.schemas import ErrorOut, StepEventOut
from doc_read_ai.interfaces.api.settings import ApiSettings
from doc_read_ai.interfaces.api.uploads import EmptyUploadError, UploadTooLargeError

logger = logging.getLogger(__name__)

DESCRIPTION = """Leitura autônoma de documentos (PDF/imagem) com agentes de IA.

Fluxo: `POST /v1/documents` cria uma execução por arquivo e responde 202; acompanhe por
`GET /v1/runs/{id}` ou ao vivo pelo SSE `GET /v1/runs/{id}/events`.

O arquivo enviado nunca é gravado em disco (só o sha256). Erros têm sempre o corpo
`{"detail": "..."}`. Se `DOC_READ_API_KEY` estiver definida, as rotas `/v1` exigem o header
`X-API-Key`."""

# Mais específico primeiro: a primeira classe que casar define o status.
_ERROR_STATUS: list[tuple[type[DocReadError], int]] = [
    (UploadTooLargeError, status.HTTP_413_CONTENT_TOO_LARGE),
    (EmptyUploadError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (UnsupportedDocumentError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (SchemaBuildError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (LLMError, status.HTTP_502_BAD_GATEWAY),
    (DocReadError, status.HTTP_400_BAD_REQUEST),
]

ERROR_RESPONSES = {
    401: {"model": ErrorOut, "description": "X-API-Key ausente ou inválida"},
    422: {"model": ErrorOut, "description": "Requisição inválida"},
}


def _app_version() -> str:
    try:
        return version("doc-read-ai")
    except PackageNotFoundError:
        return "0.0.0"


def _operation_id(route: APIRoute) -> str:
    """operationId = nome da função (únicos): clientes TS saem com getRun, listRuns..."""
    return route.name


async def _doc_read_error(_: Request, exc: Exception) -> JSONResponse:
    code = next(c for cls, c in _ERROR_STATUS if isinstance(exc, cls))
    return JSONResponse({"detail": str(exc)}, status_code=code)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return JSONResponse(
        {"detail": "; ".join(parts) or "Requisição inválida"},
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


def _with_sse_models(app: FastAPI) -> None:
    """Os eventos SSE não são resposta JSON de nenhuma rota; publica os modelos no OpenAPI
    mesmo assim para o cliente TS ter os tipos (StepEventOut; o `done` usa RunOut)."""
    original = app.openapi

    def openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        spec = original()
        schema = StepEventOut.model_json_schema(ref_template="#/components/schemas/{model}")
        components = spec.setdefault("components", {}).setdefault("schemas", {})
        components.update(schema.pop("$defs", {}))
        components.setdefault("StepEventOut", schema)
        return spec

    app.openapi = openapi  # type: ignore[method-assign]


def create_app(container: ApiContainer | None = None) -> FastAPI:
    """Sem container: lê ApiSettings, configura observabilidade e monta o pipeline no startup."""
    settings = container.settings if container else ApiSettings()

    # O Starlette despeja uploads > 1 MB num arquivo temporário. Com o limite acima do tamanho
    # máximo aceito, o arquivo fica só em memória (nunca toca o disco).
    MultiPartParser.spool_max_size = max(
        MultiPartParser.spool_max_size, settings.max_upload_bytes + 1024 * 1024
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = container or build_api_container(settings)
        runner = JobRunner(
            lambda: active.use_case,
            active.runs,
            concurrency=settings.doc_read_concurrency,
            store_results=settings.doc_read_store_results,
        )
        app.state.container = active
        app.state.runner = runner
        await runner.recover()
        try:
            yield
        finally:
            await runner.shutdown()
            active.shutdown()

    app = FastAPI(
        title="doc-read-ai",
        version=_app_version(),
        description=DESCRIPTION,
        lifespan=lifespan,
        generate_unique_id_function=_operation_id,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.doc_read_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Last-Event-ID"],
    )
    app.add_exception_handler(DocReadError, _doc_read_error)
    app.add_exception_handler(RequestValidationError, _validation_error)

    app.include_router(health.router)
    for router in V1_ROUTERS:
        app.include_router(
            router,
            prefix="/v1",
            dependencies=[Depends(require_api_key)],
            responses=ERROR_RESPONSES,
        )

    _with_sse_models(app)

    if container is None:
        try:
            from doc_read_ai.infrastructure.observability import instrument_fastapi
        except ImportError:
            logger.warning("Pacote de observabilidade indisponível; API sem instrumentação OTel")
        else:
            instrument_fastapi(app)
    return app
