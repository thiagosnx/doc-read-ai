"""Dependências FastAPI compartilhadas pelas rotas."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from doc_read_ai.interfaces.api.container import ApiContainer
from doc_read_ai.interfaces.api.jobs import JobRunner

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, scheme_name="ApiKey")


def get_container(request: Request) -> ApiContainer:
    return request.app.state.container


def get_runner(request: Request) -> JobRunner:
    return request.app.state.runner


def require_api_key(
    request: Request, api_key: Annotated[str | None, Security(api_key_header)] = None
) -> None:
    """Só exige a chave quando DOC_READ_API_KEY está definida."""
    expected = request.app.state.settings.doc_read_api_key
    if expected is None:
        return
    if not api_key or not secrets.compare_digest(
        api_key.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key ausente ou inválida",
            headers={"WWW-Authenticate": "ApiKey"},
        )


ContainerDep = Annotated[ApiContainer, Depends(get_container)]
RunnerDep = Annotated[JobRunner, Depends(get_runner)]
