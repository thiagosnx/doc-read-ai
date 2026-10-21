"""Liveness/readiness (sempre abertos, mesmo com DOC_READ_API_KEY)."""

import asyncio

from fastapi import APIRouter, Response, status

from doc_read_ai.interfaces.api.deps import ContainerDep
from doc_read_ai.interfaces.api.schemas import CheckOut, HealthOut, ReadyOut

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness")
async def get_health() -> HealthOut:
    return HealthOut()


@router.get(
    "/ready",
    summary="Readiness: config carregada, banco acessível, pipeline montado",
    responses={503: {"model": ReadyOut, "description": "Alguma verificação falhou"}},
)
async def get_readiness(container: ContainerDep, response: Response) -> ReadyOut:
    checks = {"config": CheckOut(ok=True, detail=f"{len(container.app_config.agents)} agentes")}
    try:
        await asyncio.to_thread(container.runs.list, limit=1)
        checks["database"] = CheckOut(ok=True)
    except Exception as exc:
        checks["database"] = CheckOut(ok=False, detail=f"{type(exc).__name__}: {exc}")
    checks["pipeline"] = CheckOut(ok=container.use_case is not None, detail=container.startup_error)
    checks["telemetry"] = CheckOut(
        ok=True, detail="local" if container.telemetry is not None else "desligada"
    )
    is_ready = all(c.ok for c in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyOut(status="ready" if is_ready else "not_ready", checks=checks)
