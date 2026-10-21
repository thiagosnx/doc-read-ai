"""Governança: agregados das execuções + telemetria OTel + como os dados são tratados."""

import asyncio
import os
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, status

from doc_read_ai.infrastructure.observability import otlp_enabled
from doc_read_ai.interfaces.api.container import ApiContainer
from doc_read_ai.interfaces.api.deps import ContainerDep
from doc_read_ai.interfaces.api.routes.agents import agent_list, schema_registry_info
from doc_read_ai.interfaces.api.schemas import (
    ErrorOut,
    GovernanceOut,
    ObservabilitySummaryOut,
    OtlpExportOut,
    PeriodOut,
    TraceOut,
)

router = APIRouter(prefix="/observability", tags=["observability"])

CONTENT_MODE_ENV = "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE"
_METADATA_MODES = {"", "none", "off", "false", "0", "metadata", "metadata_only", "no_content"}


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def governance(container: ApiContainer) -> GovernanceOut:
    content_mode = _env(CONTENT_MODE_ENV).lower()
    return GovernanceOut(
        results_stored=container.settings.doc_read_store_results,
        llm_content_capture="metadata_only" if content_mode in _METADATA_MODES else "content",
        otlp_export=OtlpExportOut(
            traces=otlp_enabled("traces"),
            metrics=otlp_enabled("metrics"),
            logs=otlp_enabled("logs"),
        ),
        api_auth_enabled=container.settings.api_auth_enabled,
    )


@router.get("/summary", summary="Resumo de governança: execuções, custo/tokens, latência")
async def get_observability_summary(
    container: ContainerDep,
    since: Annotated[datetime | None, Query(description="Início (ISO 8601), inclusivo")] = None,
    until: Annotated[datetime | None, Query(description="Fim (ISO 8601), exclusivo")] = None,
) -> ObservabilitySummaryOut:
    runs = await asyncio.to_thread(container.runs.stats, since, until)
    telemetry = None
    if container.telemetry is not None:
        telemetry = await asyncio.to_thread(container.telemetry.summary, since, until)
    return ObservabilitySummaryOut(
        period=PeriodOut(since=since, until=until),
        runs=runs,
        telemetry=telemetry,
        agents=agent_list(container),
        schema_registry=await schema_registry_info(container),
        governance=governance(container),
    )


@router.get(
    "/traces/{trace_id}",
    summary="Spans de um trace OpenTelemetry",
    responses={404: {"model": ErrorOut}, 503: {"model": ErrorOut}},
)
async def get_trace(
    container: ContainerDep,
    trace_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$", description="32 hex")],
) -> TraceOut:
    if container.telemetry is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Telemetria não configurada")
    spans = await asyncio.to_thread(container.telemetry.get_trace, trace_id)
    if not spans:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Trace '{trace_id}' não encontrado")
    return TraceOut(trace_id=trace_id, spans=spans)
