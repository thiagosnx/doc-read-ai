"""Dependências da API. Testes injetam um ApiContainer pronto (LLM falso, repositório em
memória); em produção `build_api_container` monta tudo a partir do YAML/.env."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from doc_read_ai.application.use_cases import ProcessDocumentUseCase
from doc_read_ai.bootstrap import build_schema_registry, build_use_case, load_config
from doc_read_ai.domain.config import AppConfig
from doc_read_ai.domain.errors import DocReadError
from doc_read_ai.domain.ports import RunRepositoryPort, SchemaRegistryPort, TelemetryStorePort
from doc_read_ai.infrastructure.persistence import SqliteRunRepository
from doc_read_ai.interfaces.api.settings import ApiSettings

logger = logging.getLogger(__name__)

SERVICE_NAME = "doc-read-ai"


@dataclass
class ApiContainer:
    settings: ApiSettings
    app_config: AppConfig
    runs: RunRepositoryPort
    schema_registry: SchemaRegistryPort
    use_case: ProcessDocumentUseCase | None
    telemetry: TelemetryStorePort | None = None
    # Ex.: chave de API ausente. A API sobe (consultas funcionam), mas não processa documentos.
    startup_error: str | None = None
    shutdown_hooks: list[Callable[[], None]] = field(default_factory=list)

    def shutdown(self) -> None:
        for hook in reversed(self.shutdown_hooks):
            try:
                hook()
            except Exception:  # desligamento não pode travar no meio
                logger.exception("Falha ao finalizar recurso da API")


def build_api_container(settings: ApiSettings) -> ApiContainer:
    """Composition root da API. A observabilidade é configurada ANTES do caso de uso: o SDK da
    Anthropic usa o tracer provider global no momento em que os clientes são criados."""
    app_config = load_config()  # também carrega o .env em os.environ (chaves, OTEL_*)
    hooks: list[Callable[[], None]] = []

    telemetry: TelemetryStorePort | None = None
    try:
        from doc_read_ai.infrastructure.observability import configure_observability
    except ImportError:
        logger.warning("Pacote de observabilidade indisponível; seguindo sem telemetria local")
    else:
        handles = configure_observability(
            service_name=SERVICE_NAME, pricing=getattr(app_config, "pricing", None) or {}
        )
        telemetry = handles.telemetry_store
        hooks.append(handles.shutdown)

    runs = SqliteRunRepository(settings.doc_read_runs_db)
    hooks.append(runs.close)
    registry = build_schema_registry(app_config)

    use_case: ProcessDocumentUseCase | None = None
    startup_error: str | None = None
    try:
        use_case = build_use_case(schema_registry=registry)
    except DocReadError as exc:
        startup_error = str(exc)
        logger.error("A API vai subir sem processar documentos: %s", exc)

    return ApiContainer(
        settings=settings,
        app_config=app_config,
        runs=runs,
        schema_registry=registry,
        use_case=use_case,
        telemetry=telemetry,
        startup_error=startup_error,
        shutdown_hooks=hooks,
    )
