from doc_read_ai.interfaces.api.routes import (
    agents,
    documents,
    health,
    observability,
    runs,
    schemas,
)

# Rotas de negócio (prefixo /v1, protegidas por X-API-Key quando configurada).
V1_ROUTERS = [
    documents.router,
    runs.router,
    schemas.router,
    agents.router,
    observability.router,
]

__all__ = ["V1_ROUTERS", "health"]
