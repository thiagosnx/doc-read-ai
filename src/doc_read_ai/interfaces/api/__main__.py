"""`python -m doc_read_ai.interfaces.api` ou `doc-read-api`: sobe a API com uvicorn."""

import os

import uvicorn

from doc_read_ai.bootstrap import configure_logging
from doc_read_ai.infrastructure.config import Settings


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "doc_read_ai.interfaces.api.app:create_app",
        factory=True,
        host=os.environ.get("DOC_READ_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("DOC_READ_API_PORT", "8000")),
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
