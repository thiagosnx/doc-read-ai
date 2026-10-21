"""API HTTP (FastAPI): adapter de entrada para o frontend."""

from doc_read_ai.interfaces.api.app import create_app
from doc_read_ai.interfaces.api.container import ApiContainer, build_api_container
from doc_read_ai.interfaces.api.settings import ApiSettings

__all__ = ["ApiContainer", "ApiSettings", "build_api_container", "create_app"]
