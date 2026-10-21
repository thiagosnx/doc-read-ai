"""Configuração da API HTTP (variáveis de ambiente ou .env)."""

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ApiSettings(BaseSettings):
    """DOC_READ_* da API. Valores do ambiente sobrescrevem o .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    doc_read_api_key: SecretStr | None = Field(
        default=None, description="Se definida, toda rota /v1 exige o header X-API-Key"
    )
    doc_read_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"],
        description="Origens liberadas no CORS (JSON ou separadas por vírgula)",
    )
    doc_read_runs_db: Path = Path("data/runs.db")
    doc_read_store_results: bool = Field(
        default=True,
        description="False = guarda só metadados da execução, nunca os dados extraídos",
    )
    doc_read_max_upload_mb: int = Field(default=32, ge=1)
    doc_read_concurrency: int = Field(default=3, ge=1)

    @field_validator("doc_read_cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                return json.loads(text)
            return [o.strip() for o in text.split(",") if o.strip()]
        return value

    @field_validator("doc_read_api_key", mode="after")
    @classmethod
    def _blank_key_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        return value if value and value.get_secret_value().strip() else None

    @property
    def max_upload_bytes(self) -> int:
        return self.doc_read_max_upload_mb * 1024 * 1024

    @property
    def api_auth_enabled(self) -> bool:
        return self.doc_read_api_key is not None
