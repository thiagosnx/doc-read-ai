"""Configuração de ambiente (.env + variáveis de ambiente)."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Lê DOC_READ_CONFIG e LOG_LEVEL do ambiente ou do .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    doc_read_config: Path = Path("config/agents.yml")
    log_level: str = "INFO"


def load_env_file(path: Path = Path(".env")) -> None:
    """Carrega o .env em os.environ sem sobrescrever variáveis já definidas.

    Necessário porque os adapters leem a chave via os.environ[provider.api_key_env].
    """
    if path.is_file():
        load_dotenv(path, override=False, encoding="utf-8")
