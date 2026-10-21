"""Carrega e valida config/agents.yml como AppConfig."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from doc_read_ai.domain.config import AppConfig
from doc_read_ai.domain.errors import DocReadError

REQUIRED_AGENTS = ("orchestrator", "analyst", "extractor")


class ConfigError(DocReadError):
    """YAML de configuração ausente, malformado ou inválido."""


def load_app_config(path: Path) -> AppConfig:
    """Lê o YAML, injeta o `name` de cada agente a partir da chave e valida com Pydantic."""
    if not path.is_file():
        raise ConfigError(f"Arquivo de configuração não encontrado: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML inválido em {path}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("agents"), dict):
        raise ConfigError(f"{path}: esperado um mapa com a chave 'agents'")

    raw["agents"] = {name: _with_name(name, spec) for name, spec in raw["agents"].items()}
    raw["providers"] = raw.get("providers") or {}

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Configuração inválida em {path}: {problems}") from exc

    missing = [name for name in REQUIRED_AGENTS if name not in config.agents]
    if missing:
        raise ConfigError(f"{path}: agentes obrigatórios ausentes: {', '.join(missing)}")
    return config


def _with_name(name: str, spec: Any) -> Any:
    if not isinstance(spec, dict):
        return spec  # deixa o Pydantic reportar o erro de tipo
    return {**spec, "name": name}
