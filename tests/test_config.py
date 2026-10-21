import os
from pathlib import Path

import pytest

from doc_read_ai.infrastructure.config import (
    ConfigError,
    Settings,
    load_app_config,
    load_env_file,
)

VALID_YAML = """
agents:
  orchestrator: {provider: anthropic, model: m1}
  classifier: {provider: anthropic, model: m1}
  analyst: {provider: anthropic, model: m2, max_tokens: 1000, params: {effort: high}}
  extractor: {provider: openai, model: m3, system_prompt: "seja breve"}
providers:
  anthropic: {api_key_env: ANTHROPIC_API_KEY}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agents.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_project_config(config_path: Path) -> None:
    config = load_app_config(config_path)
    assert set(config.agents) == {"orchestrator", "classifier", "analyst", "extractor"}
    assert config.schema_registry.enabled and config.schema_registry.path == "schemas"
    assert all(name == agent.name for name, agent in config.agents.items())
    assert config.providers["anthropic"].api_key_env == "ANTHROPIC_API_KEY"


def test_injects_names_and_defaults(tmp_path: Path) -> None:
    config = load_app_config(_write(tmp_path, VALID_YAML))
    analyst = config.agent("analyst")
    assert analyst.name == "analyst"
    assert analyst.max_tokens == 1000
    assert analyst.params == {"effort": "high"}
    assert config.agent("orchestrator").max_tokens == 16000
    assert config.agent("extractor").system_prompt == "seja breve"
    assert config.schema_registry.enabled  # padrão quando o YAML não define


def test_missing_required_agent(tmp_path: Path) -> None:
    text = "agents:\n  orchestrator: {provider: a, model: m}\n  analyst: {provider: a, model: m}\n"
    with pytest.raises(ConfigError, match="extractor"):
        load_app_config(_write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="não encontrado"):
        load_app_config(tmp_path / "nao_existe.yml")


def test_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="YAML inválido"):
        load_app_config(_write(tmp_path, "agents: [sem fechar"))


def test_validation_error_points_to_field(tmp_path: Path) -> None:
    text = VALID_YAML.replace("model: m1", "max_tokens: dez")
    with pytest.raises(ConfigError, match=r"agents\.orchestrator\.(model|max_tokens)"):
        load_app_config(_write(tmp_path, text))


def test_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_READ_CONFIG", "outro/agents.yml")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)
    assert settings.doc_read_config == Path("outro/agents.yml")
    assert settings.log_level == "DEBUG"


def test_load_env_file_does_not_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("DOC_READ_TEST_A=do_arquivo\nDOC_READ_TEST_B=do_arquivo\n", encoding="utf-8")
    monkeypatch.setenv("DOC_READ_TEST_A", "do_ambiente")
    monkeypatch.setenv("DOC_READ_TEST_B", "")
    monkeypatch.delenv("DOC_READ_TEST_B")  # garante limpeza após o teste
    load_env_file(env)
    assert os.environ["DOC_READ_TEST_A"] == "do_ambiente"
    assert os.environ["DOC_READ_TEST_B"] == "do_arquivo"
