import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from doc_read_ai.bootstrap import build_use_case
from doc_read_ai.domain.analysis import FieldType, RegisteredSchema
from doc_read_ai.infrastructure.loaders import LocalFileLoader
from doc_read_ai.infrastructure.registry import JsonFileSchemaRegistry
from doc_read_ai.interfaces import cli
from tests.conftest import field
from tests.fakes import FakeLLMFactory, pdf_bytes, png_bytes, write_file

runner = CliRunner()


def test_agents_command(config_path: Path) -> None:
    result = runner.invoke(cli.app, ["agents", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    for name in ("orchestrator", "classifier", "analyst", "extractor"):
        assert name in result.output


def test_missing_api_key_is_friendly(
    config_path: Path, png_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    result = runner.invoke(cli.app, ["process", str(png_file), "--config", str(config_path)])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output and ".env" in result.output


def test_process_directory_writes_json(
    config_path: Path,
    tmp_path: Path,
    nfe_factory: FakeLLMFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    write_file(docs, "nota.png", png_bytes())
    write_file(docs, "outra.pdf", pdf_bytes())
    write_file(docs, "ignorado.txt", b"texto")
    out = tmp_path / "out"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
    monkeypatch.setattr(
        cli,
        "build_use_case",
        lambda config_path=None: build_use_case(config_path, nfe_factory, LocalFileLoader()),
    )

    args = ["process", str(docs), "--out", str(out), "--config", str(config_path), "-n", "2"]
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == ["nota.json", "outra.json"]
    saved = json.loads((out / "nota.json").read_text(encoding="utf-8"))
    assert saved["status"] == "extracted"
    assert saved["data"]["itens"][0]["descricao"] == "Parafuso"
    assert "Parafuso" in result.output  # árvore de dados renderizada


def test_schemas_commands(config_path: Path) -> None:
    """list/show/rm sobre o registro em disco (cwd do teste é uma pasta temporária)."""
    args = ["--config", str(config_path)]
    empty = runner.invoke(cli.app, ["schemas", "list", *args])
    assert empty.exit_code == 0 and "Nenhum schema" in empty.output

    JsonFileSchemaRegistry(Path("schemas")).save(
        RegisteredSchema(
            document_type="boleto_bancario",
            document_type_label="Boleto Bancário",
            category="financeiro",
            fields=[field("linha_digitavel", FieldType.STRING)],
            created_at=datetime.now(UTC),
        )
    )

    listed = runner.invoke(cli.app, ["schemas", "list", *args])
    assert listed.exit_code == 0 and "boleto_bancario" in listed.output
    shown = runner.invoke(cli.app, ["schemas", "show", "boleto_bancario", *args])
    assert shown.exit_code == 0 and "linha_digitavel" in shown.output
    removed = runner.invoke(cli.app, ["schemas", "rm", "boleto_bancario", *args])
    assert removed.exit_code == 0
    missing = runner.invoke(cli.app, ["schemas", "show", "boleto_bancario", *args])
    assert missing.exit_code == 2
