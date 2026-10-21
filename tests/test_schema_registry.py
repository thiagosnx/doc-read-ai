"""Adapters do registro de schemas: arquivo JSON e memória têm o mesmo comportamento."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from doc_read_ai.domain.analysis import FieldType, RegisteredSchema
from doc_read_ai.domain.ports import SchemaRegistryPort
from doc_read_ai.infrastructure.registry import InMemorySchemaRegistry, JsonFileSchemaRegistry
from tests.conftest import field


def registered(document_type: str = "boleto_bancario") -> RegisteredSchema:
    return RegisteredSchema(
        document_type=document_type,
        document_type_label="Boleto Bancário",
        category="financeiro",
        fields=[field("linha_digitavel", FieldType.STRING), field("valor", FieldType.NUMBER)],
        created_at=datetime(2026, 10, 21, 12, 0, tzinfo=UTC),
    )


@pytest.fixture(params=["json", "memory"])
def registry(request: pytest.FixtureRequest, tmp_path: Path) -> SchemaRegistryPort:
    if request.param == "json":
        return JsonFileSchemaRegistry(tmp_path / "schemas")
    return InMemorySchemaRegistry()


def test_empty_registry(registry: SchemaRegistryPort) -> None:
    assert registry.list_all() == []
    assert registry.get("boleto_bancario") is None
    assert registry.delete("boleto_bancario") is False


def test_save_get_list_delete(registry: SchemaRegistryPort) -> None:
    registry.save(registered("boleto_bancario"))
    registry.save(registered("extrato_bancario"))

    assert registry.get("boleto_bancario") == registered("boleto_bancario")
    assert sorted(s.document_type for s in registry.list_all()) == [
        "boleto_bancario",
        "extrato_bancario",
    ]
    assert registry.delete("boleto_bancario") is True
    assert registry.get("boleto_bancario") is None


def test_save_overwrites_same_type(registry: SchemaRegistryPort) -> None:
    registry.save(registered())
    updated = registered().model_copy(update={"document_type_label": "Boleto"})
    registry.save(updated)

    assert registry.get("boleto_bancario").document_type_label == "Boleto"
    assert len(registry.list_all()) == 1


def test_json_file_is_readable_and_has_no_leftovers(tmp_path: Path) -> None:
    directory = tmp_path / "schemas"
    JsonFileSchemaRegistry(directory).save(registered())

    assert [p.name for p in directory.iterdir()] == ["boleto_bancario.json"]
    assert '"linha_digitavel"' in (directory / "boleto_bancario.json").read_text(encoding="utf-8")


def test_corrupt_file_is_ignored(tmp_path: Path) -> None:
    directory = tmp_path / "schemas"
    registry = JsonFileSchemaRegistry(directory)
    registry.save(registered())
    (directory / "editado_errado.json").write_text("{ isso não é json", encoding="utf-8")

    assert [s.document_type for s in registry.list_all()] == ["boleto_bancario"]


@pytest.mark.parametrize("bad_id", ["../fora", "com espaço", "Maiusculo", ""])
def test_rejects_unsafe_type_ids(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(ValueError):
        JsonFileSchemaRegistry(tmp_path).get(bad_id)
