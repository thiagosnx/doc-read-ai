import datetime
import json
import keyword
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from doc_read_ai.application.schema_builder import build_extraction_model
from doc_read_ai.domain.analysis import DocumentAnalysis, FieldType, ScalarType
from doc_read_ai.domain.errors import SchemaBuildError
from tests.conftest import analysis, field, sub


def _types(prop: dict[str, Any]) -> set[str]:
    """Tipos JSON aceitos por uma propriedade (desembrulha anyOf de campos anuláveis)."""
    options = prop.get("anyOf", [prop])
    return {o.get("type", o.get("$ref", "?")) for o in options}


def _resolve(schema: dict[str, Any], prop: dict[str, Any]) -> dict[str, Any]:
    for option in prop.get("anyOf", [prop]):
        if "$ref" in option:
            return schema["$defs"][option["$ref"].split("/")[-1]]
        if option.get("type") == "array":
            return option
    raise AssertionError(f"sem $ref/array em {prop}")


def test_scalar_type_mapping() -> None:
    model = build_extraction_model(
        analysis(
            "teste",
            [
                field("texto", FieldType.STRING),
                field("inteiro", FieldType.INTEGER),
                field("numero", FieldType.NUMBER),
                field("flag", FieldType.BOOLEAN),
                field("quando", FieldType.DATE),
            ],
        )
    )
    props = model.model_json_schema()["properties"]
    assert _types(props["texto"]) == {"string", "null"}
    assert _types(props["inteiro"]) == {"integer", "null"}
    assert _types(props["numero"]) == {"number", "null"}
    assert _types(props["flag"]) == {"boolean", "null"}
    assert {"format": "date", "type": "string"} in props["quando"]["anyOf"]

    instance = model.model_validate(
        {"texto": "a", "inteiro": 1, "numero": 1.5, "flag": True, "quando": "2026-01-15"}
    )
    assert instance.quando == datetime.date(2026, 1, 15)


def test_required_vs_optional_and_nullable() -> None:
    model = build_extraction_model(
        analysis(
            "teste",
            [field("obrigatorio", FieldType.STRING), field("opcional", FieldType.STRING, False)],
        )
    )
    schema = model.model_json_schema()
    assert "obrigatorio" in schema["required"]
    assert "opcional" not in schema.get("required", [])

    # obrigatório = a chave precisa vir, mas pode ser null (dado ilegível)
    assert model.model_validate({"obrigatorio": None}).opcional is None
    with pytest.raises(ValidationError):
        model.model_validate({"opcional": "x"})


def test_nested_object(nfe_analysis: DocumentAnalysis) -> None:
    model = build_extraction_model(nfe_analysis)
    schema = model.model_json_schema()
    emitente = _resolve(schema, schema["properties"]["emitente"])
    assert set(emitente["properties"]) == {"cnpj", "razao_social"}
    assert emitente["required"] == ["cnpj"]


def test_list_of_objects(nfe_analysis: DocumentAnalysis) -> None:
    from tests.conftest import NFE_PAYLOAD

    model = build_extraction_model(nfe_analysis)
    schema = model.model_json_schema()
    itens = _resolve(schema, schema["properties"]["itens"])
    item = schema["$defs"][itens["items"]["$ref"].split("/")[-1]]
    assert set(item["properties"]) == {"descricao", "quantidade", "valor_unitario"}

    data = model.model_validate(NFE_PAYLOAD).model_dump(mode="json")
    assert data["itens"][1] == {"descricao": "Porca", "quantidade": 20, "valor_unitario": 5.0}
    assert data["extra_fields"] == [{"key": "chave_acesso", "value": "3526 0112 3456"}]


def test_list_of_scalars_and_object_without_subfields() -> None:
    model = build_extraction_model(
        analysis(
            "teste",
            [
                field("habilidades", FieldType.LIST, item_type=ScalarType.STRING),
                field("notas", FieldType.LIST, item_type=ScalarType.NUMBER),
                field("tags", FieldType.LIST),  # sem item_type -> string
                field("endereco", FieldType.OBJECT),  # sem subfields -> string
            ],
        )
    )
    props = model.model_json_schema()["properties"]
    assert _resolve({}, props["habilidades"])["items"] == {"type": "string"}
    assert _resolve({}, props["notas"])["items"] == {"type": "number"}
    assert _resolve({}, props["tags"])["items"] == {"type": "string"}
    assert _types(props["endereco"]) == {"string", "null"}


def test_extra_fields_always_present() -> None:
    model = build_extraction_model(analysis("teste", [field("a", FieldType.STRING)]))
    assert model.model_validate({"a": "x"}).extra_fields == []
    assert "extra_fields" in model.model_json_schema()["properties"]


def test_name_sanitization() -> None:
    raw_names = [
        "Número da Nota",
        "valor (R$)",
        "class",
        "model_config",
        "schema",
        "1a_parcela",
        "total",
        "Total",
        "extra_fields",
        "!!!",
    ]
    model = build_extraction_model(
        analysis("teste", [field(n, FieldType.STRING, False) for n in raw_names])
    )
    names = [n for n in model.model_fields if n != "extra_fields"]

    assert len(names) == len(raw_names) == len(set(names))
    assert names[:2] == ["numero_da_nota", "valor_r"]
    for name in names:
        assert name.isidentifier() and name.isascii()
        assert not keyword.iskeyword(name)
        assert not name.startswith("model_")
        assert not hasattr(BaseModel, name)
        assert not name[0].isdigit()
    assert "total" in names and "total_2" in names
    assert "extra_fields" in model.model_fields  # o campo reservado não foi sobrescrito


def test_no_recursion_nor_free_form_dict(nfe_analysis: DocumentAnalysis) -> None:
    schema = build_extraction_model(nfe_analysis).model_json_schema()

    for name, definition in schema.get("$defs", {}).items():
        assert f"#/$defs/{name}" not in json.dumps(definition), f"{name} é recursivo"

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert node.get("additionalProperties") not in (True, {}), node
            if node.get("type") == "object":
                assert "properties" in node, f"dict livre: {node}"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)


def test_empty_fields_raises() -> None:
    with pytest.raises(SchemaBuildError):
        build_extraction_model(analysis("vazio", []))


def test_nested_subfield_names_sanitized() -> None:
    model = build_extraction_model(
        analysis(
            "teste",
            [field("itens", FieldType.LIST, subfields=[sub("Preço Unitário", ScalarType.NUMBER)])],
        )
    )
    schema = model.model_json_schema()
    item = schema["$defs"][_resolve(schema, schema["properties"]["itens"])["items"]["$ref"][8:]]
    assert list(item["properties"]) == ["preco_unitario"]
