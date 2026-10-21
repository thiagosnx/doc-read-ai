"""Converte o schema proposto pelo analista (list[FieldSpec]) em um modelo Pydantic em runtime.

O modelo gerado é usado como saída estruturada do extrator, então respeita as mesmas
restrições dos domain models: sem recursão, sem dict livre e sem constraints (ge, max_length...).
Todo campo é anulável: o extrator deve devolver null quando a informação não estiver legível.
"""

import datetime
import keyword
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field, create_model

from doc_read_ai.domain.analysis import DocumentAnalysis, FieldSpec, FieldType, SubFieldSpec
from doc_read_ai.domain.errors import SchemaBuildError

EXTRA_FIELDS_NAME = "extra_fields"

_SCALAR_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "date": datetime.date,
}


class ExtraField(BaseModel):
    """Informação relevante do documento que não coube no schema proposto."""

    key: str = Field(description="Nome do dado em snake_case. Ex: observacoes_pagamento")
    value: str = Field(description="Valor exatamente como aparece no documento")


def _ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _ascii(text).lower()).strip("_")


def _pascal(text: str) -> str:
    return "".join(part.capitalize() for part in _snake(text).split("_") if part)


def _shadows_basemodel(name: str) -> bool:
    return name.startswith("model_") or hasattr(BaseModel, name)


def _field_name(raw: str, position: int, used: set[str]) -> str:
    """snake_case ASCII válido como identificador, sem colidir com BaseModel nem duplicar."""
    base = _snake(raw) or f"campo_{position}"
    if base[0].isdigit() or _shadows_basemodel(base):
        base = f"campo_{base}"
    candidate, suffix = base, 2
    while (name := f"{candidate}_" if keyword.iskeyword(candidate) else candidate) in used:
        candidate, suffix = f"{base}_{suffix}", suffix + 1
    used.add(name)
    return name


class _ModelNames:
    """Garante nomes de modelo únicos (viram `$defs` no JSON Schema)."""

    def __init__(self) -> None:
        self._used: set[str] = {ExtraField.__name__}

    def make(self, base: str) -> str:
        name, suffix = base, 2
        while name in self._used:
            name, suffix = f"{base}{suffix}", suffix + 1
        self._used.add(name)
        return name


def _definition(annotation: Any, description: str, required: bool) -> tuple[Any, Any]:
    # required -> sem default: o modelo é obrigado a emitir a chave (possivelmente null).
    if required:
        return (annotation | None, Field(description=description))
    return (annotation | None, Field(default=None, description=description))


def _nested_model(name: str, description: str, subfields: list[SubFieldSpec]) -> type[BaseModel]:
    used: set[str] = set()
    definitions = {
        _field_name(sub.name, i, used): _definition(
            _SCALAR_TYPES[sub.type], sub.description, sub.required
        )
        for i, sub in enumerate(subfields, start=1)
    }
    return create_model(name, __doc__=description, **definitions)


def _annotation(spec: FieldSpec, parent: str, names: _ModelNames) -> Any:
    match spec.type:
        case FieldType.OBJECT:
            if not spec.subfields:
                return str
            model_name = names.make(f"{parent}{_pascal(spec.name) or 'Objeto'}")
            return _nested_model(model_name, spec.description, spec.subfields)
        case FieldType.LIST:
            if spec.subfields:
                model_name = names.make(f"{parent}{_pascal(spec.name)}Item")
                return list[_nested_model(model_name, spec.description, spec.subfields)]
            return list[_SCALAR_TYPES[spec.item_type] if spec.item_type else str]
        case _:
            return _SCALAR_TYPES[spec.type]


def build_extraction_model(analysis: DocumentAnalysis) -> type[BaseModel]:
    """Monta o modelo Pydantic de extração a partir do schema do analista."""
    if not analysis.fields:
        raise SchemaBuildError(
            f"O analista não propôs nenhum campo para '{analysis.document_type}'"
        )

    names = _ModelNames()
    root = _pascal(analysis.document_type) or "Documento"
    model_name = names.make(f"Documento{root}" if root[0].isdigit() else root)
    used = {EXTRA_FIELDS_NAME}
    definitions: dict[str, Any] = {}
    try:
        for i, spec in enumerate(analysis.fields, start=1):
            name = _field_name(spec.name, i, used)
            annotation = _annotation(spec, model_name, names)
            definitions[name] = _definition(annotation, spec.description, spec.required)
        definitions[EXTRA_FIELDS_NAME] = (
            list[ExtraField],
            Field(
                default_factory=list,
                description=(
                    "Informações relevantes presentes no documento que não se encaixam em "
                    "nenhum outro campo do schema, como pares chave/valor"
                ),
            ),
        )
        return create_model(model_name, __doc__=analysis.document_type_label or None, **definitions)
    except Exception as exc:  # create_model falha de vários jeitos (TypeError, PydanticUserError)
        raise SchemaBuildError(
            f"Não foi possível montar o modelo para '{analysis.document_type}': {exc}"
        ) from exc
