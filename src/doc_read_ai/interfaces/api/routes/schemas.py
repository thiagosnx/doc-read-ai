"""Registro de schemas: um schema estável por tipo de documento."""

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, status

from doc_read_ai.application.schema_builder import build_extraction_model
from doc_read_ai.domain.analysis import DocumentAnalysis, RegisteredSchema
from doc_read_ai.domain.errors import SchemaBuildError
from doc_read_ai.interfaces.api.deps import ContainerDep
from doc_read_ai.interfaces.api.schemas import ErrorOut, SchemaIn, SchemaOut, SchemaSummaryOut

router = APIRouter(prefix="/schemas", tags=["schemas"])

DocumentType = Annotated[
    str, Path(pattern=r"^[a-z0-9_]+$", max_length=100, description="Ex: nota_fiscal_eletronica")
]


def _json_schema(schema: RegisteredSchema) -> dict[str, Any]:
    """Monta o modelo de extração; levanta SchemaBuildError se os campos não fecham."""
    analysis = DocumentAnalysis(
        document_type=schema.document_type,
        document_type_label=schema.document_type_label,
        category=schema.category,
        confidence=1.0,
        summary="",
        structure_notes="",
        fields=schema.fields,
    )
    return build_extraction_model(analysis).model_json_schema()


def _to_out(schema: RegisteredSchema) -> SchemaOut:
    try:
        json_schema = _json_schema(schema)
    except SchemaBuildError:
        json_schema = None  # editado à mão e quebrado: mostra os campos mesmo assim
    return SchemaOut(**schema.model_dump(), json_schema=json_schema)


async def _get_schema(container: ContainerDep, document_type: str) -> RegisteredSchema:
    schema = await asyncio.to_thread(container.schema_registry.get, document_type)
    if schema is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Nenhum schema registrado para '{document_type}'"
        )
    return schema


@router.get("", summary="Lista os tipos de documento com schema registrado")
async def list_schemas(container: ContainerDep) -> list[SchemaSummaryOut]:
    schemas = await asyncio.to_thread(container.schema_registry.list_all)
    return [SchemaSummaryOut.from_schema(s) for s in schemas]


@router.get(
    "/{document_type}",
    summary="Campos e JSON Schema de um tipo",
    responses={404: {"model": ErrorOut}},
)
async def get_schema(container: ContainerDep, document_type: DocumentType) -> SchemaOut:
    return _to_out(await _get_schema(container, document_type))


@router.put(
    "/{document_type}",
    summary="Cria ou substitui o schema de um tipo",
    responses={422: {"model": ErrorOut, "description": "Campos não formam um modelo válido"}},
)
async def put_schema(
    container: ContainerDep, document_type: DocumentType, body: SchemaIn
) -> SchemaOut:
    """O próximo documento desse tipo será extraído com estes campos."""
    current = await asyncio.to_thread(container.schema_registry.get, document_type)
    schema = RegisteredSchema(
        document_type=document_type,
        document_type_label=body.document_type_label,
        category=body.category,
        fields=body.fields,
        created_at=current.created_at if current else datetime.now(UTC),
    )
    json_schema = _json_schema(schema)  # SchemaBuildError -> 422 (handler global)
    await asyncio.to_thread(container.schema_registry.save, schema)
    return SchemaOut(**schema.model_dump(), json_schema=json_schema)


@router.delete(
    "/{document_type}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove o schema (o próximo documento do tipo terá o schema redesenhado)",
    responses={404: {"model": ErrorOut}},
)
async def delete_schema(container: ContainerDep, document_type: DocumentType) -> None:
    if not await asyncio.to_thread(container.schema_registry.delete, document_type):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Nenhum schema registrado para '{document_type}'"
        )
