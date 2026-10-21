"""Saídas estruturadas de cada agente e o resultado final do pipeline.

Os schemas que os LLMs preenchem (ValidationVerdict, DocumentAnalysis) NÃO podem ser
recursivos nem usar dict livre: structured outputs exige objetos fechados
(additionalProperties: false). Por isso FieldSpec tem no máximo dois níveis.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ScalarType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"


class FieldType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    OBJECT = "object"  # usa `subfields`
    LIST = "list"  # lista de objetos se `subfields` existir, senão lista de `item_type`


class SubFieldSpec(BaseModel):
    """Campo escalar dentro de um objeto ou de um item de lista."""

    name: str = Field(description="Nome em snake_case, sem acentos. Ex: valor_unitario")
    type: ScalarType
    description: str = Field(description="O que o campo representa no documento")
    required: bool = Field(description="True se o campo sempre aparece nesse tipo de documento")


class FieldSpec(BaseModel):
    """Campo de primeiro nível do schema dinâmico proposto pelo analista."""

    name: str = Field(description="Nome em snake_case, sem acentos. Ex: cnpj_emitente")
    type: FieldType
    description: str = Field(description="O que o campo representa no documento")
    required: bool = Field(description="True se o campo sempre aparece nesse tipo de documento")
    item_type: ScalarType | None = Field(
        default=None, description="Tipo dos itens quando type=list e não há subfields"
    )
    subfields: list[SubFieldSpec] = Field(
        default_factory=list,
        description="Campos internos de type=object ou type=list de objetos (ex: itens da nota)",
    )


class ValidationVerdict(BaseModel):
    """Saída do orquestrador: isso é um documento legível e processável?"""

    is_valid_document: bool = Field(
        description="True se for um documento real (com texto/dados estruturados) e legível"
    )
    reason: str = Field(description="Justificativa curta da decisão")
    language: str | None = Field(default=None, description="Idioma principal, ex: pt-BR")
    legibility: Literal["high", "medium", "low"]


class DocumentClassification(BaseModel):
    """Saída do classificador: que documento é esse e como ele está organizado."""

    document_type: str = Field(
        description="Identificador em snake_case. Use o id exato de um tipo já registrado "
        "quando o documento for desse tipo. Ex: nota_fiscal_eletronica"
    )
    document_type_label: str = Field(description="Nome legível. Ex: Nota Fiscal Eletrônica (NF-e)")
    category: str = Field(description="Categoria ampla. Ex: fiscal, financeiro, identificacao")
    confidence: float = Field(description="Confiança na classificação, de 0 a 1")
    summary: str = Field(description="Resumo de 1-2 frases do conteúdo")
    structure_notes: str = Field(description="Como o documento está organizado (seções, tabelas)")


class SchemaProposal(BaseModel):
    """Saída do analista: o schema de campos a extrair para um tipo de documento novo."""

    fields: list[FieldSpec] = Field(description="Schema dos dados a extrair deste tipo")


class DocumentAnalysis(DocumentClassification):
    """Classificação + schema usado na extração (do registro ou recém-desenhado)."""

    fields: list[FieldSpec] = Field(description="Schema dos dados a extrair deste documento")


class SchemaSource(StrEnum):
    REGISTRY = "registry"  # reaproveitado do registro de schemas
    DESIGNED = "designed"  # desenhado agora pelo analista e salvo no registro


class RegisteredSchema(BaseModel):
    """Schema estável de um tipo de documento: todo documento do mesmo tipo usa os mesmos campos."""

    document_type: str
    document_type_label: str
    category: str
    fields: list[FieldSpec]
    created_at: datetime


class PipelineStatus(StrEnum):
    EXTRACTED = "extracted"
    REJECTED = "rejected"
    FAILED = "failed"


class ExtractionResult(BaseModel):
    """Resultado final entregue para quem chamou o caso de uso."""

    filename: str
    media_type: str
    status: PipelineStatus
    verdict: ValidationVerdict | None = None
    analysis: DocumentAnalysis | None = None
    schema_source: SchemaSource | None = None
    data: dict[str, Any] | None = None
    json_schema: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
