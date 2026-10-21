from pathlib import Path
from typing import Any

import pytest

from doc_read_ai.domain.analysis import (
    DocumentAnalysis,
    FieldSpec,
    FieldType,
    ScalarType,
    SubFieldSpec,
    ValidationVerdict,
)
from tests.fakes import FakeLLM, FakeLLMFactory, pdf_bytes, png_bytes, write_file

ROOT = Path(__file__).resolve().parents[1]


def field(name: str, type_: FieldType, required: bool = True, **kw: Any) -> FieldSpec:
    return FieldSpec(name=name, type=type_, description=name, required=required, **kw)


def sub(name: str, type_: ScalarType, required: bool = True) -> SubFieldSpec:
    return SubFieldSpec(name=name, type=type_, description=name, required=required)


def analysis(document_type: str, fields: list[FieldSpec], label: str = "") -> DocumentAnalysis:
    return DocumentAnalysis(
        document_type=document_type,
        document_type_label=label or document_type,
        category="teste",
        confidence=0.95,
        summary=f"Documento do tipo {document_type}.",
        structure_notes="Cabeçalho e tabela.",
        fields=fields,
    )


@pytest.fixture
def config_path() -> Path:
    return ROOT / "config" / "agents.yml"


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    return write_file(tmp_path, "nota.png", png_bytes())


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    return write_file(tmp_path, "curriculo.pdf", pdf_bytes())


@pytest.fixture
def valid_verdict() -> ValidationVerdict:
    return ValidationVerdict(
        is_valid_document=True, reason="Documento legível", language="pt-BR", legibility="high"
    )


@pytest.fixture
def nfe_analysis() -> DocumentAnalysis:
    return analysis(
        "nota_fiscal_eletronica",
        [
            field("numero", FieldType.STRING),
            field("data_emissao", FieldType.DATE),
            field("valor_total", FieldType.NUMBER),
            field(
                "emitente",
                FieldType.OBJECT,
                subfields=[
                    sub("cnpj", ScalarType.STRING),
                    sub("razao_social", ScalarType.STRING, required=False),
                ],
            ),
            field(
                "itens",
                FieldType.LIST,
                subfields=[
                    sub("descricao", ScalarType.STRING),
                    sub("quantidade", ScalarType.INTEGER),
                    sub("valor_unitario", ScalarType.NUMBER),
                ],
            ),
        ],
        label="Nota Fiscal Eletrônica (NF-e)",
    )


NFE_PAYLOAD: dict[str, Any] = {
    "numero": "000123",
    "data_emissao": "2026-01-15",
    "valor_total": 150.5,
    "emitente": {"cnpj": "12.345.678/0001-90", "razao_social": "ACME LTDA"},
    "itens": [
        {"descricao": "Parafuso", "quantidade": 10, "valor_unitario": 5.05},
        {"descricao": "Porca", "quantidade": 20, "valor_unitario": 5.0},
    ],
    "extra_fields": [{"key": "chave_acesso", "value": "3526 0112 3456"}],
}


@pytest.fixture
def nfe_factory(valid_verdict: ValidationVerdict, nfe_analysis: DocumentAnalysis) -> FakeLLMFactory:
    return FakeLLMFactory(
        {
            "orchestrator": FakeLLM({ValidationVerdict: valid_verdict}),
            "analyst": FakeLLM({DocumentAnalysis: nfe_analysis}),
            "extractor": FakeLLM(extraction=NFE_PAYLOAD),
        }
    )
