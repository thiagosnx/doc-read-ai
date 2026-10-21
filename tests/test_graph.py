"""Pipeline de ponta a ponta pelo composition root, com LLM falso (sem rede)."""

from pathlib import Path

from doc_read_ai.bootstrap import build_use_case
from doc_read_ai.domain.analysis import (
    DocumentAnalysis,
    FieldType,
    PipelineStatus,
    ScalarType,
    ValidationVerdict,
)
from doc_read_ai.domain.errors import LLMError
from doc_read_ai.infrastructure.loaders import LocalFileLoader
from tests.conftest import analysis, field, sub
from tests.fakes import KNOWN_SCHEMAS, FakeLLM, FakeLLMFactory, write_file


def _use_case(config_path: Path, factory: FakeLLMFactory):
    return build_use_case(config_path=config_path, llm_factory=factory, loader=LocalFileLoader())


async def test_valid_nfe_is_extracted(
    config_path: Path, png_file: Path, nfe_factory: FakeLLMFactory
) -> None:
    result = await _use_case(config_path, nfe_factory).execute(png_file)

    assert result.status == PipelineStatus.EXTRACTED, result.errors
    assert result.errors == []
    assert result.media_type == "image/png"
    assert result.verdict and result.verdict.is_valid_document
    assert result.analysis and result.analysis.document_type == "nota_fiscal_eletronica"
    assert result.data is not None
    assert result.data["numero"] == "000123"
    assert result.data["data_emissao"] == "2026-01-15"
    assert result.data["emitente"]["cnpj"] == "12.345.678/0001-90"
    assert [i["descricao"] for i in result.data["itens"]] == ["Parafuso", "Porca"]
    assert result.json_schema and "itens" in result.json_schema["properties"]

    # cada agente rodou uma vez, com o documento anexado
    llms = nfe_factory.llms
    assert [len(llms[a].calls) for a in ("orchestrator", "analyst", "extractor")] == [1, 1, 1]
    assert all(c.document and c.document.filename == "nota.png" for c in nfe_factory.all_calls)
    dynamic = llms["extractor"].calls[0].schema
    assert dynamic not in KNOWN_SCHEMAS and "itens" in dynamic.model_fields
    assert {c.agent for c in nfe_factory.all_calls} == {"orchestrator", "analyst", "extractor"}
    assert [c.name for c in nfe_factory.created] == ["orchestrator", "analyst", "extractor"]


async def test_invalid_document_is_rejected(config_path: Path, png_file: Path) -> None:
    verdict = ValidationVerdict(
        is_valid_document=False, reason="Foto de paisagem, sem texto", legibility="high"
    )
    factory = FakeLLMFactory({"orchestrator": FakeLLM({ValidationVerdict: verdict})})

    result = await _use_case(config_path, factory).execute(png_file)

    assert result.status == PipelineStatus.REJECTED
    assert result.verdict and result.verdict.reason == "Foto de paisagem, sem texto"
    assert result.analysis is None and result.data is None
    assert factory.llms["analyst"].calls == []
    assert factory.llms["extractor"].calls == []


async def test_llm_error_in_extractor_fails(
    config_path: Path,
    pdf_file: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
) -> None:
    factory = FakeLLMFactory(
        {
            "orchestrator": FakeLLM({ValidationVerdict: valid_verdict}),
            "analyst": FakeLLM({DocumentAnalysis: nfe_analysis}),
            "extractor": FakeLLM(extraction=LLMError("resposta truncada (max_tokens)")),
        }
    )

    result = await _use_case(config_path, factory).execute(pdf_file)

    assert result.status == PipelineStatus.FAILED
    assert result.analysis is not None  # o que já foi feito é preservado
    assert result.data is None
    assert any("resposta truncada" in e for e in result.errors)


async def test_unsupported_file_fails_without_llm_calls(
    config_path: Path, tmp_path: Path, nfe_factory: FakeLLMFactory
) -> None:
    path = write_file(tmp_path, "anotacoes.txt", b"apenas texto puro")

    result = await _use_case(config_path, nfe_factory).execute(path)

    assert result.status == PipelineStatus.FAILED
    assert result.errors and "UnsupportedDocumentError" in result.errors[0]
    assert nfe_factory.all_calls == []


async def test_non_fiscal_document_curriculo(
    config_path: Path, pdf_file: Path, valid_verdict: ValidationVerdict
) -> None:
    """O schema é desenhado em runtime: não há lista fixa de tipos de documento."""
    curriculo = analysis(
        "curriculo",
        [
            field("nome", FieldType.STRING),
            field("email", FieldType.STRING, False),
            field(
                "experiencias",
                FieldType.LIST,
                subfields=[
                    sub("empresa", ScalarType.STRING),
                    sub("cargo", ScalarType.STRING),
                    sub("inicio", ScalarType.DATE),
                    sub("fim", ScalarType.DATE, required=False),
                ],
            ),
            field("habilidades", FieldType.LIST, item_type=ScalarType.STRING),
        ],
        label="Currículo",
    )

    def payload(schema: type) -> dict:
        assert "experiencias" in schema.model_fields
        return {
            "nome": "Maria Souza",
            "email": None,
            "experiencias": [
                {"empresa": "Alfa", "cargo": "Dev", "inicio": "2020-03-01", "fim": "2023-01-31"},
                {"empresa": "Beta", "cargo": "Tech Lead", "inicio": "2023-02-01", "fim": None},
            ],
            "habilidades": ["Python", "LangGraph"],
        }

    factory = FakeLLMFactory(
        {
            "orchestrator": FakeLLM({ValidationVerdict: valid_verdict}),
            "analyst": FakeLLM({DocumentAnalysis: curriculo}),
            "extractor": FakeLLM(extraction=payload),
        }
    )

    result = await _use_case(config_path, factory).execute(pdf_file)

    assert result.status == PipelineStatus.EXTRACTED, result.errors
    assert result.media_type == "application/pdf"
    assert result.data["experiencias"][1] == {
        "empresa": "Beta",
        "cargo": "Tech Lead",
        "inicio": "2023-02-01",
        "fim": None,
    }
    assert result.data["habilidades"] == ["Python", "LangGraph"]
    assert result.data["extra_fields"] == []
