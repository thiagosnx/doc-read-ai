"""Pipeline de ponta a ponta pelo composition root, com LLM falso (sem rede)."""

from pathlib import Path

from doc_read_ai.bootstrap import build_use_case
from doc_read_ai.domain.analysis import (
    DocumentAnalysis,
    DocumentClassification,
    FieldType,
    PipelineStatus,
    ScalarType,
    SchemaProposal,
    SchemaSource,
    ValidationVerdict,
)
from doc_read_ai.domain.errors import LLMError
from doc_read_ai.domain.ports import SchemaRegistryPort
from doc_read_ai.infrastructure.loaders import LocalFileLoader
from doc_read_ai.infrastructure.registry import InMemorySchemaRegistry, JsonFileSchemaRegistry
from tests.conftest import NFE_PAYLOAD, agent_llms, analysis, field, sub
from tests.fakes import KNOWN_SCHEMAS, FakeLLM, FakeLLMFactory, write_file

AGENTS = ("orchestrator", "classifier", "analyst", "extractor")


def _use_case(
    config_path: Path, factory: FakeLLMFactory, registry: SchemaRegistryPort | None = None
):
    return build_use_case(
        config_path=config_path,
        llm_factory=factory,
        loader=LocalFileLoader(),
        schema_registry=registry or InMemorySchemaRegistry(),
    )


def _call_counts(factory: FakeLLMFactory) -> list[int]:
    return [len(factory.llms[a].calls) for a in AGENTS]


async def test_valid_nfe_is_extracted(
    config_path: Path, png_file: Path, nfe_factory: FakeLLMFactory
) -> None:
    registry = InMemorySchemaRegistry()
    result = await _use_case(config_path, nfe_factory, registry).execute(png_file)

    assert result.status == PipelineStatus.EXTRACTED, result.errors
    assert result.errors == []
    assert result.media_type == "image/png"
    assert result.verdict and result.verdict.is_valid_document
    assert result.analysis and result.analysis.document_type == "nota_fiscal_eletronica"
    assert result.schema_source == SchemaSource.DESIGNED
    assert result.data is not None
    assert result.data["numero"] == "000123"
    assert result.data["data_emissao"] == "2026-01-15"
    assert result.data["emitente"]["cnpj"] == "12.345.678/0001-90"
    assert [i["descricao"] for i in result.data["itens"]] == ["Parafuso", "Porca"]
    assert result.json_schema and "itens" in result.json_schema["properties"]

    # tipo novo: os quatro agentes rodaram uma vez, com o documento anexado
    assert _call_counts(nfe_factory) == [1, 1, 1, 1]
    assert all(c.document and c.document.filename == "nota.png" for c in nfe_factory.all_calls)
    dynamic = nfe_factory.llms["extractor"].calls[0].schema
    assert dynamic not in KNOWN_SCHEMAS and "itens" in dynamic.model_fields
    assert [c.name for c in nfe_factory.created] == list(AGENTS)
    # e o schema ficou registrado para os próximos documentos do tipo
    assert registry.get("nota_fiscal_eletronica") is not None


async def test_second_document_of_same_type_reuses_registered_schema(
    config_path: Path,
    png_file: Path,
    pdf_file: Path,
    nfe_factory: FakeLLMFactory,
    nfe_analysis: DocumentAnalysis,
) -> None:
    registry = InMemorySchemaRegistry()
    use_case = _use_case(config_path, nfe_factory, registry)

    first = await use_case.execute(png_file)
    second = await use_case.execute(pdf_file)

    assert first.schema_source == SchemaSource.DESIGNED
    assert second.status == PipelineStatus.EXTRACTED, second.errors
    assert second.schema_source == SchemaSource.REGISTRY
    assert second.json_schema == first.json_schema  # mesmo tipo, mesmos campos
    # o analista (a etapa cara) só rodou para o primeiro documento
    assert _call_counts(nfe_factory) == [2, 2, 1, 2]
    # o classificador recebeu o catálogo com o tipo já registrado
    second_instruction = nfe_factory.llms["classifier"].calls[1].instruction
    assert "nota_fiscal_eletronica: Nota Fiscal Eletrônica (NF-e) [teste]" in second_instruction


async def test_registered_label_and_category_win_over_classifier(
    config_path: Path,
    png_file: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
) -> None:
    registry = InMemorySchemaRegistry()
    await _use_case(
        config_path, FakeLLMFactory(agent_llms(valid_verdict, nfe_analysis, NFE_PAYLOAD)), registry
    ).execute(png_file)

    # mesma nota, mas o classificador descreve o tipo com outras palavras
    renamed = nfe_analysis.model_copy(
        update={"document_type_label": "DANFE", "category": "fiscal_nfe"}
    )
    factory = FakeLLMFactory(agent_llms(valid_verdict, renamed, NFE_PAYLOAD))
    result = await _use_case(config_path, factory, registry).execute(png_file)

    assert result.schema_source == SchemaSource.REGISTRY
    assert result.analysis.document_type_label == "Nota Fiscal Eletrônica (NF-e)"
    assert result.analysis.category == "teste"


async def test_document_type_is_normalized_before_registry_lookup(
    config_path: Path,
    png_file: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
) -> None:
    messy = nfe_analysis.model_copy(update={"document_type": "Nota Fiscal Eletrônica"})
    registry = InMemorySchemaRegistry()

    result = await _use_case(
        config_path, FakeLLMFactory(agent_llms(valid_verdict, messy, NFE_PAYLOAD)), registry
    ).execute(png_file)

    assert result.analysis.document_type == "nota_fiscal_eletronica"
    assert [s.document_type for s in registry.list_all()] == ["nota_fiscal_eletronica"]


async def test_schema_that_cannot_be_built_is_not_registered(
    config_path: Path,
    png_file: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
) -> None:
    registry = InMemorySchemaRegistry()
    llms = agent_llms(valid_verdict, nfe_analysis, NFE_PAYLOAD)
    llms["analyst"] = FakeLLM({SchemaProposal: SchemaProposal(fields=[])})  # schema vazio

    result = await _use_case(config_path, FakeLLMFactory(llms), registry).execute(png_file)

    assert result.status == PipelineStatus.FAILED
    assert any("SchemaBuildError" in e for e in result.errors)
    assert registry.list_all() == []
    assert llms["extractor"].calls == []


async def test_registry_persists_between_runs_on_disk(
    config_path: Path,
    png_file: Path,
    tmp_path: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
) -> None:
    directory = tmp_path / "schemas"

    async def run() -> FakeLLMFactory:
        factory = FakeLLMFactory(agent_llms(valid_verdict, nfe_analysis, NFE_PAYLOAD))
        uc = _use_case(config_path, factory, JsonFileSchemaRegistry(directory))
        assert (await uc.execute(png_file)).status == PipelineStatus.EXTRACTED
        return factory

    first, second = await run(), await run()  # dois processos diferentes, mesma pasta

    assert (directory / "nota_fiscal_eletronica.json").is_file()
    assert len(first.llms["analyst"].calls) == 1
    assert second.llms["analyst"].calls == []


async def test_invalid_document_is_rejected(config_path: Path, png_file: Path) -> None:
    verdict = ValidationVerdict(
        is_valid_document=False, reason="Foto de paisagem, sem texto", legibility="high"
    )
    factory = FakeLLMFactory({"orchestrator": FakeLLM({ValidationVerdict: verdict})})

    result = await _use_case(config_path, factory).execute(png_file)

    assert result.status == PipelineStatus.REJECTED
    assert result.verdict and result.verdict.reason == "Foto de paisagem, sem texto"
    assert result.analysis is None and result.data is None
    assert _call_counts(factory) == [1, 0, 0, 0]


async def test_llm_error_in_extractor_fails(
    config_path: Path,
    pdf_file: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
) -> None:
    error = LLMError("resposta truncada (max_tokens)")
    factory = FakeLLMFactory(agent_llms(valid_verdict, nfe_analysis, error))

    result = await _use_case(config_path, factory).execute(pdf_file)

    assert result.status == PipelineStatus.FAILED
    assert result.analysis is not None  # o que já foi feito é preservado
    assert result.data is None
    assert any("resposta truncada" in e for e in result.errors)


async def test_llm_error_in_classifier_fails_before_analyst(
    config_path: Path,
    png_file: Path,
    valid_verdict: ValidationVerdict,
    nfe_analysis: DocumentAnalysis,
) -> None:
    llms = agent_llms(valid_verdict, nfe_analysis, NFE_PAYLOAD)
    llms["classifier"] = FakeLLM({DocumentClassification: LLMError("recusa")})
    factory = FakeLLMFactory(llms)

    result = await _use_case(config_path, factory).execute(png_file)

    assert result.status == PipelineStatus.FAILED
    assert any(e.startswith("[classify]") for e in result.errors)
    assert _call_counts(factory) == [1, 1, 0, 0]


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

    factory = FakeLLMFactory(agent_llms(valid_verdict, curriculo, payload))

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


async def test_stream_emits_one_event_per_step_then_done(
    config_path: Path, png_file: Path, nfe_factory: FakeLLMFactory
) -> None:
    """Contrato do progresso consumido pela API/SSE."""
    document = LocalFileLoader().load(png_file)
    use_case = _use_case(config_path, nfe_factory)

    events = [e async for e in use_case.stream_document(document, run_id="run-1")]

    assert [e.step for e in events] == [
        "validate",
        "classify",
        "resolve_schema",
        "design_schema",
        "build_schema",
        "extract",
        "done",
    ]
    assert events[1].detail["document_type"] == "nota_fiscal_eletronica"
    assert events[3].detail["schema_source"] == "designed"
    assert events[-1].result and events[-1].result.status == PipelineStatus.EXTRACTED
    assert all(e.result is None for e in events[:-1])
