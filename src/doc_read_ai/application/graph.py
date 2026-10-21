"""Grafo LangGraph do pipeline autônomo.

validate (orquestrador) -> classify (classificador) -> resolve_schema (registro)
    -> tipo conhecido: build_schema
    -> tipo novo:      design_schema (analista, salva no registro) -> build_schema
build_schema (determinístico) -> extract (extrator)

Qualquer DocReadError num nó vira status FAILED e encerra o grafo. Para estender (ex: um nó
`review` depois de `extract`), crie o nó, troque a aresta `extract -> END` por
`_continue_or_end("review")` e adicione `review -> END`.
"""

import operator
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from doc_read_ai.application.agents import (
    AnalystAgent,
    ClassifierAgent,
    ExtractorAgent,
    OrchestratorAgent,
)
from doc_read_ai.application.schema_builder import build_extraction_model, normalize_type_id
from doc_read_ai.domain.analysis import (
    DocumentAnalysis,
    DocumentClassification,
    PipelineStatus,
    RegisteredSchema,
    SchemaSource,
    ValidationVerdict,
)
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.errors import DocReadError
from doc_read_ai.domain.ports import SchemaRegistryPort


class GraphState(TypedDict, total=False):
    document: Document
    verdict: ValidationVerdict
    classification: DocumentClassification
    analysis: DocumentAnalysis
    schema_source: SchemaSource
    extraction_model: type[BaseModel]
    data: dict[str, Any]
    json_schema: dict[str, Any]
    status: PipelineStatus  # só é definido quando o pipeline termina
    errors: Annotated[list[str], operator.add]


Node = Callable[[GraphState], Awaitable[dict[str, Any]]]


def _guarded(name: str, node: Node) -> Node:
    """Converte DocReadError em status FAILED + mensagem de erro, sem derrubar o grafo."""

    async def wrapper(state: GraphState) -> dict[str, Any]:
        try:
            return await node(state)
        except DocReadError as exc:
            return {
                "status": PipelineStatus.FAILED,
                "errors": [f"[{name}] {type(exc).__name__}: {exc}"],
            }

    return wrapper


def _continue_or_end(next_node: str) -> Callable[[GraphState], str]:
    """Segue para o próximo nó, a menos que um status final (REJECTED/FAILED) já exista."""

    def route(state: GraphState) -> str:
        return END if state.get("status") else next_node

    return route


def _after_resolve(state: GraphState) -> str:
    if state.get("status"):
        return END
    return "build_schema" if "analysis" in state else "design_schema"


def build_graph(
    orchestrator: OrchestratorAgent,
    classifier: ClassifierAgent,
    analyst: AnalystAgent,
    extractor: ExtractorAgent,
    registry: SchemaRegistryPort,
) -> Any:
    async def validate(state: GraphState) -> dict[str, Any]:
        verdict = await orchestrator.run(state["document"])
        update: dict[str, Any] = {"verdict": verdict}
        if not verdict.is_valid_document:
            update["status"] = PipelineStatus.REJECTED
        return update

    async def classify(state: GraphState) -> dict[str, Any]:
        classification = await classifier.run(
            state["document"], state["verdict"], registry.list_all()
        )
        type_id = normalize_type_id(classification.document_type)
        return {"classification": classification.model_copy(update={"document_type": type_id})}

    async def resolve_schema(state: GraphState) -> dict[str, Any]:
        classification = state["classification"]
        registered = registry.get(classification.document_type)
        if registered is None:
            return {}  # tipo novo: o analista desenha o schema
        # Nome e categoria vêm do registro para que o mesmo tipo saia sempre igual.
        analysis = DocumentAnalysis(
            **classification.model_dump(exclude={"document_type_label", "category"}),
            document_type_label=registered.document_type_label,
            category=registered.category,
            fields=registered.fields,
        )
        return {"analysis": analysis, "schema_source": SchemaSource.REGISTRY}

    async def design_schema(state: GraphState) -> dict[str, Any]:
        classification = state["classification"]
        fields = await analyst.run(state["document"], classification)
        analysis = DocumentAnalysis(**classification.model_dump(), fields=fields)
        build_extraction_model(analysis)  # só registra schema que vira modelo válido
        registry.save(
            RegisteredSchema(
                document_type=analysis.document_type,
                document_type_label=analysis.document_type_label,
                category=analysis.category,
                fields=fields,
                created_at=datetime.now(UTC),
            )
        )
        return {"analysis": analysis, "schema_source": SchemaSource.DESIGNED}

    async def build_schema(state: GraphState) -> dict[str, Any]:
        model = build_extraction_model(state["analysis"])
        return {"extraction_model": model, "json_schema": model.model_json_schema()}

    async def extract(state: GraphState) -> dict[str, Any]:
        instance = await extractor.run(
            state["document"], state["analysis"], state["extraction_model"]
        )
        return {"data": instance.model_dump(mode="json"), "status": PipelineStatus.EXTRACTED}

    graph = StateGraph(GraphState)
    for name, node in [
        ("validate", validate),
        ("classify", classify),
        ("resolve_schema", resolve_schema),
        ("design_schema", design_schema),
        ("build_schema", build_schema),
        ("extract", extract),
    ]:
        graph.add_node(name, _guarded(name, node))

    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", _continue_or_end("classify"), ["classify", END])
    graph.add_conditional_edges(
        "classify", _continue_or_end("resolve_schema"), ["resolve_schema", END]
    )
    graph.add_conditional_edges(
        "resolve_schema", _after_resolve, ["build_schema", "design_schema", END]
    )
    graph.add_conditional_edges(
        "design_schema", _continue_or_end("build_schema"), ["build_schema", END]
    )
    graph.add_conditional_edges("build_schema", _continue_or_end("extract"), ["extract", END])
    graph.add_edge("extract", END)
    return graph.compile()
