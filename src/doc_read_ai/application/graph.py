"""Grafo LangGraph do pipeline autônomo.

validate (orquestrador) -> analyze (analista) -> build_schema (determinístico) -> extract (extrator)

Qualquer DocReadError num nó vira status FAILED e encerra o grafo. Para estender (ex: um nó
`review` depois de `extract`), crie o nó, troque a aresta `extract -> END` por
`_continue_or_end("review")` e adicione `review -> END`.
"""

import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from doc_read_ai.application.agents import AnalystAgent, ExtractorAgent, OrchestratorAgent
from doc_read_ai.application.schema_builder import build_extraction_model
from doc_read_ai.domain.analysis import DocumentAnalysis, PipelineStatus, ValidationVerdict
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.errors import DocReadError


class GraphState(TypedDict, total=False):
    document: Document
    verdict: ValidationVerdict
    analysis: DocumentAnalysis
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


def build_graph(
    orchestrator: OrchestratorAgent, analyst: AnalystAgent, extractor: ExtractorAgent
) -> Any:
    async def validate(state: GraphState) -> dict[str, Any]:
        verdict = await orchestrator.run(state["document"])
        update: dict[str, Any] = {"verdict": verdict}
        if not verdict.is_valid_document:
            update["status"] = PipelineStatus.REJECTED
        return update

    async def analyze(state: GraphState) -> dict[str, Any]:
        return {"analysis": await analyst.run(state["document"], state["verdict"])}

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
        ("analyze", analyze),
        ("build_schema", build_schema),
        ("extract", extract),
    ]:
        graph.add_node(name, _guarded(name, node))

    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", _continue_or_end("analyze"), ["analyze", END])
    graph.add_conditional_edges("analyze", _continue_or_end("build_schema"), ["build_schema", END])
    graph.add_conditional_edges("build_schema", _continue_or_end("extract"), ["extract", END])
    graph.add_edge("extract", END)
    return graph.compile()
