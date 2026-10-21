"""Composition root: único lugar que conhece todas as classes concretas."""

import logging
from pathlib import Path

from doc_read_ai.application.agents import AnalystAgent, ExtractorAgent, OrchestratorAgent
from doc_read_ai.application.graph import build_graph
from doc_read_ai.application.use_cases import ProcessDocumentUseCase
from doc_read_ai.domain.config import AppConfig
from doc_read_ai.domain.ports import DocumentLoaderPort, LLMFactoryPort
from doc_read_ai.infrastructure.config import Settings, load_app_config, load_env_file
from doc_read_ai.infrastructure.llm import build_default_llm_factory
from doc_read_ai.infrastructure.loaders import LocalFileLoader


def load_config(config_path: Path | None = None) -> AppConfig:
    """Carrega .env e o YAML de agentes (argumento > DOC_READ_CONFIG > padrão)."""
    load_env_file()
    return load_app_config(config_path or Settings().doc_read_config)


def build_use_case(
    config_path: Path | None = None,
    llm_factory: LLMFactoryPort | None = None,
    loader: DocumentLoaderPort | None = None,
) -> ProcessDocumentUseCase:
    """Monta agentes, grafo e caso de uso. Factory/loader injetáveis permitem testes sem rede."""
    app_config = load_config(config_path)
    factory = llm_factory or build_default_llm_factory(app_config.providers)

    def make(name: str, agent_cls):
        cfg = app_config.agent(name)
        return agent_cls(factory.create(cfg), cfg)

    graph = build_graph(
        make("orchestrator", OrchestratorAgent),
        make("analyst", AnalystAgent),
        make("extractor", ExtractorAgent),
    )
    return ProcessDocumentUseCase(loader or LocalFileLoader(), graph)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Bibliotecas HTTP são verbosas demais em INFO.
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))
