import pytest

from doc_read_ai.domain.config import AgentConfig, ProviderConfig
from doc_read_ai.domain.errors import ProviderNotRegisteredError
from doc_read_ai.infrastructure.llm import LLMRegistry, build_default_llm_factory
from tests.fakes import FakeLLM


def test_unknown_provider_raises() -> None:
    registry = build_default_llm_factory({"anthropic": ProviderConfig()})
    agent = AgentConfig(name="analyst", provider="openai", model="gpt-x")
    with pytest.raises(ProviderNotRegisteredError, match="openai"):
        registry.create(agent)


def test_default_factory_registers_anthropic() -> None:
    assert "anthropic" in build_default_llm_factory({}).registered


def test_custom_provider_strategy_swap() -> None:
    """Trocar de provedor = registrar um builder novo; agentes e grafo não mudam."""
    received: dict = {}

    def build_openai(agent: AgentConfig, provider: ProviderConfig) -> FakeLLM:
        received["agent"], received["provider"] = agent, provider
        return FakeLLM(name=f"openai:{agent.model}")

    providers = {"openai": ProviderConfig(api_key_env="OPENAI_API_KEY", options={"org": "x"})}
    registry = LLMRegistry(providers)
    registry.register("openai", build_openai)

    llm = registry.create(AgentConfig(name="extractor", provider="openai", model="gpt-x"))

    assert isinstance(llm, FakeLLM)
    assert llm.name == "openai:gpt-x"
    assert received["agent"].name == "extractor"
    assert received["provider"].options == {"org": "x"}
    assert registry.registered == ["openai"]
