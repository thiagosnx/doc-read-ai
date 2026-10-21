"""Registry de provedores (Strategy): mapeia `provider` do YAML para o adapter de LLMPort."""

from collections.abc import Callable

from doc_read_ai.domain.config import AgentConfig, ProviderConfig
from doc_read_ai.domain.errors import ProviderNotRegisteredError
from doc_read_ai.domain.ports import LLMFactoryPort, LLMPort
from doc_read_ai.infrastructure.llm.anthropic_adapter import AnthropicLLM

ProviderBuilder = Callable[[AgentConfig, ProviderConfig], LLMPort]


class LLMRegistry(LLMFactoryPort):
    """Cria o LLMPort de cada agente a partir do nome do provider registrado."""

    def __init__(self, providers: dict[str, ProviderConfig]) -> None:
        self._providers = providers
        self._builders: dict[str, ProviderBuilder] = {}

    def register(self, name: str, builder: ProviderBuilder) -> None:
        self._builders[name] = builder

    @property
    def registered(self) -> list[str]:
        return sorted(self._builders)

    def create(self, config: AgentConfig) -> LLMPort:
        builder = self._builders.get(config.provider)
        if builder is None:
            available = ", ".join(self.registered) or "nenhum"
            raise ProviderNotRegisteredError(
                f"Provider '{config.provider}' (agente '{config.name}') não tem adapter "
                f"registrado. Disponíveis: {available}."
            )
        provider = self._providers.get(config.provider) or ProviderConfig()
        return builder(config, provider)


def build_default_llm_factory(providers: dict[str, ProviderConfig]) -> LLMRegistry:
    registry = LLMRegistry(providers)
    registry.register("anthropic", AnthropicLLM)
    # Outros provedores: escreva o adapter de LLMPort e registre aqui, ex:
    #   registry.register("openai", OpenAILLM)
    #   registry.register("vertex", VertexLLM)
    return registry
