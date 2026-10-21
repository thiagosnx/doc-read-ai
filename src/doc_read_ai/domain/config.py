"""Configuração declarativa dos agentes (espelha config/agents.yml)."""

from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Um agente = papel + provedor + modelo. `params` são repassados ao adapter do provedor."""

    name: str
    provider: str = Field(description="Chave registrada no LLM registry. Ex: anthropic")
    model: str
    max_tokens: int = 16000
    system_prompt: str | None = Field(
        default=None, description="Sobrescreve o prompt padrão do agente quando definido"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Opções específicas do provedor (effort, fallbacks...)"
    )


class ProviderConfig(BaseModel):
    """Credenciais/cliente de um provedor. `options` também é específico do provedor."""

    api_key_env: str | None = None
    timeout_seconds: float = 300.0
    max_retries: int = 3
    options: dict[str, Any] = Field(default_factory=dict)


class SchemaRegistryConfig(BaseModel):
    """Onde ficam os schemas por tipo de documento. Desligado = schema novo por documento."""

    enabled: bool = True
    path: str = "schemas"


class AppConfig(BaseModel):
    agents: dict[str, AgentConfig]
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    schema_registry: SchemaRegistryConfig = Field(default_factory=SchemaRegistryConfig)

    def agent(self, name: str) -> AgentConfig:
        if name not in self.agents:
            raise KeyError(f"Agente '{name}' não está definido no YAML de configuração")
        return self.agents[name]
