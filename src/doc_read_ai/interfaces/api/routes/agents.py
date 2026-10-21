"""Configuração dos agentes, sem segredos (nem valores de variáveis de ambiente)."""

import asyncio
import os

from fastapi import APIRouter

from doc_read_ai.interfaces.api.container import ApiContainer
from doc_read_ai.interfaces.api.deps import ContainerDep
from doc_read_ai.interfaces.api.schemas import AgentOut, AgentsOut, ProviderOut, SchemaRegistryOut

router = APIRouter(prefix="/agents", tags=["agents"])


def agent_list(container: ApiContainer) -> list[AgentOut]:
    return [AgentOut.from_config(a) for a in container.app_config.agents.values()]


async def schema_registry_info(container: ApiContainer) -> SchemaRegistryOut:
    schemas = await asyncio.to_thread(container.schema_registry.list_all)
    return SchemaRegistryOut(
        enabled=container.app_config.schema_registry.enabled, count=len(schemas)
    )


@router.get("", summary="Agentes configurados (provedor, modelo, parâmetros)")
async def list_agents(container: ContainerDep) -> AgentsOut:
    providers = [
        ProviderOut(
            name=name,
            timeout_seconds=p.timeout_seconds,
            max_retries=p.max_retries,
            credentials_configured=(
                not p.api_key_env or bool(os.environ.get(p.api_key_env, "").strip())
            ),
        )
        for name, p in container.app_config.providers.items()
    ]
    return AgentsOut(
        agents=agent_list(container),
        providers=providers,
        schema_registry=await schema_registry_info(container),
    )
