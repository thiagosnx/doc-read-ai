"""Orquestrador: porteiro do pipeline. Decide se o arquivo é um documento legível."""

from doc_read_ai.application.agents.base import BaseAgent
from doc_read_ai.application.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from doc_read_ai.domain.analysis import ValidationVerdict
from doc_read_ai.domain.document import Document


class OrchestratorAgent(BaseAgent):
    default_system_prompt = ORCHESTRATOR_SYSTEM_PROMPT

    async def run(self, document: Document) -> ValidationVerdict:
        instruction = (
            "Avalie o arquivo anexado e decida se ele é um documento legível que deve seguir "
            "para extração.\n"
            f"Nome do arquivo: {document.filename} (pode não refletir o conteúdo)\n"
            f"Formato: {document.media_type}"
        )
        return await self._generate(instruction, ValidationVerdict, document)
