"""Extrator genérico: preenche o modelo dinâmico (qualquer tipo de documento)."""

from pydantic import BaseModel

from doc_read_ai.application.agents.base import BaseAgent
from doc_read_ai.application.prompts import EXTRACTOR_SYSTEM_PROMPT
from doc_read_ai.domain.analysis import DocumentAnalysis
from doc_read_ai.domain.document import Document


class ExtractorAgent(BaseAgent):
    default_system_prompt = EXTRACTOR_SYSTEM_PROMPT

    async def run(
        self, document: Document, analysis: DocumentAnalysis, model: type[BaseModel]
    ) -> BaseModel:
        instruction = (
            "Extraia os dados do documento anexado preenchendo o schema de saída.\n\n"
            f"Tipo do documento: {analysis.document_type_label} ({analysis.document_type})\n"
            f"Categoria: {analysis.category}\n"
            f"Resumo: {analysis.summary}\n"
            f"Estrutura (notas do analista): {analysis.structure_notes}"
        )
        return await self._generate(instruction, model, document)
