"""Analista: desenha o schema de extração de um tipo de documento que o sistema não conhece."""

from doc_read_ai.application.agents.base import BaseAgent
from doc_read_ai.application.prompts import ANALYST_SYSTEM_PROMPT
from doc_read_ai.domain.analysis import DocumentClassification, FieldSpec, SchemaProposal
from doc_read_ai.domain.document import Document


class AnalystAgent(BaseAgent):
    default_system_prompt = ANALYST_SYSTEM_PROMPT

    async def run(
        self, document: Document, classification: DocumentClassification
    ) -> list[FieldSpec]:
        instruction = (
            "Desenhe o schema de campos a extrair para este tipo de documento, usando o "
            "exemplar anexado como referência.\n\n"
            f"Tipo: {classification.document_type_label} ({classification.document_type})\n"
            f"Categoria: {classification.category}\n"
            f"Resumo deste exemplar: {classification.summary}\n"
            f"Estrutura deste exemplar: {classification.structure_notes}"
        )
        proposal = await self._generate(instruction, SchemaProposal, document)
        return proposal.fields
