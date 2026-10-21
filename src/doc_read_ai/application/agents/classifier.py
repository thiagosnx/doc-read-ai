"""Classificador: identifica o tipo do documento, reaproveitando ids do registro de schemas."""

from doc_read_ai.application.agents.base import BaseAgent
from doc_read_ai.application.prompts import CLASSIFIER_SYSTEM_PROMPT
from doc_read_ai.domain.analysis import DocumentClassification, RegisteredSchema, ValidationVerdict
from doc_read_ai.domain.document import Document


class ClassifierAgent(BaseAgent):
    default_system_prompt = CLASSIFIER_SYSTEM_PROMPT

    async def run(
        self, document: Document, verdict: ValidationVerdict, catalog: list[RegisteredSchema]
    ) -> DocumentClassification:
        instruction = (
            "Classifique o documento anexado e descreva a estrutura dele.\n\n"
            f"{_catalog_text(catalog)}\n\n"
            "Contexto da triagem inicial:\n"
            f"- Idioma principal: {verdict.language or 'não identificado'}\n"
            f"- Legibilidade: {verdict.legibility}\n"
            f"- Observação: {verdict.reason}"
        )
        if verdict.legibility != "high":
            instruction += (
                "\n\nA legibilidade não é alta: registre em structure_notes os trechos difíceis "
                "e considere isso na confiança da classificação."
            )
        return await self._generate(instruction, DocumentClassification, document)


def _catalog_text(catalog: list[RegisteredSchema]) -> str:
    if not catalog:
        return "Catálogo de tipos já registrados: vazio (qualquer tipo será novo)."
    lines = [
        f"- {s.document_type}: {s.document_type_label} [{s.category}]"
        for s in sorted(catalog, key=lambda s: s.document_type)
    ]
    return "Catálogo de tipos já registrados (id: nome [categoria]):\n" + "\n".join(lines)
