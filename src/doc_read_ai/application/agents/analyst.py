"""Analista: descobre o tipo do documento, entende a estrutura e desenha o schema de extração."""

from doc_read_ai.application.agents.base import BaseAgent
from doc_read_ai.application.prompts import ANALYST_SYSTEM_PROMPT
from doc_read_ai.domain.analysis import DocumentAnalysis, ValidationVerdict
from doc_read_ai.domain.document import Document


class AnalystAgent(BaseAgent):
    default_system_prompt = ANALYST_SYSTEM_PROMPT

    async def run(self, document: Document, verdict: ValidationVerdict) -> DocumentAnalysis:
        instruction = (
            "Analise o documento anexado: identifique o tipo, descreva a estrutura e desenhe o "
            "schema de campos a extrair.\n\n"
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
        return await self._generate(instruction, DocumentAnalysis, document)
