"""Dublês de teste: LLM e factory falsos (sem rede) e arquivos mínimos válidos."""

import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from doc_read_ai.domain.analysis import DocumentAnalysis, ValidationVerdict
from doc_read_ai.domain.config import AgentConfig
from doc_read_ai.domain.document import Document
from doc_read_ai.domain.errors import LLMError
from doc_read_ai.domain.ports import LLMFactoryPort, LLMPort

KNOWN_SCHEMAS = (ValidationVerdict, DocumentAnalysis)

# Payload do modelo dinâmico: dict fixo, callable(schema) -> dict, ou exceção a levantar.
ExtractionResponse = dict[str, Any] | Callable[[type[BaseModel]], dict[str, Any]] | Exception


@dataclass
class Call:
    agent: str
    system_prompt: str
    instruction: str
    schema: type[BaseModel]
    document: Document | None


class FakeLLM(LLMPort):
    """Devolve respostas prontas conforme o output_schema pedido e grava cada chamada."""

    def __init__(
        self,
        responses: dict[type[BaseModel], BaseModel | Exception] | None = None,
        extraction: ExtractionResponse | None = None,
        name: str = "fake",
    ) -> None:
        self.responses = responses or {}
        self.extraction = extraction
        self.name = name
        self.calls: list[Call] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        instruction: str,
        output_schema: type[BaseModel],
        document: Document | None = None,
    ) -> Any:
        self.calls.append(Call(self.name, system_prompt, instruction, output_schema, document))

        if output_schema in KNOWN_SCHEMAS:
            response = self.responses.get(output_schema)
            if response is None:
                raise LLMError(f"FakeLLM '{self.name}' sem resposta para {output_schema.__name__}")
        else:
            response = self.extraction
            if response is None:
                raise LLMError(f"FakeLLM '{self.name}' sem resposta para o schema dinâmico")
            if callable(response):
                response = response(output_schema)
            if isinstance(response, dict):
                response = output_schema.model_validate(response)

        if isinstance(response, Exception):
            raise response
        return response


class FakeLLMFactory(LLMFactoryPort):
    """Mapeia nome do agente -> FakeLLM. Agentes sem entrada recebem um FakeLLM vazio."""

    def __init__(self, llms: dict[str, FakeLLM] | None = None) -> None:
        self.llms = dict(llms or {})
        self.created: list[AgentConfig] = []

    def create(self, config: AgentConfig) -> LLMPort:
        self.created.append(config)
        llm = self.llms.setdefault(config.name, FakeLLM())
        llm.name = config.name
        return llm

    @property
    def all_calls(self) -> list[Call]:
        return [c for llm in self.llms.values() for c in llm.calls]


# --- arquivos mínimos -------------------------------------------------------------------


def png_bytes() -> bytes:
    """PNG 1x1 válido."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def pdf_bytes() -> bytes:
    """PDF mínimo de uma página em branco."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 10 10]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )


def jpeg_bytes() -> bytes:
    """Só os magic bytes de JPEG (suficiente para detecção)."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 16 + b"\xff\xd9"


def write_file(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.write_bytes(content)
    return path
