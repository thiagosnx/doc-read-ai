"""Entidade Document: o arquivo de entrada, já carregado em memória e agnóstico de provedor."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DocumentKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"


SUPPORTED_MEDIA_TYPES: dict[str, DocumentKind] = {
    "application/pdf": DocumentKind.PDF,
    "image/png": DocumentKind.IMAGE,
    "image/jpeg": DocumentKind.IMAGE,
    "image/webp": DocumentKind.IMAGE,
    "image/gif": DocumentKind.IMAGE,
}


class Document(BaseModel):
    """Arquivo bruto (PDF ou imagem). Cada adapter de LLM decide como enviá-lo ao provedor."""

    model_config = ConfigDict(frozen=True)

    filename: str
    media_type: str
    content: bytes

    @property
    def kind(self) -> DocumentKind:
        return SUPPORTED_MEDIA_TYPES[self.media_type]

    @property
    def size_bytes(self) -> int:
        return len(self.content)
