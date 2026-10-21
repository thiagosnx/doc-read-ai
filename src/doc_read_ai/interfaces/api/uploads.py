"""Validação de upload em memória: mesmo critério do LocalFileLoader (magic bytes), sem disco."""

from doc_read_ai.domain.document import SUPPORTED_MEDIA_TYPES, Document
from doc_read_ai.domain.errors import UnsupportedDocumentError

# Detecção compartilhada com o loader de arquivos (fonte única das assinaturas aceitas).
from doc_read_ai.infrastructure.loaders import detect_media_type


class UploadTooLargeError(UnsupportedDocumentError):
    """Arquivo acima de DOC_READ_MAX_UPLOAD_MB."""


class EmptyUploadError(UnsupportedDocumentError):
    """Arquivo sem conteúdo."""


def document_from_bytes(filename: str, content: bytes, max_bytes: int) -> Document:
    """Valida tamanho e formato pelo conteúdo e devolve o Document (só em memória)."""
    if not content:
        raise EmptyUploadError(f"'{filename}' está vazio")
    if len(content) > max_bytes:
        raise UploadTooLargeError(
            f"'{filename}' passa do limite de {max_bytes / 1024 / 1024:.0f} MB"
        )
    media_type = detect_media_type(content[:12])
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedDocumentError(
            f"'{filename}' não é um formato suportado. Aceitamos apenas PDF, PNG, JPEG, WEBP e "
            "GIF (verificado pelo conteúdo, não pela extensão)."
        )
    return Document(filename=filename, media_type=media_type, content=content)
