"""Loader de arquivos locais. O tipo é detectado pelos magic bytes, não pela extensão."""

from pathlib import Path

from doc_read_ai.domain.document import Document
from doc_read_ai.domain.errors import UnsupportedDocumentError
from doc_read_ai.domain.ports import DocumentLoaderPort

# Usado pela CLI só para filtrar diretórios; a validação real é por conteúdo.
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}

_SIGNATURES: list[tuple[bytes, str]] = [
    (b"%PDF", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def _detect_media_type(head: bytes) -> str | None:
    for signature, media_type in _SIGNATURES:
        if head.startswith(signature):
            return media_type
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


class LocalFileLoader(DocumentLoaderPort):
    """Lê PDF/imagem do disco e devolve um Document validado."""

    def __init__(self, max_bytes: int = 32 * 1024 * 1024) -> None:
        self._max_bytes = max_bytes

    def load(self, source: Path) -> Document:
        path = Path(source)
        if not path.exists():
            raise UnsupportedDocumentError(f"Arquivo não encontrado: {path}")
        if not path.is_file():
            raise UnsupportedDocumentError(f"'{path}' não é um arquivo")

        size = path.stat().st_size
        if size == 0:
            raise UnsupportedDocumentError(f"'{path.name}' está vazio")
        if size > self._max_bytes:
            raise UnsupportedDocumentError(
                f"'{path.name}' tem {size / 1024 / 1024:.1f} MB; o limite é "
                f"{self._max_bytes / 1024 / 1024:.0f} MB"
            )

        content = path.read_bytes()
        media_type = _detect_media_type(content[:12])
        if media_type is None:
            raise UnsupportedDocumentError(
                f"'{path.name}' não é um formato suportado. Aceitamos apenas PDF, PNG, JPEG, "
                "WEBP e GIF (verificado pelo conteúdo, não pela extensão)."
            )
        return Document(filename=path.name, media_type=media_type, content=content)
