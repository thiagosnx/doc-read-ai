from pathlib import Path

import pytest

from doc_read_ai.domain.document import DocumentKind
from doc_read_ai.domain.errors import UnsupportedDocumentError
from doc_read_ai.infrastructure.loaders import SUPPORTED_EXTENSIONS, LocalFileLoader
from tests.fakes import jpeg_bytes, pdf_bytes, png_bytes, write_file


@pytest.mark.parametrize(
    ("name", "content", "media_type", "kind"),
    [
        ("a.png", png_bytes(), "image/png", DocumentKind.IMAGE),
        ("a.pdf", pdf_bytes(), "application/pdf", DocumentKind.PDF),
        ("a.jpg", jpeg_bytes(), "image/jpeg", DocumentKind.IMAGE),
    ],
)
def test_detects_supported_types(
    tmp_path: Path, name: str, content: bytes, media_type: str, kind: DocumentKind
) -> None:
    doc = LocalFileLoader().load(write_file(tmp_path, name, content))
    assert doc.filename == name
    assert doc.media_type == media_type
    assert doc.kind == kind
    assert doc.content == content


def test_detection_uses_content_not_extension(tmp_path: Path) -> None:
    doc = LocalFileLoader().load(write_file(tmp_path, "na_verdade_png.pdf", png_bytes()))
    assert doc.media_type == "image/png"


@pytest.mark.parametrize(
    ("name", "content"),
    [("notas.txt", b"apenas texto"), ("falso.png", b"isso nao e png"), ("vazio.pdf", b"")],
)
def test_rejects_unsupported(tmp_path: Path, name: str, content: bytes) -> None:
    with pytest.raises(UnsupportedDocumentError):
        LocalFileLoader().load(write_file(tmp_path, name, content))


def test_rejects_too_large(tmp_path: Path) -> None:
    path = write_file(tmp_path, "grande.pdf", pdf_bytes())
    with pytest.raises(UnsupportedDocumentError, match="limite"):
        LocalFileLoader(max_bytes=10).load(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedDocumentError):
        LocalFileLoader().load(tmp_path / "nao_existe.pdf")


def test_supported_extensions() -> None:
    assert {".pdf", ".png", ".jpg"} <= SUPPORTED_EXTENSIONS
