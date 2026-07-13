from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
SUPPORTED_DOCUMENTS = {".pdf", *SUPPORTED_TEXT}


@dataclass(frozen=True)
class DocumentPage:
    """Extracted text for one logical document page."""

    text: str
    page: int | None
    title: str


def load_document(path: str | Path) -> str:
    """Load a supported document as plain text."""
    return "\n".join(page.text for page in load_document_pages(path))


def load_document_pages(path: str | Path) -> list[DocumentPage]:
    """Load a supported document as page-level text records."""
    document = Path(path)
    suffix = document.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(document))
        title = ""
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip()
        return [
            DocumentPage(text=page.extract_text() or "", page=index, title=title)
            for index, page in enumerate(reader.pages, start=1)
        ]
    if suffix in SUPPORTED_TEXT:
        return [DocumentPage(text=document.read_text(encoding="utf-8-sig"), page=None, title="")]
    raise ValueError(f"Formato no soportado: {document.suffix}")
