from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_TEXT = {".txt", ".md"}


def load_document(path: str | Path) -> str:
    document = Path(path)
    suffix = document.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(document))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix in SUPPORTED_TEXT:
        return document.read_text(encoding="utf-8")
    raise ValueError(f"Formato no soportado: {document.suffix}")
