from __future__ import annotations

from pathlib import Path

from src.rag.chunking import chunk_text
from src.rag.document_loader import load_document


def collect_chunks(documents_dir: str | Path, chunk_size: int = 700, overlap: int = 100) -> list[dict]:
    records: list[dict] = []
    root = Path(documents_dir)
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        text = load_document(path)
        for index, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            records.append(
                {
                    "document": path.relative_to(root).as_posix(),
                    "chunk_id": index,
                    "title": path.stem,
                    "text": chunk,
                }
            )
    return records
