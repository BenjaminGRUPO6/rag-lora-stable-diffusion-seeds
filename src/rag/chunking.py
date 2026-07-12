from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap")
    clean = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        chunks.append(clean[start : start + chunk_size])
        start += chunk_size - overlap
    return [chunk for chunk in chunks if chunk.strip()]
