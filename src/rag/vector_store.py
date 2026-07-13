from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    """Thin FAISS IndexFlatIP wrapper with aligned chunk metadata."""

    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []

    @property
    def dimension(self) -> int:
        """Return the embedding dimension expected by the index."""
        return int(self.index.d)

    @property
    def size(self) -> int:
        """Return the number of vectors stored in the index."""
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """Add vectors and metadata records, preserving one-to-one alignment."""
        if len(vectors) != len(metadata):
            raise ValueError("Cada vector debe tener un registro de metadatos")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Las dimensiones de embeddings no coinciden con el índice FAISS")
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search the index and return scored metadata records."""
        scores, indices = self.index.search(query_vector, top_k)
        results: list[dict] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append({"score": float(score), **self.metadata[index]})
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Persist the FAISS index and aligned metadata JSON."""
        index_output = Path(index_path)
        metadata_output = Path(metadata_path)
        index_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_output))
        metadata_output.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> "FaissStore":
        """Load a persisted FAISS index and aligned metadata JSON."""
        index = faiss.read_index(str(Path(index_path)))
        store = cls(int(index.d))
        store.index = index
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("metadata.json debe contener una lista de chunks")
        if len(metadata) != store.size:
            raise ValueError("metadata.json no está alineado con index.faiss")
        store.metadata = metadata
        return store
