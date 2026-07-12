from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("Cada vector debe tener un registro de metadatos")
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        scores, indices = self.index.search(query_vector, top_k)
        results: list[dict] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append({"score": float(score), **self.metadata[index]})
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        index_output = Path(index_path)
        metadata_output = Path(metadata_path)
        index_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_output))
        metadata_output.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
