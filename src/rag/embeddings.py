from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized float32 vectors, leaving zero vectors unchanged."""
    array = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return (array / safe_norms).astype("float32")


class TextEmbedder:
    """SentenceTransformer-backed embedder used by the RAG vector database."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode texts as float32 vectors."""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")
