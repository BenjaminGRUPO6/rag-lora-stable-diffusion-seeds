from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.rag.embeddings import TextEmbedder, normalize_vectors
from src.rag.prompt_builder import build_retrieval_query
from src.rag.vector_store import FaissStore


class Retriever(Protocol):
    """Callable retrieval interface used by the analysis pipeline."""

    def __call__(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return retrieved source fragments for a query."""


class FaissRetriever:
    """Retrieve source fragments from a persisted FAISS vector database."""

    def __init__(
        self,
        store: FaissStore,
        embedder: TextEmbedder,
        top_k: int = 5,
        normalize_embeddings: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.normalize_embeddings = normalize_embeddings

    @classmethod
    def from_paths(
        cls,
        index_dir: str | Path,
        embedding_model: str,
        top_k: int = 5,
        normalize_embeddings: bool = True,
    ) -> FaissRetriever:
        """Load FAISS index and metadata from an index directory."""
        root = Path(index_dir)
        store = FaissStore.load(root / "index.faiss", root / "metadata.json")
        embedder = TextEmbedder(embedding_model)
        return cls(
            store=store,
            embedder=embedder,
            top_k=top_k,
            normalize_embeddings=normalize_embeddings,
        )

    def __call__(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search the vector store for the query text."""
        query_vector = self.embedder.encode([query], normalize=self.normalize_embeddings)
        if self.normalize_embeddings:
            query_vector = normalize_vectors(query_vector)
        return self.store.search(query_vector, top_k=top_k or self.top_k)
