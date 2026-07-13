from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Protocol

from src.rag.embeddings import TextEmbedder, normalize_vectors
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


class MetadataKeywordRetriever:
    """Local keyword retriever over persisted chunk metadata.

    This fallback keeps the RAG path usable when the FAISS index exists but the
    embedding model is unavailable locally and network access is disabled.
    """

    def __init__(self, metadata: list[dict], top_k: int = 5) -> None:
        self.metadata = metadata
        self.top_k = top_k

    @classmethod
    def from_path(cls, metadata_path: str | Path, top_k: int = 5) -> MetadataKeywordRetriever:
        """Load chunk metadata from a vector database metadata file."""
        import json

        loaded = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("metadata.json debe contener una lista de chunks")
        return cls([record for record in loaded if isinstance(record, dict)], top_k=top_k)

    def __call__(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return metadata chunks ranked by deterministic lexical overlap."""
        query_terms = tokenize(query)
        if not query_terms:
            return []

        ranked: list[tuple[float, int, dict]] = []
        for index, record in enumerate(self.metadata):
            haystack = " ".join(
                str(record.get(key) or "")
                for key in ("title", "topic", "text", "document_id")
            )
            terms = tokenize(haystack)
            overlap = query_terms.intersection(terms)
            if not overlap:
                continue
            score = len(overlap) / max(len(query_terms), 1)
            ranked.append((score, index, record))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"score": float(score), **record}
            for score, _, record in ranked[: top_k or self.top_k]
        ]


def tokenize(text: str) -> set[str]:
    """Tokenize text for lightweight local retrieval."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return {
        token
        for token in re.findall(r"[a-z0-9_]{3,}", ascii_text)
        if token not in {"para", "with", "this", "that", "from", "sobre", "como", "por"}
    }
