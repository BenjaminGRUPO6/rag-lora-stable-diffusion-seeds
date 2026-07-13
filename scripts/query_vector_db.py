from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from src.pipelines.build_rag import load_rag_config
from src.rag.embeddings import TextEmbedder, normalize_vectors
from src.rag.vector_store import FaissStore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for vector DB retrieval."""
    parser = argparse.ArgumentParser(description="Query the persisted FAISS RAG index.")
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--index", type=Path, default=Path("vector_db"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def format_fragment(text: str, width: int = 260) -> str:
    """Return a compact one-line text fragment."""
    fragment = " ".join(text.split())
    return textwrap.shorten(fragment, width=width, placeholder="...")


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_rag_config(args.config)
    rag_config = config.get("rag", {})
    if not isinstance(rag_config, dict):
        raise ValueError("La clave rag debe contener un objeto de configuración")

    top_k = args.top_k if args.top_k is not None else int(rag_config.get("top_k", 5))
    embedding_model = str(rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    normalize = bool(rag_config.get("normalize_embeddings", True))

    store = FaissStore.load(args.index / "index.faiss", args.index / "metadata.json")
    embedder = TextEmbedder(embedding_model)
    query_vector = embedder.encode([args.query], normalize=normalize)
    if normalize:
        query_vector = normalize_vectors(query_vector)

    results = store.search(query_vector, top_k=top_k)
    for rank, result in enumerate(results, start=1):
        source = result.get("source_url") or result.get("local_path", "")
        page = result.get("page") if result.get("page") is not None else "n/a"
        print(f"{rank}. score={result['score']:.4f}")
        print(f"   title: {result.get('title', '')}")
        print(f"   page: {page}")
        print(f"   source: {source}")
        print(f"   fragment: {format_fragment(str(result.get('text', '')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
