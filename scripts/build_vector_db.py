from __future__ import annotations

import argparse
from pathlib import Path

from src.pipelines.build_rag import build_vector_database


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for FAISS vector DB construction."""
    parser = argparse.ArgumentParser(description="Build the FAISS vector database for RAG.")
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--documents", type=Path, default=Path("data/documents/accepted"))
    parser.add_argument("--sources", type=Path, default=Path("data/metadata/document_sources.csv"))
    parser.add_argument("--output", type=Path, default=Path("vector_db"))
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    summary = build_vector_database(
        config_path=args.config,
        documents_dir=args.documents,
        sources_path=args.sources,
        output_dir=args.output,
    )
    print(f"Documents: {summary.documents}")
    print(f"Chunks: {summary.chunks}")
    print(f"Embedding dimension: {summary.embedding_dimension}")
    print(f"Embedding model: {summary.embedding_model}")
    print(f"Index size bytes: {summary.index_size_bytes}")
    print(f"Output: {summary.output_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
