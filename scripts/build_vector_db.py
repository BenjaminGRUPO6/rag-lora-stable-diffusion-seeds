from __future__ import annotations

import json
from pathlib import Path

from src.pipelines.build_rag import collect_chunks
from src.rag.embeddings import TextEmbedder
from src.rag.vector_store import FaissStore
from src.utils.config import load_yaml


def main() -> None:
    config = load_yaml("configs/rag_config.yaml")
    chunks = collect_chunks(
        config["paths"]["documents_dir"],
        config["retrieval"]["chunk_size"],
        config["retrieval"]["chunk_overlap"],
    )
    if not chunks:
        raise RuntimeError("No se encontraron documentos compatibles en data/documents.")
    embedder = TextEmbedder(config["embedding"]["model_name"])
    vectors = embedder.encode([item["text"] for item in chunks], config["embedding"]["normalize"])
    store = FaissStore(vectors.shape[1])
    store.add(vectors, chunks)
    index_dir = Path(config["paths"]["index_dir"])
    store.save(index_dir / "documents.faiss", config["paths"]["metadata_file"])
    print(json.dumps({"chunks": len(chunks), "dimension": int(vectors.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
