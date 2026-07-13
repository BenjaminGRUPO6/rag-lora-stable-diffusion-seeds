from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.pipelines.build_rag import build_vector_database, compute_sha256
from src.rag.vector_store import FaissStore


class FakeEmbedder:
    """Deterministic test embedder that does not load external models."""

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "mechanical" in lowered or "damage" in lowered else 0.1,
                    1.0 if "storage" in lowered or "moisture" in lowered else 0.1,
                    float(len(text) % 17) / 17.0,
                    1.0,
                ]
            )
        return np.asarray(vectors, dtype="float32")


def write_sources_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write document source metadata for tests."""
    fieldnames = [
        "document_id",
        "title",
        "authors",
        "year",
        "organization",
        "source_url",
        "license",
        "local_path",
        "sha256",
        "file_type",
        "pages",
        "language",
        "topics",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_vector_database_with_fake_embeddings(tmp_path: Path) -> None:
    documents = tmp_path / "data" / "documents" / "accepted"
    documents.mkdir(parents=True)
    doc_path = documents / "soybean_damage.md"
    doc_path.write_text(
        "Mechanical damage in soybean seeds can happen during harvest, handling, "
        "threshing and transport. Storage moisture and drying conditions affect quality.",
        encoding="utf-8",
    )
    sources = tmp_path / "data" / "metadata" / "document_sources.csv"
    write_sources_csv(
        sources,
        [
            {
                "document_id": "DOC001",
                "title": "Soybean Damage Notes",
                "authors": "",
                "year": "",
                "organization": "",
                "source_url": "https://example.test/soybean-damage",
                "license": "",
                "local_path": str(doc_path),
                "sha256": compute_sha256(doc_path),
                "file_type": "md",
                "pages": "",
                "language": "en",
                "topics": "broken/mechanical damage; storage; handling",
                "status": "accepted",
                "notes": "",
            }
        ],
    )
    config = tmp_path / "configs" / "rag.yaml"
    config.parent.mkdir()
    config.write_text(
        "\n".join(
            [
                "rag:",
                "  chunk_size: 80",
                "  chunk_overlap: 10",
                "  top_k: 3",
                "  embedding_model: fake/test-embedder",
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "vector_db"
    summary = build_vector_database(
        config_path=config,
        documents_dir=documents,
        sources_path=sources,
        output_dir=output,
        embedder=FakeEmbedder(),
    )

    assert summary.documents == 1
    assert summary.chunks > 0
    assert summary.embedding_dimension == 4
    assert summary.validation["ok"] is True
    assert (output / "index.faiss").exists()
    assert (output / "metadata.json").exists()
    assert (output / "config.json").exists()
    assert (output / "build_manifest.json").exists()

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata) == summary.chunks
    required_keys = {
        "chunk_id",
        "document_id",
        "title",
        "source_url",
        "local_path",
        "page",
        "topic",
        "text",
        "sha256",
    }
    assert all(required_keys <= set(chunk) for chunk in metadata)
    assert all(chunk["text"].strip() for chunk in metadata)
    assert len({chunk["chunk_id"] for chunk in metadata}) == len(metadata)

    store = FaissStore.load(output / "index.faiss", output / "metadata.json")
    query = FakeEmbedder().encode(["mechanical damage during handling"], normalize=True)
    results = store.search(query, top_k=2)
    assert results
    assert results[0]["document_id"] == "DOC001"
