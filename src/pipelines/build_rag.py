from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from src.rag.chunking import chunk_text
from src.rag.document_loader import SUPPORTED_DOCUMENTS, load_document, load_document_pages
from src.rag.embeddings import TextEmbedder, normalize_vectors
from src.rag.vector_store import FaissStore


class Embedder(Protocol):
    """Protocol for production and test embedders."""

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode texts as a two-dimensional float32 array."""


@dataclass(frozen=True)
class DocumentSource:
    """Metadata loaded from document_sources.csv for one accepted document."""

    document_id: str
    title: str
    source_url: str
    local_path: str
    sha256: str
    topic: str
    status: str


@dataclass(frozen=True)
class BuildSummary:
    """Summary of a vector database build."""

    documents: int
    chunks: int
    embedding_dimension: int
    embedding_model: str
    index_size_bytes: int
    output_dir: Path
    validation: dict[str, object]


def load_rag_config(config_path: str | Path) -> dict[str, object]:
    """Load a RAG YAML config file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("La configuración RAG debe ser un objeto YAML")
    return config


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_relative(path: Path, base_dir: Path) -> str:
    """Return a POSIX-style relative path when possible."""
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_metadata_path(local_path: str, base_dir: Path) -> Path:
    """Resolve a metadata local_path field relative to the repository root."""
    path = Path(local_path)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_document_sources(
    sources_path: str | Path,
    documents_dir: str | Path,
    base_dir: str | Path | None = None,
) -> dict[Path, DocumentSource]:
    """Load accepted document source metadata keyed by resolved local path."""
    base = Path(base_dir or Path.cwd()).resolve()
    documents_root = Path(documents_dir).resolve()
    path = Path(sources_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de fuentes: {path}")

    sources: dict[Path, DocumentSource] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = row.get("status", "").strip()
            if status and status != "accepted":
                continue
            local_path = row.get("local_path", "").strip()
            if not local_path:
                continue
            resolved = resolve_metadata_path(local_path, base)
            try:
                resolved.relative_to(documents_root)
            except ValueError:
                continue
            if not resolved.exists():
                raise FileNotFoundError(f"La fuente registrada no existe: {local_path}")
            source = DocumentSource(
                document_id=row.get("document_id", "").strip(),
                title=row.get("title", "").strip(),
                source_url=row.get("source_url", "").strip(),
                local_path=local_path,
                sha256=row.get("sha256", "").strip(),
                topic=row.get("topics", row.get("topic", "")).strip(),
                status=status or "accepted",
            )
            sources[resolved] = source
    return sources


def collect_chunks(
    documents_dir: str | Path,
    chunk_size: int = 700,
    overlap: int = 100,
    sources_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> list[dict]:
    """Collect chunk metadata records from accepted documents.

    When sources_path is omitted, this preserves the legacy lightweight behavior used by
    earlier code. When sources_path is provided, every chunk includes source metadata.
    """
    root = Path(documents_dir)
    if sources_path is None:
        return collect_legacy_chunks(root, chunk_size, overlap)

    sources = load_document_sources(sources_path, root, base_dir)
    return build_chunk_records(root, sources, chunk_size, overlap, Path(base_dir or Path.cwd()))


def collect_legacy_chunks(root: Path, chunk_size: int, overlap: int) -> list[dict]:
    """Collect chunks with the original minimal metadata schema."""
    records: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_DOCUMENTS:
            continue
        text = load_document(path)
        for index, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            records.append(
                {
                    "document": path.relative_to(root).as_posix(),
                    "chunk_id": index,
                    "title": path.stem,
                    "text": chunk,
                }
            )
    return records


def build_chunk_records(
    documents_dir: Path,
    sources: dict[Path, DocumentSource],
    chunk_size: int,
    overlap: int,
    base_dir: Path,
) -> list[dict]:
    """Build chunk records with document source metadata."""
    records: list[dict] = []
    documents = sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENTS
    )
    for path in documents:
        resolved = path.resolve()
        source = sources.get(resolved)
        if source is None:
            raise ValueError(f"Documento sin fuente registrada: {path.as_posix()}")
        checksum = compute_sha256(path)
        if source.sha256 and checksum != source.sha256:
            raise ValueError(f"SHA-256 no coincide para {path.as_posix()}")

        title = source.title or path.stem
        chunk_index = 0
        for page in load_document_pages(path):
            for chunk in chunk_text(page.text, chunk_size, overlap):
                records.append(
                    {
                        "chunk_id": build_chunk_id(source.document_id, page.page, chunk_index),
                        "document_id": source.document_id,
                        "title": title,
                        "source_url": source.source_url,
                        "local_path": source.local_path or normalize_relative(path, base_dir),
                        "page": page.page,
                        "topic": source.topic,
                        "text": chunk,
                        "sha256": checksum,
                    }
                )
                chunk_index += 1
    return records


def build_chunk_id(document_id: str, page: int | None, chunk_index: int) -> str:
    """Build a stable chunk identifier."""
    page_value = 0 if page is None else page
    return f"{document_id}-p{page_value:03d}-c{chunk_index:04d}"


def validate_chunks(chunks: list[dict], source_paths: set[str]) -> list[str]:
    """Validate chunk metadata before indexing."""
    errors: list[str] = []
    if not chunks:
        errors.append("number of chunks must be greater than zero")

    ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    if len(ids) != len(set(ids)):
        errors.append("chunk IDs must be unique")

    for chunk in chunks:
        if not str(chunk.get("text", "")).strip():
            errors.append(f"empty chunk: {chunk.get('chunk_id', '<missing>')}")
        if chunk.get("local_path") not in source_paths:
            errors.append(f"source not registered: {chunk.get('local_path', '<missing>')}")

    return errors


def validate_store(
    *,
    store: FaissStore,
    vectors: np.ndarray,
    metadata: list[dict],
    output_dir: Path,
) -> dict[str, object]:
    """Validate the built and persisted vector store."""
    errors: list[str] = []
    if store.size != len(metadata):
        errors.append("metadata and index are not aligned")
    if store.size <= 0:
        errors.append("number of chunks must be greater than zero")
    if any(not str(record.get("text", "")).strip() for record in metadata):
        errors.append("metadata contains empty chunks")

    reloaded = FaissStore.load(output_dir / "index.faiss", output_dir / "metadata.json")
    if reloaded.size != store.size:
        errors.append("persisted metadata and index are not aligned")

    reproducible = False
    if len(vectors) > 0:
        top_k = min(5, len(vectors))
        first = store.search(vectors[0:1], top_k=top_k)
        second = reloaded.search(vectors[0:1], top_k=top_k)
        reproducible = [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
        if not reproducible:
            errors.append("retrieval is not reproducible after reload")

    return {
        "ok": not errors,
        "errors": errors,
        "metadata_aligned": store.size == len(metadata) == reloaded.size,
        "reproducible_retrieval": reproducible,
    }


def build_vector_database(
    *,
    config_path: str | Path,
    documents_dir: str | Path,
    sources_path: str | Path,
    output_dir: str | Path,
    embedder: Embedder | None = None,
) -> BuildSummary:
    """Build, validate, and persist a FAISS vector database for the RAG corpus."""
    config = load_rag_config(config_path)
    rag_config = config.get("rag", {})
    if not isinstance(rag_config, dict):
        raise ValueError("La clave rag debe contener un objeto de configuración")

    chunk_size = int(rag_config.get("chunk_size", 700))
    overlap = int(rag_config.get("chunk_overlap", 100))
    embedding_model = str(rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    normalize = bool(rag_config.get("normalize_embeddings", True))

    documents_root = Path(documents_dir)
    output = Path(output_dir)
    sources = load_document_sources(sources_path, documents_root)
    chunks = build_chunk_records(
        documents_root,
        sources,
        chunk_size,
        overlap,
        Path.cwd(),
    )

    chunk_errors = validate_chunks(chunks, {source.local_path for source in sources.values()})
    if chunk_errors:
        raise ValueError("; ".join(chunk_errors))

    texts = [str(chunk["text"]) for chunk in chunks]
    active_embedder = embedder or TextEmbedder(embedding_model)
    vectors = active_embedder.encode(texts, normalize=normalize)
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2 or len(vectors) != len(chunks):
        raise ValueError("El embedder debe devolver un vector por chunk")
    if normalize:
        vectors = normalize_vectors(vectors)

    store = FaissStore(int(vectors.shape[1]))
    store.add(vectors, chunks)

    output.mkdir(parents=True, exist_ok=True)
    store.save(output / "index.faiss", output / "metadata.json")
    write_json(output / "config.json", normalize_config(config, config_path, documents_dir, sources_path))

    validation = validate_store(store=store, vectors=vectors, metadata=chunks, output_dir=output)
    if not validation["ok"]:
        raise ValueError("; ".join(str(error) for error in validation["errors"]))

    index_size = (output / "index.faiss").stat().st_size
    manifest = build_manifest(
        documents=len(sources),
        chunks=len(chunks),
        dimension=int(vectors.shape[1]),
        embedding_model=embedding_model,
        index_size=index_size,
        validation=validation,
        config_path=config_path,
        documents_dir=documents_dir,
        sources_path=sources_path,
    )
    write_json(output / "build_manifest.json", manifest)

    return BuildSummary(
        documents=len(sources),
        chunks=len(chunks),
        embedding_dimension=int(vectors.shape[1]),
        embedding_model=embedding_model,
        index_size_bytes=index_size,
        output_dir=output,
        validation=validation,
    )


def normalize_config(
    config: dict[str, object],
    config_path: str | Path,
    documents_dir: str | Path,
    sources_path: str | Path,
) -> dict[str, object]:
    """Return the persisted build configuration."""
    normalized = dict(config)
    normalized["build_inputs"] = {
        "config": Path(config_path).as_posix(),
        "documents": Path(documents_dir).as_posix(),
        "sources": Path(sources_path).as_posix(),
    }
    return normalized


def build_manifest(
    *,
    documents: int,
    chunks: int,
    dimension: int,
    embedding_model: str,
    index_size: int,
    validation: dict[str, object],
    config_path: str | Path,
    documents_dir: str | Path,
    sources_path: str | Path,
) -> dict[str, object]:
    """Build a JSON-serializable manifest for a completed vector DB build."""
    return {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "chunks": chunks,
        "embedding_dimension": dimension,
        "embedding_model": embedding_model,
        "index_size_bytes": index_size,
        "files": {
            "index": "index.faiss",
            "metadata": "metadata.json",
            "config": "config.json",
        },
        "inputs": {
            "config": Path(config_path).as_posix(),
            "documents": Path(documents_dir).as_posix(),
            "sources": Path(sources_path).as_posix(),
        },
        "validation": validation,
    }


def write_json(path: Path, data: object) -> None:
    """Write pretty JSON with UTF-8 encoding."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
