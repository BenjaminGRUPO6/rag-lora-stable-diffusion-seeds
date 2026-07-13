from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.pipelines.build_rag import load_rag_config
from src.rag.embeddings import TextEmbedder, normalize_vectors
from src.rag.vector_store import FaissStore

METRIC_K_VALUES = (1, 3, 5)
REQUIRED_QUERY_COLUMNS = {
    "query_id",
    "query",
    "target_topic",
    "expected_document_ids",
    "notes",
    "reviewed_by",
}


@dataclass(frozen=True)
class EvaluationQuery:
    """One retrieval-only evaluation query."""

    query_id: str
    query: str
    target_topic: str
    expected_document_ids: list[str]
    notes: str
    reviewed_by: str

    @property
    def has_complete_relevance(self) -> bool:
        """Return whether the expected ids are marked as an exhaustive relevance set."""
        return "complete_relevance=true" in self.notes.lower()


@dataclass(frozen=True)
class QueryEvaluation:
    """Computed retrieval results and metrics for one query."""

    query: EvaluationQuery
    retrieved_chunk_ids: list[str]
    retrieved_document_ids: list[str]
    retrieved_titles: list[str]
    top_scores: list[float]
    first_relevant_rank: int | None
    hits: dict[int, float]
    reciprocal_rank: float
    precision: dict[int, float | None]
    latency_ms: float


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for RAG retrieval evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate only the RAG document retriever.")
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--index", type=Path, default=Path("vector_db"))
    parser.add_argument("--queries", type=Path, default=Path("data/metadata/rag_evaluation_queries.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/rag/evaluation"))
    return parser.parse_args()


def split_document_ids(value: str) -> list[str]:
    """Parse a semicolon-delimited document-id field."""
    return [item.strip() for item in value.split(";") if item.strip()]


def hit_at_k(ranked_document_ids: list[str], expected_document_ids: list[str], k: int) -> float:
    """Return 1.0 when any expected document appears in the first k retrieved ranks."""
    if k <= 0 or not expected_document_ids:
        return 0.0
    expected = set(expected_document_ids)
    return 1.0 if any(document_id in expected for document_id in ranked_document_ids[:k]) else 0.0


def first_relevant_rank(
    ranked_document_ids: list[str],
    expected_document_ids: list[str],
) -> int | None:
    """Return the one-based rank of the first expected document, if present."""
    expected = set(expected_document_ids)
    if not expected:
        return None
    for rank, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in expected:
            return rank
    return None


def reciprocal_rank(ranked_document_ids: list[str], expected_document_ids: list[str]) -> float:
    """Return reciprocal rank for the first expected document."""
    rank = first_relevant_rank(ranked_document_ids, expected_document_ids)
    return 0.0 if rank is None else 1.0 / rank


def precision_at_k(
    ranked_document_ids: list[str],
    expected_document_ids: list[str],
    k: int,
    *,
    complete_relevance: bool,
) -> float | None:
    """Return Precision@k only when expected ids are marked exhaustive."""
    if not complete_relevance:
        return None
    if k <= 0:
        return 0.0
    expected = set(expected_document_ids)
    retrieved = ranked_document_ids[:k]
    if not retrieved:
        return 0.0
    return sum(1 for document_id in retrieved if document_id in expected) / k


def mean_or_none(values: list[float | None]) -> float | None:
    """Return the arithmetic mean of available values, or None when absent."""
    available = [value for value in values if value is not None]
    if not available:
        return None
    return float(statistics.fmean(available))


def read_queries(path: Path) -> list[EvaluationQuery]:
    """Load evaluation queries from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de consultas: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_QUERY_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Faltan columnas en consultas: {', '.join(sorted(missing))}")

        queries: list[EvaluationQuery] = []
        for row_number, row in enumerate(reader, start=2):
            query_id = str(row.get("query_id", "")).strip()
            query = str(row.get("query", "")).strip()
            target_topic = str(row.get("target_topic", "")).strip()
            expected_document_ids = split_document_ids(str(row.get("expected_document_ids", "")))
            if not query_id or not query or not target_topic:
                raise ValueError(f"Consulta incompleta en fila {row_number}")
            if not expected_document_ids:
                raise ValueError(f"La fila {row_number} no tiene expected_document_ids")
            queries.append(
                EvaluationQuery(
                    query_id=query_id,
                    query=query,
                    target_topic=target_topic,
                    expected_document_ids=expected_document_ids,
                    notes=str(row.get("notes", "")).strip(),
                    reviewed_by=str(row.get("reviewed_by", "")).strip(),
                )
            )
    return queries


def ensure_inputs(config_path: Path, index_dir: Path, queries_path: Path) -> None:
    """Validate required evaluation inputs without mutating them."""
    required_paths = [
        config_path,
        index_dir / "index.faiss",
        index_dir / "metadata.json",
        Path("data/metadata/document_sources.csv"),
        queries_path,
    ]
    missing = [path.as_posix() for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan entradas requeridas: " + ", ".join(missing))


def evaluate_query(
    query: EvaluationQuery,
    *,
    store: FaissStore,
    embedder: TextEmbedder,
    normalize: bool,
    top_k: int,
) -> QueryEvaluation:
    """Run one retrieval query and compute query-level metrics."""
    started = time.perf_counter()
    query_vector = embedder.encode([query.query], normalize=normalize)
    query_vector = np.asarray(query_vector, dtype="float32")
    if normalize:
        query_vector = normalize_vectors(query_vector)
    results = store.search(query_vector, top_k=top_k)
    latency_ms = (time.perf_counter() - started) * 1000.0

    retrieved_chunk_ids = [str(result.get("chunk_id", "")) for result in results]
    retrieved_document_ids = [str(result.get("document_id", "")) for result in results]
    retrieved_titles = [str(result.get("title", "")) for result in results]
    top_scores = [float(result.get("score", 0.0)) for result in results]
    rank = first_relevant_rank(retrieved_document_ids, query.expected_document_ids)
    hits = {
        k: hit_at_k(retrieved_document_ids, query.expected_document_ids, k)
        for k in METRIC_K_VALUES
    }
    precision = {
        k: precision_at_k(
            retrieved_document_ids,
            query.expected_document_ids,
            k,
            complete_relevance=query.has_complete_relevance,
        )
        for k in METRIC_K_VALUES
    }

    return QueryEvaluation(
        query=query,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_document_ids=retrieved_document_ids,
        retrieved_titles=retrieved_titles,
        top_scores=top_scores,
        first_relevant_rank=rank,
        hits=hits,
        reciprocal_rank=0.0 if rank is None else 1.0 / rank,
        precision=precision,
        latency_ms=latency_ms,
    )


def warm_up_retriever(
    *,
    store: FaissStore,
    embedder: TextEmbedder,
    normalize: bool,
) -> None:
    """Run one unmeasured query so latency excludes model warm-up cost."""
    query_vector = embedder.encode(["warmup retrieval query"], normalize=normalize)
    query_vector = np.asarray(query_vector, dtype="float32")
    if normalize:
        query_vector = normalize_vectors(query_vector)
    store.search(query_vector, top_k=1)


def aggregate_metrics(evaluations: list[QueryEvaluation]) -> dict[str, Any]:
    """Aggregate query-level retrieval metrics."""
    query_count = len(evaluations)
    if query_count == 0:
        raise ValueError("No hay consultas para evaluar")

    complete_count = sum(1 for item in evaluations if item.query.has_complete_relevance)
    human_pending = sum(1 for item in evaluations if not item.query.reviewed_by)
    metrics: dict[str, Any] = {
        "query_count": query_count,
        "hit_at_1": statistics.fmean(item.hits[1] for item in evaluations),
        "hit_at_3": statistics.fmean(item.hits[3] for item in evaluations),
        "hit_at_5": statistics.fmean(item.hits[5] for item in evaluations),
        "mrr": statistics.fmean(item.reciprocal_rank for item in evaluations),
        "precision_at_1": mean_or_none([item.precision[1] for item in evaluations]),
        "precision_at_3": mean_or_none([item.precision[3] for item in evaluations]),
        "precision_at_5": mean_or_none([item.precision[5] for item in evaluations]),
        "precision_evaluable_queries": complete_count,
        "mean_retrieval_latency_ms": statistics.fmean(item.latency_ms for item in evaluations),
        "failed_query_ids_at_5": [
            item.query.query_id for item in evaluations if item.hits[5] == 0.0
        ],
        "human_review": {
            "status": "pending" if human_pending else "complete",
            "pending_queries": human_pending,
            "reviewed_queries": query_count - human_pending,
            "metrics": "pending" if human_pending else "available",
        },
    }
    return metrics


def aggregate_topic_metrics(evaluations: list[QueryEvaluation]) -> list[dict[str, Any]]:
    """Aggregate metrics grouped by target topic."""
    topics = sorted({item.query.target_topic for item in evaluations})
    rows: list[dict[str, Any]] = []
    for topic in topics:
        group = [item for item in evaluations if item.query.target_topic == topic]
        rows.append(
            {
                "target_topic": topic,
                "query_count": len(group),
                "hit_at_1": statistics.fmean(item.hits[1] for item in group),
                "hit_at_3": statistics.fmean(item.hits[3] for item in group),
                "hit_at_5": statistics.fmean(item.hits[5] for item in group),
                "mrr": statistics.fmean(item.reciprocal_rank for item in group),
                "precision_at_1": mean_or_none([item.precision[1] for item in group]),
                "precision_at_3": mean_or_none([item.precision[3] for item in group]),
                "precision_at_5": mean_or_none([item.precision[5] for item in group]),
                "mean_retrieval_latency_ms": statistics.fmean(item.latency_ms for item in group),
                "failed_query_ids_at_5": ";".join(
                    item.query.query_id for item in group if item.hits[5] == 0.0
                ),
                "human_review_status": "pending"
                if any(not item.query.reviewed_by for item in group)
                else "complete",
            }
        )
    return rows


def joined(values: list[Any]) -> str:
    """Return a semicolon-delimited string for CSV cells."""
    return ";".join(str(value) for value in values)


def write_query_results(path: Path, evaluations: list[QueryEvaluation]) -> None:
    """Write per-query retrieval results and review placeholders."""
    fieldnames = [
        "query_id",
        "query",
        "target_topic",
        "expected_document_ids",
        "complete_relevance",
        "reviewed_by",
        "retrieved_chunk_ids",
        "retrieved_document_ids",
        "retrieved_titles",
        "top_scores",
        "first_relevant_rank",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "latency_ms",
        "relevance_score",
        "source_support",
        "human_notes",
        "human_metrics_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in evaluations:
            writer.writerow(
                {
                    "query_id": item.query.query_id,
                    "query": item.query.query,
                    "target_topic": item.query.target_topic,
                    "expected_document_ids": joined(item.query.expected_document_ids),
                    "complete_relevance": str(item.query.has_complete_relevance).lower(),
                    "reviewed_by": item.query.reviewed_by,
                    "retrieved_chunk_ids": joined(item.retrieved_chunk_ids),
                    "retrieved_document_ids": joined(item.retrieved_document_ids),
                    "retrieved_titles": joined(item.retrieved_titles),
                    "top_scores": joined([f"{score:.6f}" for score in item.top_scores]),
                    "first_relevant_rank": item.first_relevant_rank or "",
                    "hit_at_1": f"{item.hits[1]:.6f}",
                    "hit_at_3": f"{item.hits[3]:.6f}",
                    "hit_at_5": f"{item.hits[5]:.6f}",
                    "mrr": f"{item.reciprocal_rank:.6f}",
                    "precision_at_1": format_optional_float(item.precision[1]),
                    "precision_at_3": format_optional_float(item.precision[3]),
                    "precision_at_5": format_optional_float(item.precision[5]),
                    "latency_ms": f"{item.latency_ms:.3f}",
                    "relevance_score": "",
                    "source_support": "",
                    "human_notes": "",
                    "human_metrics_status": "pending" if not item.query.reviewed_by else "available",
                }
            )


def format_optional_float(value: float | None) -> str:
    """Format optional float values for CSV output."""
    return "" if value is None else f"{value:.6f}"


def write_topic_metrics(path: Path, topic_rows: list[dict[str, Any]]) -> None:
    """Write topic-level metrics to CSV."""
    fieldnames = [
        "target_topic",
        "query_count",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "mean_retrieval_latency_ms",
        "failed_query_ids_at_5",
        "human_review_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in topic_rows:
            writer.writerow(
                {
                    key: format_optional_float(value)
                    if isinstance(value, float) or value is None
                    else value
                    for key, value in row.items()
                }
            )


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with stable UTF-8 formatting."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    metrics: dict[str, Any],
    topic_rows: list[dict[str, Any]],
    evaluations: list[QueryEvaluation],
    config_path: Path,
    index_dir: Path,
    queries_path: Path,
) -> None:
    """Write a concise Markdown report for the retrieval evaluation."""
    failed = metrics["failed_query_ids_at_5"]
    failed_text = ", ".join(failed) if failed else "None"
    precision_note = (
        "Precision@k was computed only for queries marked complete_relevance=true."
        if metrics["precision_evaluable_queries"]
        else "Precision@k is pending because no query has complete relevance labels."
    )
    lines = [
        "# RAG Retrieval Evaluation",
        "",
        f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "This evaluation measures retrieval only. It does not call or score an LLM.",
        "The FAISS index is loaded from disk and is not rebuilt or modified.",
        "",
        "## Inputs",
        "",
        f"- Config: `{config_path.as_posix()}`",
        f"- Index: `{(index_dir / 'index.faiss').as_posix()}`",
        f"- Metadata: `{(index_dir / 'metadata.json').as_posix()}`",
        f"- Queries: `{queries_path.as_posix()}`",
        "",
        "## Metrics",
        "",
        f"- Queries: {metrics['query_count']}",
        f"- Hit@1: {metrics['hit_at_1']:.6f}",
        f"- Hit@3: {metrics['hit_at_3']:.6f}",
        f"- Hit@5: {metrics['hit_at_5']:.6f}",
        f"- MRR: {metrics['mrr']:.6f}",
        f"- Mean retrieval latency ms: {metrics['mean_retrieval_latency_ms']:.3f}",
        f"- Failed queries at Hit@5: {failed_text}",
        f"- Precision evaluable queries: {metrics['precision_evaluable_queries']}",
        f"- {precision_note}",
        "- Recall@k was not computed because the corpus does not have a complete relevant-document universe per query.",
        "",
        "## Per Topic",
        "",
        "| Topic | Queries | Hit@1 | Hit@3 | Hit@5 | MRR | Mean latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in topic_rows:
        lines.append(
            "| {target_topic} | {query_count} | {hit_at_1:.6f} | {hit_at_3:.6f} | "
            "{hit_at_5:.6f} | {mrr:.6f} | {mean_retrieval_latency_ms:.3f} |".format(**row)
        )

    human_status = metrics["human_review"]["status"]
    lines.extend(
        [
            "",
            "## Human Review",
            "",
            "Per-query output includes `relevance_score` from 1 to 5, `source_support` from 1 to 5, and `human_notes`.",
            f"Human metrics status: {human_status}.",
            f"Pending human review queries: {metrics['human_review']['pending_queries']}",
            "",
            "## Query Failures",
            "",
        ]
    )
    if failed:
        for item in evaluations:
            if item.query.query_id in failed:
                lines.append(
                    f"- {item.query.query_id}: expected {joined(item.query.expected_document_ids)}, "
                    f"retrieved {joined(item.retrieved_document_ids)}"
                )
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    ensure_inputs(args.config, args.index, args.queries)

    config = load_rag_config(args.config)
    rag_config = config.get("rag", {})
    if not isinstance(rag_config, dict):
        raise ValueError("La clave rag debe contener un objeto de configuracion")
    top_k = max(METRIC_K_VALUES[-1], int(rag_config.get("top_k", METRIC_K_VALUES[-1])))
    embedding_model = str(rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    normalize = bool(rag_config.get("normalize_embeddings", True))

    queries = read_queries(args.queries)
    store = FaissStore.load(args.index / "index.faiss", args.index / "metadata.json")
    embedder = TextEmbedder(embedding_model)
    warm_up_retriever(store=store, embedder=embedder, normalize=normalize)
    evaluations = [
        evaluate_query(
            query,
            store=store,
            embedder=embedder,
            normalize=normalize,
            top_k=top_k,
        )
        for query in queries
    ]

    metrics = aggregate_metrics(evaluations)
    metrics.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": args.config.as_posix(),
            "index": args.index.as_posix(),
            "queries": args.queries.as_posix(),
            "top_k": top_k,
            "embedding_model": embedding_model,
        }
    )
    topic_rows = aggregate_topic_metrics(evaluations)

    args.output.mkdir(parents=True, exist_ok=True)
    write_query_results(args.output / "query_results.csv", evaluations)
    write_json(args.output / "metrics.json", metrics)
    write_topic_metrics(args.output / "per_topic_metrics.csv", topic_rows)
    write_report(
        args.output / "evaluation_report.md",
        metrics=metrics,
        topic_rows=topic_rows,
        evaluations=evaluations,
        config_path=args.config,
        index_dir=args.index,
        queries_path=args.queries,
    )

    print(f"Queries: {metrics['query_count']}")
    print(f"Hit@1: {metrics['hit_at_1']:.6f}")
    print(f"Hit@3: {metrics['hit_at_3']:.6f}")
    print(f"Hit@5: {metrics['hit_at_5']:.6f}")
    print(f"MRR: {metrics['mrr']:.6f}")
    print(f"Mean retrieval latency ms: {metrics['mean_retrieval_latency_ms']:.3f}")
    print(f"Failed queries at Hit@5: {', '.join(metrics['failed_query_ids_at_5']) or 'None'}")
    print(f"Human metrics: {metrics['human_review']['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
