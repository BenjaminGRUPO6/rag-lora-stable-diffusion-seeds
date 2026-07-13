"""Consolidate final project evaluation artifacts without training."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
N_A = "N/A"
DEMO_CLASSES = ("intact", "broken", "immature", "spotted", "skin_damaged")


@dataclass(frozen=True)
class DemoCase:
    """One non-test image selected for system demonstration."""

    case_id: str
    class_name: str
    image_id: str
    image_path: Path
    split: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Consolidate final system metrics.")
    parser.add_argument("--vision-results", type=Path, default=Path("results/vision/resnet18_baseline"))
    parser.add_argument("--rag-results", type=Path, default=Path("results/rag/evaluation"))
    parser.add_argument("--lora-results", type=Path, default=Path("results/lora"))
    parser.add_argument("--output", type=Path, default=Path("results/system"))
    parser.add_argument("--dataset-split", type=Path, default=Path("data/metadata/dataset_split.csv"))
    parser.add_argument("--demo-split", default="validation")
    parser.add_argument("--vision-config", type=Path, default=Path("configs/vision_config.yaml"))
    parser.add_argument("--rag-config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/vision/resnet18_baseline_best.pt"))
    parser.add_argument("--index", type=Path, default=Path("vector_db"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-demo-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def repo_path(path: Path) -> str:
    """Return a repository-relative POSIX path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json_optional(path: Path) -> dict[str, Any]:
    """Read JSON from a path when it exists."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Any) -> None:
    """Write stable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> str:
    """Format scalar values for CSV."""
    if value is None:
        return N_A
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with an explicit column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def infer_class_order(metrics: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    """Infer class order from saved canonical outputs."""
    classes = summary.get("classes")
    if isinstance(classes, list) and classes:
        return [str(item) for item in classes]
    per_class = metrics.get("per_class")
    if isinstance(per_class, dict) and per_class:
        return [str(item) for item in per_class]
    if rows:
        columns = [key.removeprefix("probability_") for key in rows[0] if key.startswith("probability_")]
        if columns:
            return columns
    return list(DEMO_CLASSES)


def build_confusion_matrix(rows: list[dict[str, str]], classes: list[str]) -> dict[str, dict[str, int]] | str:
    """Build confusion counts from saved predictions."""
    if not rows:
        return N_A
    matrix = {true: {pred: 0 for pred in classes} for true in classes}
    for row in rows:
        true = str(row.get("true_label", "")).strip()
        pred = str(row.get("predicted_label", "")).strip()
        if true not in matrix:
            matrix[true] = {label: 0 for label in classes}
        if pred not in matrix[true]:
            matrix[true][pred] = 0
        matrix[true][pred] += 1
    return matrix


def consolidate_vision(vision_results: Path) -> tuple[dict[str, Any], list[str]]:
    """Consolidate canonical ResNet18 metrics."""
    metrics_path = vision_results / "metrics_test.json"
    predictions_path = vision_results / "test_predictions.csv"
    metrics = read_json_optional(metrics_path)
    summary = read_json_optional(vision_results / "run_summary.json")
    rows = read_csv_rows(predictions_path)
    classes = infer_class_order(metrics, summary, rows)
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), dict) else {}
    missing = [f"Vision metric missing: {key}" for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1") if key not in metrics]
    if not rows:
        missing.append("Vision predictions missing; confusion matrix is N/A.")
    for class_name in classes:
        values = per_class.get(class_name) if isinstance(per_class, dict) else None
        if not isinstance(values, dict) or "f1" not in values:
            missing.append(f"Vision class F1 missing: {class_name}")
    return {
        "source": repo_path(metrics_path),
        "prediction_source": repo_path(predictions_path),
        "classification_report_source": repo_path(vision_results / "classification_report.csv"),
        "canonical_evaluation": summary.get("canonical_evaluation", N_A),
        "test_samples": summary.get("test_samples", len(rows) or N_A),
        "accuracy": metrics.get("accuracy", N_A),
        "macro_precision": metrics.get("macro_precision", N_A),
        "macro_recall": metrics.get("macro_recall", N_A),
        "macro_f1": metrics.get("macro_f1", N_A),
        "per_class_f1": {class_name: per_class.get(class_name, {}).get("f1", N_A) if isinstance(per_class.get(class_name, {}), dict) else N_A for class_name in classes},
        "confusion_matrix": build_confusion_matrix(rows, classes),
    }, missing


def consolidate_rag(rag_results: Path) -> tuple[dict[str, Any], list[str]]:
    """Consolidate RAG retrieval metrics."""
    metrics_path = rag_results / "metrics.json"
    query_path = rag_results / "query_results.csv"
    metrics = read_json_optional(metrics_path)
    query_rows = read_csv_rows(query_path)
    review = metrics.get("human_review") if isinstance(metrics.get("human_review"), dict) else {}
    missing = [f"RAG metric missing: {key}" for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_retrieval_latency_ms") if key not in metrics]
    if not review or review.get("metrics") == "pending":
        missing.append("RAG human-review metrics are pending.")
    return {
        "source": repo_path(metrics_path),
        "query_results_source": repo_path(query_path),
        "query_count": metrics.get("query_count", len(query_rows) or N_A),
        "hit_at_1": metrics.get("hit_at_1", N_A),
        "hit_at_3": metrics.get("hit_at_3", N_A),
        "hit_at_5": metrics.get("hit_at_5", N_A),
        "mrr": metrics.get("mrr", N_A),
        "latency_ms": metrics.get("mean_retrieval_latency_ms", N_A),
        "human_review": review or N_A,
        "failed_query_ids_at_5": metrics.get("failed_query_ids_at_5", []),
    }, missing


def consolidate_lora(lora_results: Path) -> tuple[dict[str, Any], list[str]]:
    """Consolidate LoRA evidence without asserting model improvement."""
    manifest_path = lora_results / "run_manifest.json"
    summary_path = lora_results / "training_summary.json"
    inventory_path = lora_results / "evidence_inventory.json"
    comparison_path = lora_results / "base_vs_lora_manifest.csv"
    manifest = read_json_optional(manifest_path)
    summary = read_json_optional(summary_path)
    inventory = read_json_optional(inventory_path)
    comparison_rows = read_csv_rows(comparison_path)
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    adapter = manifest.get("adapter") if isinstance(manifest.get("adapter"), dict) else {}
    parameters = manifest.get("confirmed_parameters") or summary.get("training_parameters") or {}
    missing = list(manifest.get("missing_evidence") or [])
    missing.extend(str(item) for item in inventory.get("missing", []))
    comparison_available = any(str(row.get("evidence_status", "")).upper() != "MISSING" for row in comparison_rows)
    if not comparison_available:
        missing.append("No base-vs-LoRA comparison evidence is available.")
    missing.append("Human visual evaluation of LoRA samples is not available.")
    return {
        "source": repo_path(manifest_path),
        "training_summary_source": repo_path(summary_path),
        "status": manifest.get("status", summary.get("status", N_A)),
        "no_retraining_performed": manifest.get("no_retraining_performed", summary.get("no_retraining_performed", N_A)),
        "confirmed_parameters": parameters or N_A,
        "adapter": adapter or summary.get("adapter_file", N_A),
        "sample_count": len(manifest.get("samples_copied") or []),
        "dataset_metadata_records": dataset.get("metadata_records", summary.get("dataset_images", N_A)),
        "dataset_class_distribution": dataset.get("class_distribution", summary.get("class_distribution", N_A)),
        "base_vs_lora_evidence": "available" if comparison_available else N_A,
        "visual_evaluation_status": N_A,
        "missing_evidence": sorted(set(missing)),
    }, sorted(set(missing))


def select_demo_cases(manifest_path: Path, *, split: str, class_names: tuple[str, ...] = DEMO_CLASSES) -> tuple[list[DemoCase], list[str]]:
    """Select one non-test, non-synthetic demo image per class."""
    if split == "test":
        raise ValueError("Demo cases must not be selected from the test split.")
    rows = read_csv_rows(manifest_path)
    cases: list[DemoCase] = []
    missing: list[str] = []
    for class_name in class_names:
        match = next((row for row in rows if row.get("split") == split and row.get("label") == class_name and str(row.get("is_synthetic", "")).lower() == "false" and row.get("exclusion_status") == "included" and row.get("processed_path")), None)
        if match is None:
            missing.append(f"Demo image missing for class {class_name} in split {split}.")
            continue
        cases.append(DemoCase(f"demo_{len(cases) + 1:02d}", class_name, str(match.get("image_id", "")), Path(str(match["processed_path"])), split))
    return cases, missing


def build_real_demo_analyzer(*, vision_config_path: Path, rag_config_path: Path, checkpoint_path: Path, index_dir: Path, device_name: str | None) -> Callable[[DemoCase], dict[str, Any]]:
    """Build a cached runtime analyzer for demo images."""
    import torch

    from src.pipelines.analyze_seed import analyze_seed, get_nested, load_yaml_config
    from src.rag.retrieval import FaissRetriever
    from src.vision.inference import VisionInferenceEngine

    vision_config = load_yaml_config(vision_config_path)
    rag_config = load_yaml_config(rag_config_path)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    engine = VisionInferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        config=vision_config,
    )
    top_k = int(get_nested(rag_config, ("rag", "top_k"), 5))
    retriever = FaissRetriever.from_paths(
        index_dir=index_dir,
        embedding_model=str(get_nested(rag_config, ("rag", "embedding_model"), "sentence-transformers/all-MiniLM-L6-v2")),
        top_k=top_k,
        normalize_embeddings=bool(get_nested(rag_config, ("rag", "normalize_embeddings"), True)),
    )

    def analyzer(case: DemoCase) -> dict[str, Any]:
        return analyze_seed(
            image=case.image_path,
            vision_config_path=vision_config_path,
            rag_config_path=rag_config_path,
            inference_engine=engine,
            retriever=retriever,
            device_name=str(device),
            top_k=top_k,
        )

    return analyzer


def demo_failure_case(base: dict[str, Any], reason: str) -> dict[str, Any]:
    """Build a failed demo row with N/A metric fields."""
    return {**base, "success": False, "prediction": N_A, "confidence": N_A, "uncertainty_status": N_A, "retrieved_sources_count": N_A, "retrieval_query": N_A, "error": reason, "vision_seconds": N_A, "retrieval_seconds": N_A, "report_seconds": N_A, "total_seconds": N_A}


def run_demo_cases(cases: list[DemoCase], analyzer: Callable[[DemoCase], dict[str, Any]] | None, *, skipped_reason: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run demo cases and capture per-case failures."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        base = {"case_id": case.case_id, "class_name": case.class_name, "image_id": case.image_id, "image_path": repo_path(case.image_path), "split": case.split}
        if analyzer is None:
            reason = skipped_reason or "Demo analyzer is unavailable."
            rows.append(demo_failure_case(base, reason))
            failures.append(failure_row("demo", case.case_id, "system", reason, repo_path(case.image_path)))
            continue
        try:
            result = analyzer(case)
            times = result.get("processing_times") if isinstance(result, dict) else {}
            sources = result.get("retrieved_sources") if isinstance(result, dict) else []
            rows.append({**base, "success": True, "prediction": result.get("prediction", N_A), "confidence": result.get("confidence", N_A), "uncertainty_status": result.get("uncertainty_status", N_A), "retrieved_sources_count": len(sources or []), "retrieval_query": result.get("retrieval_query", N_A), "error": "", "vision_seconds": times.get("vision_seconds", N_A), "retrieval_seconds": times.get("retrieval_seconds", N_A), "report_seconds": times.get("report_seconds", N_A), "total_seconds": times.get("total_seconds", N_A)})
        except Exception as exc:  # noqa: BLE001 - evaluation output captures failures.
            message = str(exc)
            rows.append(demo_failure_case(base, message))
            failures.append(failure_row("demo", case.case_id, "system", message, repo_path(case.image_path)))
    return rows, failures


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Collect numeric values from row dictionaries."""
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key)))
        except (TypeError, ValueError):
            continue
    return values


def mean_or_na(rows: list[dict[str, Any]], key: str) -> float | str:
    """Return a mean for available numeric values or N/A."""
    values = numeric_values(rows, key)
    return statistics.fmean(values) if values else N_A


def source_availability(paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Build source availability rows."""
    return [{"source_name": name, "path": repo_path(path), "available": path.exists()} for name, path in sorted(paths.items())]


def failure_row(failure_type: str, item_id: str, component: str, details: str, source: str) -> dict[str, Any]:
    """Build one failure row."""
    return {"failure_type": failure_type, "item_id": item_id, "component": component, "details": details, "source": source}


def source_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert unavailable sources into failure rows."""
    return [failure_row("source", str(row["source_name"]), "system", "Required source is unavailable.", str(row["path"])) for row in rows if not row["available"]]


def rag_failures(rag: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert RAG Hit@5 failures into failure rows."""
    return [failure_row("rag_hit_at_5", str(query_id), "rag", "Expected document was not retrieved in top 5.", str(rag.get("query_results_source", ""))) for query_id in rag.get("failed_query_ids_at_5") or []]


def metric_row(section: str, metric: str, label: str, value: Any, source: Any) -> dict[str, Any]:
    """Build one flattened metric row."""
    return {"section": section, "metric": metric, "label": label, "value": N_A if value is None else value, "status": "missing" if value in (None, N_A, "") else "available", "source": source or N_A}


def build_final_metric_rows(vision: dict[str, Any], rag: dict[str, Any], lora: dict[str, Any], system: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten consolidated metrics to final_metrics.csv rows."""
    rows: list[dict[str, Any]] = []
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        rows.append(metric_row("vision", metric, "", vision.get(metric), vision.get("source")))
    for class_name, value in (vision.get("per_class_f1") or {}).items():
        rows.append(metric_row("vision", "f1_by_class", class_name, value, vision.get("source")))
    confusion = vision.get("confusion_matrix")
    if isinstance(confusion, dict):
        for true_label, predictions in confusion.items():
            for predicted_label, value in predictions.items():
                rows.append(metric_row("vision", "confusion_matrix_count", f"true={true_label};pred={predicted_label}", value, vision.get("prediction_source")))
    else:
        rows.append(metric_row("vision", "confusion_matrix", "", N_A, vision.get("prediction_source")))
    for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "latency_ms"):
        rows.append(metric_row("rag", metric, "", rag.get(metric), rag.get("source")))
    review_metrics = rag.get("human_review", {}).get("metrics") if isinstance(rag.get("human_review"), dict) else N_A
    rows.append(metric_row("rag", "human_review_metrics", "", review_metrics, rag.get("source")))
    for metric in ("status", "sample_count", "dataset_metadata_records", "base_vs_lora_evidence", "visual_evaluation_status"):
        rows.append(metric_row("lora", metric, "", lora.get(metric), lora.get("source")))
    for metric in ("demo_case_count", "demo_success_count", "mean_visual_inference_seconds", "mean_retrieval_seconds", "mean_total_seconds"):
        rows.append(metric_row("system", metric, "", system.get(metric), system.get("source")))
    for row in system.get("source_availability", []):
        rows.append(metric_row("system", "source_available", row["source_name"], row["available"], row["path"]))
    return rows


def format_report_value(value: Any) -> str:
    """Format values for Markdown."""
    if value is None:
        return N_A
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_report(path: Path, final_metrics: dict[str, Any], generated_files: list[str]) -> None:
    """Write the final Markdown report."""
    vision = final_metrics["vision"]
    rag = final_metrics["rag"]
    lora = final_metrics["lora"]
    system = final_metrics["system"]
    lines = [
        "# Final System Evaluation",
        "",
        f"Generated at UTC: {final_metrics['generated_at_utc']}",
        "",
        "## Scope",
        "",
        "This report consolidates existing results and demo pipeline timings. No training was executed.",
        "The `spotted` label is a visual category and is not treated as a fungal diagnosis.",
        "",
        "## Vision",
        "",
        f"- Accuracy: {format_report_value(vision['accuracy'])}",
        f"- Macro precision: {format_report_value(vision['macro_precision'])}",
        f"- Macro recall: {format_report_value(vision['macro_recall'])}",
        f"- Macro-F1: {format_report_value(vision['macro_f1'])}",
        "",
        "### F1 by Class",
        "",
    ]
    lines.extend(f"- {class_name}: {format_report_value(value)}" for class_name, value in vision.get("per_class_f1", {}).items())
    lines.extend([
        "",
        "## RAG",
        "",
        f"- Hit@1: {format_report_value(rag['hit_at_1'])}",
        f"- Hit@3: {format_report_value(rag['hit_at_3'])}",
        f"- Hit@5: {format_report_value(rag['hit_at_5'])}",
        f"- MRR: {format_report_value(rag['mrr'])}",
        f"- Retrieval latency ms: {format_report_value(rag['latency_ms'])}",
        f"- Human review: {format_report_value(rag.get('human_review'))}",
        "",
        "## LoRA",
        "",
        f"- Status: {format_report_value(lora['status'])}",
        f"- Confirmed parameters: {format_report_value(lora['confirmed_parameters'])}",
        f"- Samples copied: {format_report_value(lora['sample_count'])}",
        f"- Base vs. LoRA evidence: {format_report_value(lora['base_vs_lora_evidence'])}",
        f"- Visual evaluation status: {format_report_value(lora['visual_evaluation_status'])}",
        "",
        "## System Demo",
        "",
        f"- Demo cases: {format_report_value(system['demo_case_count'])}",
        f"- Successful cases: {format_report_value(system['demo_success_count'])}",
        f"- Mean visual inference seconds: {format_report_value(system['mean_visual_inference_seconds'])}",
        f"- Mean retrieval seconds: {format_report_value(system['mean_retrieval_seconds'])}",
        f"- Mean total seconds: {format_report_value(system['mean_total_seconds'])}",
        "",
        "## Missing Data",
        "",
    ])
    lines.extend([f"- {item}" for item in final_metrics["missing_data"]] or ["- None"])
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {item['component']} / {item['item_id']}: {item['details']} ({item['source']})" for item in final_metrics["failures"]] or ["- None"])
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{name}`" for name in generated_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(*, vision_results: Path, rag_results: Path, lora_results: Path, output_dir: Path, dataset_split: Path, demo_split: str, analyzer: Callable[[DemoCase], dict[str, Any]] | None, skipped_demo_reason: str | None = None, source_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """Build and write all final evaluation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vision, missing_vision = consolidate_vision(vision_results)
    rag, missing_rag = consolidate_rag(rag_results)
    lora, missing_lora = consolidate_lora(lora_results)
    demo_cases, missing_demo = select_demo_cases(dataset_split, split=demo_split)
    demo_rows, demo_failures = run_demo_cases(demo_cases, analyzer, skipped_reason=skipped_demo_reason)
    sources = source_availability(source_paths or {"vision_metrics": vision_results / "metrics_test.json", "vision_predictions": vision_results / "test_predictions.csv", "rag_metrics": rag_results / "metrics.json", "rag_query_results": rag_results / "query_results.csv", "lora_run_manifest": lora_results / "run_manifest.json", "lora_training_summary": lora_results / "training_summary.json", "dataset_split": dataset_split})
    failures = source_failures(sources) + rag_failures(rag) + demo_failures
    system = {
        "source": "results/system/demo_cases.csv",
        "demo_split": demo_split,
        "demo_case_count": len(demo_rows),
        "demo_success_count": sum(1 for row in demo_rows if row.get("success") is True),
        "mean_visual_inference_seconds": mean_or_na(demo_rows, "vision_seconds"),
        "mean_retrieval_seconds": mean_or_na(demo_rows, "retrieval_seconds"),
        "mean_total_seconds": mean_or_na(demo_rows, "total_seconds"),
        "source_availability": sources,
    }
    final_metrics = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "vision": vision, "rag": rag, "lora": lora, "system": system, "missing_data": sorted(set(missing_vision + missing_rag + missing_lora + missing_demo)), "failures": failures}
    files = {
        "json": output_dir / "final_metrics.json",
        "metrics_csv": output_dir / "final_metrics.csv",
        "demo_csv": output_dir / "demo_cases.csv",
        "latency_csv": output_dir / "latency_report.csv",
        "failures_csv": output_dir / "failure_cases.csv",
        "report": output_dir / "final_evaluation_report.md",
    }
    write_json(files["json"], final_metrics)
    write_csv(files["metrics_csv"], build_final_metric_rows(vision, rag, lora, system), ["section", "metric", "label", "value", "status", "source"])
    write_csv(files["demo_csv"], demo_rows, ["case_id", "class_name", "image_id", "image_path", "split", "success", "prediction", "confidence", "uncertainty_status", "retrieved_sources_count", "retrieval_query", "error"])
    write_csv(files["latency_csv"], [{"case_id": row["case_id"], "class_name": row["class_name"], "image_path": row["image_path"], "vision_seconds": row["vision_seconds"], "retrieval_seconds": row["retrieval_seconds"], "report_seconds": row["report_seconds"], "total_seconds": row["total_seconds"], "success": row["success"]} for row in demo_rows], ["case_id", "class_name", "image_path", "vision_seconds", "retrieval_seconds", "report_seconds", "total_seconds", "success"])
    write_csv(files["failures_csv"], failures, ["failure_type", "item_id", "component", "details", "source"])
    generated_files = [repo_path(path) for path in files.values()]
    write_report(files["report"], final_metrics, generated_files)
    return final_metrics


def print_summary(final_metrics: dict[str, Any], output_dir: Path) -> None:
    """Print concise CLI output."""
    print("Final system evaluation written to " + repo_path(output_dir))
    print(f"Vision accuracy: {format_report_value(final_metrics['vision']['accuracy'])}")
    print(f"Vision macro-F1: {format_report_value(final_metrics['vision']['macro_f1'])}")
    print(f"RAG Hit@5: {format_report_value(final_metrics['rag']['hit_at_5'])}")
    print(f"RAG MRR: {format_report_value(final_metrics['rag']['mrr'])}")
    print(f"Demo successes: {final_metrics['system']['demo_success_count']}/{final_metrics['system']['demo_case_count']}")
    print(f"Mean total seconds: {format_report_value(final_metrics['system']['mean_total_seconds'])}")
    print(f"Missing data items: {len(final_metrics['missing_data'])}")
    print(f"Failure rows: {len(final_metrics['failures'])}")


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    vision_results = resolve_path(args.vision_results)
    rag_results = resolve_path(args.rag_results)
    lora_results = resolve_path(args.lora_results)
    output_dir = resolve_path(args.output)
    dataset_split = resolve_path(args.dataset_split)
    checkpoint = resolve_path(args.checkpoint)
    index_dir = resolve_path(args.index)
    analyzer: Callable[[DemoCase], dict[str, Any]] | None = None
    skipped_reason = "Demo run was skipped by CLI flag." if args.skip_demo_run else None
    if not args.skip_demo_run:
        try:
            analyzer = build_real_demo_analyzer(vision_config_path=resolve_path(args.vision_config), rag_config_path=resolve_path(args.rag_config), checkpoint_path=checkpoint, index_dir=index_dir, device_name=args.device)
        except Exception as exc:  # noqa: BLE001 - reported as evaluation evidence.
            skipped_reason = f"Demo analyzer setup failed: {exc}"
    source_paths = {"vision_metrics": vision_results / "metrics_test.json", "vision_predictions": vision_results / "test_predictions.csv", "rag_metrics": rag_results / "metrics.json", "rag_query_results": rag_results / "query_results.csv", "lora_run_manifest": lora_results / "run_manifest.json", "lora_training_summary": lora_results / "training_summary.json", "dataset_split": dataset_split, "vision_checkpoint": checkpoint, "rag_index": index_dir / "index.faiss", "rag_metadata": index_dir / "metadata.json"}
    final_metrics = build_outputs(vision_results=vision_results, rag_results=rag_results, lora_results=lora_results, output_dir=output_dir, dataset_split=dataset_split, demo_split=str(args.demo_split), analyzer=analyzer, skipped_demo_reason=skipped_reason, source_paths=source_paths)
    print_summary(final_metrics, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
