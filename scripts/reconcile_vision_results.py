from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.vision.dataset import (
    EXPECTED_CLASSES,
    OrderedImageFolder,
    build_transforms,
    class_distribution,
)
from src.vision.evaluation import compute_metrics, image_paths, load_checkpoint
from src.vision.model import create_model
from src.vision.train import resolve_device

EXPECTED_TEST_SAMPLES = 522
EXPERIMENT_TITLE = "Resultados 1 — Baseline ResNet18"
DEFAULT_RESULTS_DIR = Path("results/vision/resultados_1_baseline")
DEFAULT_SOURCE_DIR = Path("results/vision/resnet18_baseline")
DEFAULT_CHECKPOINT = Path("models/vision/resnet18_baseline_best.pt")
DEFAULT_MANIFEST = Path("data/metadata/dataset_split.csv")
DEFAULT_CONFIG = Path("configs/vision_config.yaml")
ARCHIVE_DIRNAME = "archive_original"
CANONICAL_FILES: tuple[str, ...] = (
    "r1_metricas.json",
    "r1_reporte_clasificacion.csv",
    "r1_predicciones_test.csv",
    "r1_reconciliation_report.json",
    "r1_reconciliation_report.md",
    "r1_metricas_resumen.png",
    "r1_f1_por_clase.png",
    "r1_precision_recall_por_clase.png",
    "r1_matriz_confusion.png",
    "r1_matriz_confusion_normalizada.png",
    "r1_distribucion_confianza.png",
    "run_summary.json",
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for baseline result reconciliation."""
    parser = argparse.ArgumentParser(
        description="Reconcile Resultados 1 baseline metrics without retraining."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--expected-test-samples",
        type=int,
        default=EXPECTED_TEST_SAMPLES,
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    """Return a stable repository-relative path string."""
    return path.as_posix()


def parse_bool(value: object) -> bool:
    """Parse CSV boolean values robustly."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_path(value: str | Path) -> str:
    """Normalize paths for manifest and ImageFolder comparisons."""
    return Path(value).as_posix().lower()


def ensure_archive_original(results_dir: Path) -> Path:
    """Back up current Resultados 1 files before any overwrite."""
    results_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = results_dir / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(results_dir.iterdir()):
        if source.name == ARCHIVE_DIRNAME or not source.is_file():
            continue
        target = archive_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
    return archive_dir


def validate_class_contract(
    config_classes: Sequence[str],
    checkpoint_class_to_idx: dict[str, int],
) -> dict[str, int]:
    """Validate the expected five soybean visual classes and checkpoint mapping."""
    expected_classes = list(EXPECTED_CLASSES)
    classes = list(config_classes)
    if classes != expected_classes:
        raise ValueError(f"Unexpected config classes. Expected {expected_classes}, got {classes}.")
    expected_mapping = {
        class_name: index for index, class_name in enumerate(expected_classes)
    }
    if checkpoint_class_to_idx != expected_mapping:
        raise ValueError(
            "Checkpoint class_to_idx mismatch. "
            f"Expected {expected_mapping}, got {checkpoint_class_to_idx}."
        )
    return expected_mapping


def validate_manifest_test_split(
    manifest_path: Path,
    expected_classes: Sequence[str],
    expected_test_samples: int,
) -> dict[str, Any]:
    """Validate the registered test rows and prove they contain no synthetic data."""
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    test_rows = [row for row in rows if row.get("split") == "test"]
    if len(test_rows) != expected_test_samples:
        raise ValueError(
            f"Expected {expected_test_samples} test rows in {manifest_path}, "
            f"found {len(test_rows)}."
        )

    synthetic_rows = [row for row in test_rows if parse_bool(row.get("is_synthetic"))]
    if synthetic_rows:
        raise ValueError(f"Test split contains {len(synthetic_rows)} synthetic rows.")

    labels = [str(row.get("label", "")) for row in test_rows]
    support = Counter(labels)
    expected_set = set(expected_classes)
    found_set = set(support)
    if found_set != expected_set:
        raise ValueError(
            f"Unexpected test classes. Expected {sorted(expected_set)}, "
            f"found {sorted(found_set)}."
        )
    ordered_support = {
        class_name: int(support[class_name]) for class_name in expected_classes
    }
    support_sum = sum(ordered_support.values())
    if support_sum != expected_test_samples:
        raise ValueError(
            f"Expected support sum {expected_test_samples}, found {support_sum}."
        )

    return {
        "sample_count": len(test_rows),
        "support": ordered_support,
        "synthetic_count": len(synthetic_rows),
        "processed_paths": sorted(
            normalize_path(row["processed_path"]) for row in test_rows
        ),
    }


def build_test_loader(
    config: dict[str, Any],
    class_names: Sequence[str],
) -> DataLoader:
    """Create the deterministic test dataloader from the physical processed split."""
    data_config = config["data"]
    test_dataset = OrderedImageFolder(
        Path(data_config["root"]) / "test",
        expected_classes=class_names,
        transform=build_transforms(
            image_size=int(data_config["image_size"]),
            train=False,
        ),
    )
    return DataLoader(
        test_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def validate_test_dataset(
    loader: DataLoader,
    manifest_validation: dict[str, Any],
    class_to_idx: dict[str, int],
    expected_test_samples: int,
) -> dict[str, Any]:
    """Validate physical test files, support and deterministic transform."""
    dataset = loader.dataset
    sample_count = len(dataset)  # type: ignore[arg-type]
    if sample_count != expected_test_samples:
        raise ValueError(
            f"Expected {expected_test_samples} physical test images, found {sample_count}."
        )

    sample_paths = sorted(
        normalize_path(Path(sample[0])) for sample in getattr(dataset, "samples", [])
    )
    if sample_paths != manifest_validation["processed_paths"]:
        raise ValueError("Physical test files do not match manifest processed_path rows.")

    support = class_distribution(dataset, class_to_idx)
    if support != manifest_validation["support"]:
        raise ValueError(
            f"Physical support differs from manifest. "
            f"Manifest={manifest_validation['support']}, physical={support}."
        )
    return {
        "deterministic_transforms": True,
        "sample_count": sample_count,
        "support": support,
        "transform": str(getattr(dataset, "transform", "")),
    }


def evaluate_with_inference_mode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], list[int], list[int], list[list[float]], dict[str, bool]]:
    """Evaluate the model using model.eval() and torch.inference_mode()."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    inference_mode_seen = False
    with torch.inference_mode():
        inference_mode_seen = torch.is_inference_mode_enabled()
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            batch_probabilities = torch.softmax(logits, dim=1).detach().cpu()
            predictions = batch_probabilities.argmax(dim=1)
            y_true.extend(int(label) for label in labels.cpu().tolist())
            y_pred.extend(int(prediction) for prediction in predictions.tolist())
            probabilities.extend(
                [float(value) for value in row] for row in batch_probabilities.tolist()
            )
    flags = {
        "model_eval": not model.training,
        "torch_inference_mode": inference_mode_seen,
    }
    return compute_metrics(y_true, y_pred, class_names), y_true, y_pred, probabilities, flags


def save_predictions_csv(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
    dataset: Any,
    output_path: Path,
) -> None:
    """Write one canonical prediction row per test image."""
    paths = image_paths(dataset)
    rows: list[dict[str, Any]] = []
    for index, (true_index, pred_index, row_probabilities) in enumerate(
        zip(y_true, y_pred, probabilities, strict=True)
    ):
        row: dict[str, Any] = {
            "image_path": paths[index] if index < len(paths) else "",
            "true_label": class_names[int(true_index)],
            "predicted_label": class_names[int(pred_index)],
            "predicted_probability": float(row_probabilities[int(pred_index)]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"probability_{class_name}"] = float(row_probabilities[class_index])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_classification_report_csv(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> pd.DataFrame:
    """Write the canonical sklearn classification report."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).transpose()
    frame.to_csv(output_path)
    return frame


def old_result_snapshot(archive_dir: Path) -> dict[str, Any]:
    """Read archived values for old-vs-canonical comparison."""
    snapshot: dict[str, Any] = {}
    for filename in ("metrics_test.json", "run_summary.json", "classification_report.csv"):
        path = archive_dir / filename
        if not path.exists():
            continue
        if path.suffix == ".json":
            snapshot[path.stem] = read_json(path)
        elif path.name == "classification_report.csv":
            report = pd.read_csv(path, index_col=0)
            if "macro avg" in report.index and "f1-score" in report.columns:
                snapshot["classification_report_macro_f1"] = float(
                    report.loc["macro avg", "f1-score"]
                )
    return snapshot


def has_metric_discrepancy(snapshot: dict[str, Any]) -> bool:
    """Return whether a snapshot contains a summary-vs-metrics macro-F1 mismatch."""
    summary_f1 = snapshot.get("run_summary", {}).get("test_macro_f1")
    metrics_f1 = snapshot.get("metrics_test", {}).get("macro_f1")
    if summary_f1 is None or metrics_f1 is None:
        return False
    return abs(float(summary_f1) - float(metrics_f1)) > 1e-12


def collect_historical_results(output_archive_dir: Path, source_dir: Path) -> dict[str, Any]:
    """Collect available historical result snapshots and select discrepancy evidence."""
    snapshots: dict[str, Any] = {
        "resultados_1_archive_original": old_result_snapshot(output_archive_dir)
    }
    source_archive_dir = source_dir / "archive_before_reconciliation"
    if source_archive_dir.exists():
        snapshots["source_archive_before_reconciliation"] = old_result_snapshot(
            source_archive_dir
        )
    if source_dir.exists():
        snapshots["source_current"] = old_result_snapshot(source_dir)

    selected_name = "resultados_1_archive_original"
    for name in (
        "source_archive_before_reconciliation",
        "resultados_1_archive_original",
        "source_current",
    ):
        snapshot = snapshots.get(name, {})
        if has_metric_discrepancy(snapshot):
            selected_name = name
            break
    return {
        "selected_discrepancy_source": selected_name,
        "selected_snapshot": snapshots.get(selected_name, {}),
        "snapshots": snapshots,
    }


def build_discrepancy_explanation(
    old_snapshot: dict[str, Any],
    canonical_metrics: dict[str, Any],
    checkpoint: dict[str, Any],
) -> list[str]:
    """Explain the demonstrated cause of the historical metric discrepancy."""
    explanations: list[str] = []
    old_summary = old_snapshot.get("run_summary", {})
    old_metrics = old_snapshot.get("metrics_test", {})
    old_summary_f1 = old_summary.get("test_macro_f1")
    old_metrics_f1 = old_metrics.get("macro_f1")
    old_report_f1 = old_snapshot.get("classification_report_macro_f1")

    if old_summary_f1 is not None and old_metrics_f1 is not None:
        if abs(float(old_summary_f1) - float(old_metrics_f1)) > 1e-12:
            explanations.append(
                "Archived run_summary.json reported test_macro_f1="
                f"{float(old_summary_f1):.12f}, while archived metrics_test.json "
                f"reported macro_f1={float(old_metrics_f1):.12f}."
            )
    if old_report_f1 is not None and old_metrics_f1 is not None:
        if abs(float(old_report_f1) - float(old_metrics_f1)) <= 1e-12:
            explanations.append(
                "Archived classification_report.csv agrees with metrics_test.json "
                f"at macro-F1={float(old_report_f1):.12f}."
            )
    if old_metrics_f1 is not None:
        if abs(float(old_metrics_f1) - float(canonical_metrics["macro_f1"])) <= 1e-12:
            explanations.append(
                "The new checkpoint evaluation exactly reproduces metrics_test.json "
                "and classification_report.csv, so the higher run_summary value is stale."
            )

    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_best = checkpoint.get("best_validation_macro_f1")
    old_epochs = old_summary.get("epochs_ran")
    old_best = old_summary.get("best_validation_macro_f1")
    if checkpoint_epoch != old_epochs or checkpoint_best != old_best:
        explanations.append(
            "Checkpoint metadata also contradicts the archived summary: "
            f"checkpoint epoch={checkpoint_epoch}, checkpoint best_validation_macro_f1="
            f"{checkpoint_best}, summary epochs_ran={old_epochs}, "
            f"summary best_validation_macro_f1={old_best}."
        )
    if not explanations:
        explanations.append("No historical discrepancy was demonstrated from archived files.")
    return explanations


def build_run_summary(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    manifest_path: Path,
    metrics_path: Path,
    report_path: Path,
    checkpoint: dict[str, Any],
    classes: Sequence[str],
    manifest_validation: dict[str, Any],
    metrics: dict[str, Any],
    explanation: Sequence[str],
) -> dict[str, Any]:
    """Build the corrected run summary from canonical metrics."""
    return {
        "canonical_evaluation": True,
        "checkpoint": repo_path(checkpoint_path),
        "checkpoint_best_validation_macro_f1": checkpoint.get("best_validation_macro_f1"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": checkpoint_sha256,
        "classes": list(classes),
        "correction_reason": list(explanation),
        "metrics_source": repo_path(metrics_path),
        "reconciliation_report": repo_path(report_path),
        "split_manifest": repo_path(manifest_path),
        "synthetic_in_test": int(manifest_validation["synthetic_count"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "test_macro_precision": float(metrics["macro_precision"]),
        "test_macro_recall": float(metrics["macro_recall"]),
        "test_samples": int(manifest_validation["sample_count"]),
        "test_support": manifest_validation["support"],
    }


def save_plots(
    metrics: dict[str, Any],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    """Generate all requested PNG charts with the experiment title."""
    output_paths = [
        output_dir / "r1_metricas_resumen.png",
        output_dir / "r1_f1_por_clase.png",
        output_dir / "r1_precision_recall_por_clase.png",
        output_dir / "r1_matriz_confusion.png",
        output_dir / "r1_matriz_confusion_normalizada.png",
        output_dir / "r1_distribucion_confianza.png",
    ]
    _plot_metric_summary(metrics, output_paths[0])
    _plot_f1_by_class(metrics, class_names, output_paths[1])
    _plot_precision_recall_by_class(metrics, class_names, output_paths[2])
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize="true",
    )
    _plot_confusion_matrix(matrix, class_names, output_paths[3], "Matriz de confusion", "d")
    _plot_confusion_matrix(
        normalized,
        class_names,
        output_paths[4],
        "Matriz de confusion normalizada",
        ".2f",
    )
    confidences = [float(row[int(prediction)]) for row, prediction in zip(probabilities, y_pred)]
    _plot_confidence_distribution(confidences, output_paths[5])
    return output_paths


def _figure(title: str, figsize: tuple[float, float]) -> tuple[Any, Any]:
    figure, axis = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_title(f"{EXPERIMENT_TITLE}\n{title}")
    return figure, axis


def _save_figure(figure: Any, output_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_metric_summary(metrics: dict[str, Any], output_path: Path) -> None:
    figure, axis = _figure("Metricas resumen", (7, 4.5))
    names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    values = [float(metrics[name]) for name in names]
    labels = ["Accuracy", "Macro precision", "Macro recall", "Macro-F1"]
    bars = axis.bar(labels, values, color=["#2563eb", "#059669", "#d97706", "#7c3aed"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Valor")
    axis.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    _save_figure(figure, output_path)


def _plot_f1_by_class(
    metrics: dict[str, Any],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    figure, axis = _figure("F1 por clase", (7, 4.5))
    values = [float(metrics["per_class"][class_name]["f1"]) for class_name in class_names]
    bars = axis.bar(class_names, values, color="#2563eb")
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    axis.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    _save_figure(figure, output_path)


def _plot_precision_recall_by_class(
    metrics: dict[str, Any],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    figure, axis = _figure("Precision y recall por clase", (8, 4.8))
    x_positions = range(len(class_names))
    precision = [
        float(metrics["per_class"][class_name]["precision"]) for class_name in class_names
    ]
    recall = [float(metrics["per_class"][class_name]["recall"]) for class_name in class_names]
    width = 0.38
    axis.bar([x - width / 2 for x in x_positions], precision, width, label="Precision")
    axis.bar([x + width / 2 for x in x_positions], recall, width, label="Recall")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Valor")
    axis.set_xticks(list(x_positions), class_names, rotation=30, ha="right")
    axis.legend()
    _save_figure(figure, output_path)


def _plot_confusion_matrix(
    matrix: Any,
    class_names: Sequence[str],
    output_path: Path,
    title: str,
    value_format: str,
) -> None:
    figure, axis = _figure(title, (7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Clase real",
        xlabel="Clase predicha",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if getattr(matrix, "size", 0) else 0
    for row_index in range(len(class_names)):
        for column_index in range(len(class_names)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    _save_figure(figure, output_path)


def _plot_confidence_distribution(confidences: Sequence[float], output_path: Path) -> None:
    figure, axis = _figure("Distribucion de confianza", (7, 4.5))
    axis.hist(confidences, bins=20, color="#059669", edgecolor="white")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Confianza predicha")
    axis.set_ylabel("Cantidad de imagenes")
    _save_figure(figure, output_path)


def build_report_markdown(report: dict[str, Any]) -> str:
    """Render the canonical reconciliation report in Markdown."""
    metrics = report["canonical_results"]["metrics"]
    lines = [
        f"# {EXPERIMENT_TITLE}",
        "",
        f"Generated at UTC: {report['generated_at_utc']}",
        "",
        "## Evaluacion canonica",
        "",
        f"- Checkpoint: `{report['inputs']['checkpoint']}`",
        f"- Checkpoint SHA-256: `{report['inputs']['checkpoint_sha256']}`",
        f"- Manifest: `{report['inputs']['manifest']}`",
        f"- Test images: {report['validations']['manifest']['sample_count']}",
        f"- Synthetic test images: {report['validations']['manifest']['synthetic_count']}",
        f"- Accuracy: {metrics['accuracy']:.12f}",
        f"- Macro precision: {metrics['macro_precision']:.12f}",
        f"- Macro recall: {metrics['macro_recall']:.12f}",
        f"- Macro-F1: {metrics['macro_f1']:.12f}",
        "",
        "## F1 por clase",
        "",
        "| class | support | f1 |",
        "| --- | ---: | ---: |",
    ]
    for class_name in EXPECTED_CLASSES:
        class_metrics = metrics["per_class"][class_name]
        lines.append(
            f"| {class_name} | {class_metrics['support']} | "
            f"{class_metrics['f1']:.12f} |"
        )
    lines.extend(["", "## Causa de la discrepancia", ""])
    lines.extend(f"- {item}" for item in report["comparison"]["explanation"])
    lines.append("")
    return "\n".join(lines)


def update_manifest(
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    metrics: dict[str, Any],
    test_samples: int,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Update Resultados 1 manifest with reconciled canonical artifacts."""
    previous_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    file_entries = []
    for filename in CANONICAL_FILES:
        path = output_dir / filename
        if not path.exists() or not path.is_file():
            continue
        file_entries.append(
            {
                "file": repo_path(path),
                "sha256": sha256_file(path),
                "status": "RECONCILED",
                "generated_at_utc": generated_at_utc,
            }
        )
    manifest = {
        "archive_dir": repo_path(output_dir / ARCHIVE_DIRNAME),
        "checkpoint": repo_path(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "created_at_utc": previous_manifest.get("created_at_utc", generated_at_utc),
        "destination_dir": repo_path(output_dir),
        "experiment_id": "resultados_1_baseline",
        "files": file_entries,
        "final_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
        },
        "reconciled_at_utc": generated_at_utc,
        "source_dir": repo_path(DEFAULT_SOURCE_DIR),
        "status": "RECONCILED",
        "test_samples": int(test_samples),
    }
    write_json(manifest_path, manifest)
    return manifest


def reconcile(
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    device_name: str | None = None,
    expected_test_samples: int = EXPECTED_TEST_SAMPLES,
) -> dict[str, Any]:
    """Run one canonical checkpoint evaluation and write Resultados 1 artifacts."""
    archive_dir = ensure_archive_original(output_dir)
    config = load_config(config_path)
    device = resolve_device(device_name)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    checkpoint_mapping = {
        str(class_name): int(index)
        for class_name, index in dict(checkpoint.get("class_to_idx", {})).items()
    }
    class_names = list(config.get("classes", EXPECTED_CLASSES))
    class_to_idx = validate_class_contract(class_names, checkpoint_mapping)

    manifest_validation = validate_manifest_test_split(
        manifest_path=manifest_path,
        expected_classes=class_names,
        expected_test_samples=expected_test_samples,
    )
    loader = build_test_loader(config=config, class_names=class_names)
    dataset_validation = validate_test_dataset(
        loader=loader,
        manifest_validation=manifest_validation,
        class_to_idx=class_to_idx,
        expected_test_samples=expected_test_samples,
    )

    model = create_model(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, y_true, y_pred, probabilities, evaluation_flags = evaluate_with_inference_mode(
        model=model,
        loader=loader,
        device=device,
        class_names=class_names,
    )

    metrics_path = output_dir / "r1_metricas.json"
    predictions_path = output_dir / "r1_predicciones_test.csv"
    class_report_path = output_dir / "r1_reporte_clasificacion.csv"
    report_json_path = output_dir / "r1_reconciliation_report.json"
    report_md_path = output_dir / "r1_reconciliation_report.md"
    write_json(metrics_path, metrics)
    report_frame = save_classification_report_csv(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=class_report_path,
    )
    save_predictions_csv(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
        dataset=loader.dataset,
        output_path=predictions_path,
    )
    png_paths = save_plots(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
        output_dir=output_dir,
    )

    support_sum = int(report_frame.loc[list(class_names), "support"].sum())
    if support_sum != expected_test_samples:
        raise RuntimeError(f"Classification report support sum is {support_sum}.")

    historical_results = collect_historical_results(
        output_archive_dir=archive_dir,
        source_dir=source_dir,
    )
    old_snapshot = historical_results["selected_snapshot"]
    explanation = build_discrepancy_explanation(
        old_snapshot=old_snapshot,
        canonical_metrics=metrics,
        checkpoint=checkpoint,
    )
    run_summary = build_run_summary(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        report_path=report_json_path,
        checkpoint=checkpoint,
        classes=class_names,
        manifest_validation=manifest_validation,
        metrics=metrics,
        explanation=explanation,
    )
    write_json(output_dir / "run_summary.json", run_summary)

    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report = {
        "canonical_results": {
            "classification_report": repo_path(class_report_path),
            "metrics": metrics,
            "predictions": repo_path(predictions_path),
            "run_summary": run_summary,
        },
        "comparison": {"explanation": explanation},
        "generated_at_utc": generated_at_utc,
        "inputs": {
            "checkpoint": repo_path(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "config": repo_path(config_path),
            "manifest": repo_path(manifest_path),
            "output_dir": repo_path(output_dir),
            "source_dir": repo_path(source_dir),
        },
        "old_results": historical_results,
        "outputs": {
            "classification_report": repo_path(class_report_path),
            "metrics": repo_path(metrics_path),
            "png": [repo_path(path) for path in png_paths],
            "predictions": repo_path(predictions_path),
            "reconciliation_report_json": repo_path(report_json_path),
            "reconciliation_report_md": repo_path(report_md_path),
            "manifest": repo_path(output_dir / "manifest.json"),
            "run_summary": repo_path(output_dir / "run_summary.json"),
        },
        "validations": {
            "checkpoint": {
                "architecture": checkpoint.get("architecture"),
                "best_validation_macro_f1": checkpoint.get("best_validation_macro_f1"),
                "epoch": checkpoint.get("epoch"),
                "sha256": checkpoint_sha256,
            },
            "checkpoint_class_to_idx": checkpoint_mapping,
            "classes": list(class_names),
            "dataset": dataset_validation,
            "evaluation": evaluation_flags,
            "manifest": {
                "sample_count": manifest_validation["sample_count"],
                "support": manifest_validation["support"],
                "synthetic_count": manifest_validation["synthetic_count"],
            },
            "support_sum": support_sum,
        },
    }
    write_json(report_json_path, report)
    report_md_path.write_text(build_report_markdown(report), encoding="utf-8")
    update_manifest(
        manifest_path=output_dir / "manifest.json",
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        metrics=metrics,
        test_samples=manifest_validation["sample_count"],
        generated_at_utc=generated_at_utc,
    )
    return report


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    report = reconcile(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        device_name=args.device,
        expected_test_samples=int(args.expected_test_samples),
    )
    metrics = report["canonical_results"]["metrics"]
    print(
        yaml.safe_dump(
            {
                "test_samples": report["validations"]["manifest"]["sample_count"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": {
                    class_name: values["f1"]
                    for class_name, values in metrics["per_class"].items()
                },
                "report": report["outputs"]["reconciliation_report_json"],
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
