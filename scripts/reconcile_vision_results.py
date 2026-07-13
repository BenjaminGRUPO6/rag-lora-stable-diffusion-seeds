from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.vision.dataset import (
    EXPECTED_CLASSES,
    OrderedImageFolder,
    build_transforms,
    class_distribution,
)
from src.vision.evaluation import evaluate_model, load_checkpoint, save_evaluation_outputs
from src.vision.model import create_model
from src.vision.train import resolve_device

EXPECTED_TEST_SAMPLES = 522
RESULT_FILENAMES: tuple[str, ...] = (
    "classification_report.csv",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "metrics_test.json",
    "metrics_validation.json",
    "run_config.yaml",
    "run_summary.json",
    "test_predictions.csv",
    "training_curves.png",
    "training_history.csv",
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for canonical test reconciliation."""
    parser = argparse.ArgumentParser(
        description="Reconcile ResNet18 baseline metrics from an existing checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vision_config.yaml"),
        help="Path to the vision configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/dataset_split.csv"),
        help="Dataset split manifest used to validate the original test split.",
    )
    parser.add_argument("--device", type=str, default=None, help="Evaluation device.")
    parser.add_argument(
        "--expected-test-samples",
        type=int,
        default=EXPECTED_TEST_SAMPLES,
        help="Expected number of original test images.",
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


def validate_class_contract(
    config_classes: Sequence[str],
    checkpoint_class_to_idx: dict[str, int],
) -> dict[str, int]:
    """Validate the five soybean visual classes and checkpoint mapping."""
    expected = list(EXPECTED_CLASSES)
    classes = list(config_classes)
    if classes != expected:
        raise ValueError(f"Unexpected config classes. Expected {expected}, got {classes}.")
    expected_mapping = {class_name: index for index, class_name in enumerate(expected)}
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
    """Validate test rows in the original split manifest."""
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
    support_ordered = {class_name: int(support[class_name]) for class_name in expected_classes}
    support_sum = sum(support_ordered.values())
    if support_sum != expected_test_samples:
        raise ValueError(
            f"Expected support sum {expected_test_samples}, found {support_sum}."
        )

    return {
        "rows": test_rows,
        "sample_count": len(test_rows),
        "support": support_ordered,
        "synthetic_count": len(synthetic_rows),
        "processed_paths": sorted(
            normalize_path(row["processed_path"]) for row in test_rows
        ),
    }


def parse_bool(value: object) -> bool:
    """Parse CSV boolean values robustly."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_path(value: str | Path) -> str:
    """Normalize paths for manifest and ImageFolder comparisons."""
    return Path(value).as_posix().lower()


def build_test_loader(
    config: dict[str, Any],
    class_names: Sequence[str],
) -> DataLoader:
    """Create a deterministic dataloader for the configured physical test split."""
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
    """Validate physical test dataset against the original manifest."""
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
        raise ValueError("Physical test files do not match manifest test processed paths.")

    support = class_distribution(dataset, class_to_idx)
    if support != manifest_validation["support"]:
        raise ValueError(
            f"Physical test support differs from manifest. "
            f"Manifest={manifest_validation['support']}, physical={support}."
        )
    return {
        "sample_count": sample_count,
        "support": support,
        "transform": str(getattr(dataset, "transform", "")),
    }


def ensure_archive(results_dir: Path) -> Path:
    """Create an archive of existing result files if one is not already present."""
    archive_dir = results_dir / "archive_before_reconciliation"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for filename in RESULT_FILENAMES:
        source = results_dir / filename
        target = archive_dir / filename
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    return archive_dir


def old_result_snapshot(archive_dir: Path) -> dict[str, Any]:
    """Read archived result values for old-vs-new comparison."""
    snapshot: dict[str, Any] = {}
    metrics_path = archive_dir / "metrics_test.json"
    summary_path = archive_dir / "run_summary.json"
    report_path = archive_dir / "classification_report.csv"
    if metrics_path.exists():
        snapshot["metrics_test"] = read_json(metrics_path)
    if summary_path.exists():
        snapshot["run_summary"] = read_json(summary_path)
    if report_path.exists():
        report = pd.read_csv(report_path, index_col=0)
        if "macro avg" in report.index and "f1-score" in report.columns:
            snapshot["classification_report_macro_f1"] = float(
                report.loc["macro avg", "f1-score"]
            )
    return snapshot


def build_discrepancy_explanation(
    old_snapshot: dict[str, Any],
    canonical_metrics: dict[str, Any],
    checkpoint: dict[str, Any],
) -> list[str]:
    """Explain differences between archived and canonical result files."""
    explanations: list[str] = []
    old_summary = old_snapshot.get("run_summary", {})
    old_metrics = old_snapshot.get("metrics_test", {})
    old_summary_f1 = old_summary.get("test_macro_f1")
    old_metrics_f1 = old_metrics.get("macro_f1")

    if old_summary_f1 is not None and old_metrics_f1 is not None:
        delta = abs(float(old_summary_f1) - float(old_metrics_f1))
        if delta > 1e-12:
            explanations.append(
                "Archived run_summary.json reported test_macro_f1="
                f"{float(old_summary_f1):.12f}, but archived metrics_test.json "
                f"reported macro_f1={float(old_metrics_f1):.12f}."
            )
    if old_metrics_f1 is not None:
        canonical_delta = abs(float(old_metrics_f1) - float(canonical_metrics["macro_f1"]))
        if canonical_delta <= 1e-12:
            explanations.append(
                "Canonical evaluation matches the archived metrics_test.json and "
                "classification_report.csv, so the higher run_summary value was stale."
            )
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_best = checkpoint.get("best_validation_macro_f1")
    old_epochs = old_summary.get("epochs_ran")
    old_best = old_summary.get("best_validation_macro_f1")
    if checkpoint_epoch != old_epochs or checkpoint_best != old_best:
        explanations.append(
            "Checkpoint metadata is incompatible with archived run_summary.json: "
            f"checkpoint epoch={checkpoint_epoch}, checkpoint best_validation_macro_f1="
            f"{checkpoint_best}, summary epochs_ran={old_epochs}, summary "
            f"best_validation_macro_f1={old_best}."
        )
    if not explanations:
        explanations.append("No metric discrepancy was detected in archived files.")
    return explanations


def build_run_summary(
    checkpoint_path: Path,
    manifest_path: Path,
    metrics_path: Path,
    checkpoint: dict[str, Any],
    classes: Sequence[str],
    test_validation: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    """Build run summary from the saved canonical metrics file."""
    metrics = read_json(metrics_path)
    return {
        "canonical_evaluation": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_validation_macro_f1": checkpoint.get(
            "best_validation_macro_f1"
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "classes": list(classes),
        "metrics_source": str(metrics_path),
        "reconciliation_report": str(report_path),
        "split_manifest": str(manifest_path),
        "synthetic_in_test": int(test_validation["synthetic_count"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "test_samples": int(test_validation["sample_count"]),
        "test_support": test_validation["support"],
    }


def build_report_markdown(report: dict[str, Any]) -> str:
    """Render a concise markdown reconciliation report."""
    metrics = report["canonical_results"]["metrics_test"]
    per_class = metrics["per_class"]
    lines = [
        "# ResNet18 baseline reconciliation",
        "",
        f"Generated at UTC: {report['generated_at_utc']}",
        "",
        "## Canonical evaluation",
        "",
        f"- Checkpoint: `{report['inputs']['checkpoint']}`",
        f"- Manifest: `{report['inputs']['manifest']}`",
        f"- Test images: {report['validations']['manifest']['sample_count']}",
        f"- Accuracy: {metrics['accuracy']:.12f}",
        f"- Macro-F1: {metrics['macro_f1']:.12f}",
        "",
        "## F1 by class",
        "",
        "| class | support | f1 |",
        "| --- | ---: | ---: |",
    ]
    for class_name in EXPECTED_CLASSES:
        class_metrics = per_class[class_name]
        lines.append(
            f"| {class_name} | {class_metrics['support']} | "
            f"{class_metrics['f1']:.12f} |"
        )
    lines.extend(["", "## Discrepancy explanation", ""])
    lines.extend(f"- {item}" for item in report["comparison"]["explanation"])
    lines.append("")
    return "\n".join(lines)


def reconcile(
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    device_name: str | None = None,
    expected_test_samples: int = EXPECTED_TEST_SAMPLES,
) -> dict[str, Any]:
    """Run one canonical test evaluation and write reconciled artifacts."""
    config = load_config(config_path)
    results_dir = Path(config["output"]["results_dir"])
    archive_dir = ensure_archive(results_dir)
    old_snapshot = old_result_snapshot(archive_dir)

    device = resolve_device(device_name)
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

    metrics, y_true, y_pred, probabilities = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        class_names=class_names,
    )
    save_evaluation_outputs(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
        dataset=loader.dataset,
        output_dir=results_dir,
        metrics_filename="metrics_test.json",
        save_predictions=True,
    )

    metrics_path = results_dir / "metrics_test.json"
    canonical_metrics = read_json(metrics_path)
    if canonical_metrics != metrics:
        raise RuntimeError("Saved metrics_test.json differs from in-memory evaluation.")

    report_json_path = results_dir / "reconciliation_report.json"
    run_summary = build_run_summary(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        checkpoint=checkpoint,
        classes=class_names,
        test_validation=manifest_validation,
        report_path=report_json_path,
    )
    write_json(results_dir / "run_summary.json", run_summary)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "results_dir": str(results_dir),
        },
        "validations": {
            "classes": list(class_names),
            "checkpoint_class_to_idx": checkpoint_mapping,
            "manifest": {
                "sample_count": manifest_validation["sample_count"],
                "support": manifest_validation["support"],
                "synthetic_count": manifest_validation["synthetic_count"],
            },
            "dataset": dataset_validation,
            "checkpoint": {
                "epoch": checkpoint.get("epoch"),
                "best_validation_macro_f1": checkpoint.get(
                    "best_validation_macro_f1"
                ),
                "architecture": checkpoint.get("architecture"),
            },
        },
        "old_results": old_snapshot,
        "canonical_results": {
            "metrics_test": canonical_metrics,
            "run_summary": run_summary,
        },
        "comparison": {
            "explanation": build_discrepancy_explanation(
                old_snapshot=old_snapshot,
                canonical_metrics=canonical_metrics,
                checkpoint=checkpoint,
            )
        },
        "outputs": {
            "metrics_test": str(metrics_path),
            "classification_report": str(results_dir / "classification_report.csv"),
            "confusion_matrix": str(results_dir / "confusion_matrix.png"),
            "confusion_matrix_normalized": str(
                results_dir / "confusion_matrix_normalized.png"
            ),
            "test_predictions": str(results_dir / "test_predictions.csv"),
            "run_summary": str(results_dir / "run_summary.json"),
            "reconciliation_report_json": str(report_json_path),
            "reconciliation_report_md": str(
                results_dir / "reconciliation_report.md"
            ),
        },
    }
    write_json(report_json_path, report)
    (results_dir / "reconciliation_report.md").write_text(
        build_report_markdown(report),
        encoding="utf-8",
    )
    return report


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    report = reconcile(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        device_name=args.device,
        expected_test_samples=int(args.expected_test_samples),
    )
    metrics = report["canonical_results"]["metrics_test"]
    print(
        yaml.safe_dump(
            {
                "test_samples": report["validations"]["manifest"]["sample_count"],
                "accuracy": metrics["accuracy"],
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
