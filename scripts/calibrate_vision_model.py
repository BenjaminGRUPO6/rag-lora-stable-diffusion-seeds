"""Calibrate ResNet18 V2 confidence with temperature scaling."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.calibration import (
    calibration_bins,
    classes_unchanged_after_temperature,
    evaluate_calibration,
    fit_temperature,
    softmax_with_temperature,
    write_bins_csv,
    write_json,
)
from src.vision.evaluation import load_checkpoint
from src.vision.model import create_model
from src.vision.train_v2 import create_v2_dataloaders, prepare_v2_config, resolve_device


OUTPUT_DIR = PROJECT_ROOT / "results" / "vision" / "resultados_2_mejoras" / "06_calibracion"
TEMPERATURE_PATH = PROJECT_ROOT / "models" / "vision" / "resnet18_v2_temperature.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vision_v2_resnet18.yaml"
DEFAULT_BINS = 10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calibrate ResNet18 V2 with temperature scaling.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--temperature-path", type=Path, default=TEMPERATURE_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    return parser.parse_args()


def main() -> int:
    """Run calibration and write metrics, plots and temperature JSON."""
    args = parse_args()
    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    temperature_path = resolve_project_path(args.temperature_path)
    config = prepare_v2_config(load_yaml_config(config_path), smoke_test=False)
    checkpoint_path = resolve_project_path(args.checkpoint or config["output"]["checkpoint_path"])
    class_names = list(config.get("classes", []))
    if not class_names:
        raise ValueError("V2 config must define classes.")

    device = resolve_device(args.device)
    loaders = create_v2_dataloaders(
        data_root=config["data"]["root"],
        classes=class_names,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
        seed=int(config["seed"]),
        auto_crop=bool(config["preprocessing"].get("auto_crop", True)),
        smoke_test=False,
    )
    model = load_model(
        checkpoint_path=checkpoint_path,
        config=config,
        class_count=len(class_names),
        device=device,
    )

    validation_logits, validation_labels = collect_logits(model, loaders.validation, device)
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_before = evaluate_calibration(
        validation_logits,
        validation_labels,
        class_names,
        temperature=1.0,
        n_bins=int(args.bins),
    )
    validation_after = evaluate_calibration(
        validation_logits,
        validation_labels,
        class_names,
        temperature=temperature,
        n_bins=int(args.bins),
    )

    test_logits, test_labels = collect_logits(model, loaders.test, device)
    before_metrics = evaluate_calibration(
        test_logits,
        test_labels,
        class_names,
        temperature=1.0,
        n_bins=int(args.bins),
    )
    after_metrics = evaluate_calibration(
        test_logits,
        test_labels,
        class_names,
        temperature=temperature,
        n_bins=int(args.bins),
    )
    if not classes_unchanged_after_temperature(test_logits, temperature):
        raise RuntimeError("Temperature scaling changed predicted classes on test.")

    output_dir.mkdir(parents=True, exist_ok=True)
    before_payload = build_metrics_payload(
        metrics=before_metrics.to_dict(),
        split="test",
        temperature=1.0,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        calibration_applied=False,
    )
    after_payload = build_metrics_payload(
        metrics=after_metrics.to_dict(),
        split="test",
        temperature=temperature,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        calibration_applied=True,
    )
    write_json(output_dir / "calibration_metrics_before.json", before_payload)
    write_json(output_dir / "calibration_metrics_after.json", after_payload)

    temperature_payload = {
        "temperature": float(temperature),
        "method": "temperature_scaling",
        "optimized_on": "validation",
        "evaluated_on": "test",
        "checkpoint": repo_path(checkpoint_path),
        "class_names": class_names,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "validation_nll_before": validation_before.nll,
        "validation_nll_after": validation_after.nll,
        "test_nll_before": before_metrics.nll,
        "test_nll_after": after_metrics.nll,
    }
    write_json(output_dir / "temperature.json", temperature_payload)
    write_json(temperature_path, temperature_payload)

    before_bins = calibration_bins(
        softmax_with_temperature(test_logits, 1.0),
        test_labels,
        n_bins=int(args.bins),
    )
    after_bins = calibration_bins(
        softmax_with_temperature(test_logits, temperature),
        test_labels,
        n_bins=int(args.bins),
    )
    write_bins_csv(output_dir / "calibration_bins.csv", before_bins, after_bins)
    write_plots(
        output_dir=output_dir,
        before_bins=before_bins,
        after_bins=after_bins,
        before_metrics=before_metrics.to_dict(),
        after_metrics=after_metrics.to_dict(),
    )

    print_summary(
        temperature=temperature,
        before=before_metrics.to_dict(),
        after=after_metrics.to_dict(),
        output_dir=output_dir,
        temperature_path=temperature_path,
    )
    return 0


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def load_model(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    class_count: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load ResNet18 V2 weights without starting any training loop."""
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model_config = dict(config.get("model", {}))
    checkpoint_config = checkpoint.get("config", {})
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        model_config.update(checkpoint_config["model"])
    model = create_model(
        architecture=str(model_config.get("architecture", "resnet18")),
        num_classes=class_count,
        pretrained=False,
        dropout=float(model_config.get("dropout", 0.2)),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and labels from one split in deterministic eval mode."""
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = inputs.to(device)
            batch_logits = model(inputs).detach().cpu()
            logits.append(batch_logits)
            labels.append(batch_labels.detach().cpu().long())
    if not logits:
        raise ValueError("No logits collected; split is empty.")
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def build_metrics_payload(
    *,
    metrics: dict[str, float],
    split: str,
    temperature: float,
    checkpoint_path: Path,
    class_names: Sequence[str],
    calibration_applied: bool,
) -> dict[str, Any]:
    """Build a JSON payload for one before/after calibration evaluation."""
    return {
        **metrics,
        "split": split,
        "temperature": float(temperature),
        "checkpoint": repo_path(checkpoint_path),
        "class_names": list(class_names),
        "calibration_applied": calibration_applied,
        "argmax_policy": "raw_logits",
    }


def write_plots(
    *,
    output_dir: Path,
    before_bins: Sequence[dict[str, float | int]],
    after_bins: Sequence[dict[str, float | int]],
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> None:
    """Write reliability and confidence-gap PNG artifacts."""
    plot_reliability(
        bins=before_bins,
        output_path=output_dir / "r2_reliability_before.png",
        title="Reliability before calibration",
        color="#2563eb",
    )
    plot_reliability(
        bins=after_bins,
        output_path=output_dir / "r2_reliability_after.png",
        title="Reliability after calibration",
        color="#0f766e",
    )
    plot_before_vs_after(
        before_bins=before_bins,
        after_bins=after_bins,
        output_path=output_dir / "r2_calibration_before_vs_after.png",
    )
    plot_confidence_accuracy_gap(
        before=before_metrics,
        after=after_metrics,
        output_path=output_dir / "r2_confidence_accuracy_gap.png",
    )


def plot_reliability(
    *,
    bins: Sequence[dict[str, float | int]],
    output_path: Path,
    title: str,
    color: str,
) -> None:
    """Plot one reliability diagram."""
    x_values = [float(row["confidence"]) for row in bins if int(row["count"]) > 0]
    y_values = [float(row["accuracy"]) for row in bins if int(row["count"]) > 0]
    figure, axis = plt.subplots(figsize=(6.5, 6.0), facecolor="white")
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#111827", linewidth=1.0, label="ideal")
    axis.plot(x_values, y_values, marker="o", color=color, label="model")
    axis.set_title(title)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_before_vs_after(
    *,
    before_bins: Sequence[dict[str, float | int]],
    after_bins: Sequence[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Plot before and after reliability curves together."""
    figure, axis = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#111827", linewidth=1.0, label="ideal")
    for rows, color, label in (
        (before_bins, "#2563eb", "before"),
        (after_bins, "#0f766e", "after"),
    ):
        x_values = [float(row["confidence"]) for row in rows if int(row["count"]) > 0]
        y_values = [float(row["accuracy"]) for row in rows if int(row["count"]) > 0]
        axis.plot(x_values, y_values, marker="o", color=color, label=label)
    axis.set_title("Calibration before vs after")
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_confidence_accuracy_gap(
    *,
    before: dict[str, float],
    after: dict[str, float],
    output_path: Path,
) -> None:
    """Plot mean confidence, accuracy and signed confidence-accuracy gap."""
    labels = ["confidence", "accuracy", "gap"]
    before_values = [
        float(before["mean_confidence"]),
        float(before["accuracy"]),
        float(before["confidence_accuracy_gap"]),
    ]
    after_values = [
        float(after["mean_confidence"]),
        float(after["accuracy"]),
        float(after["confidence_accuracy_gap"]),
    ]
    x_positions = range(len(labels))
    figure, axis = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
    axis.bar([x - 0.18 for x in x_positions], before_values, width=0.36, label="before", color="#2563eb")
    axis.bar([x + 0.18 for x in x_positions], after_values, width=0.36, label="after", color="#0f766e")
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xticks(list(x_positions), labels)
    axis.set_title("Confidence and accuracy gap")
    axis.set_ylim(min(-0.2, min(before_values + after_values) - 0.05), 1.05)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def repo_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def print_summary(
    *,
    temperature: float,
    before: dict[str, float],
    after: dict[str, float],
    output_dir: Path,
    temperature_path: Path,
) -> None:
    """Print the calibration result summary expected by the runbook."""
    print(f"temperature: {temperature:.6f}")
    print(f"ECE before/after: {before['ece']:.6f} -> {after['ece']:.6f}")
    print(f"NLL before/after: {before['nll']:.6f} -> {after['nll']:.6f}")
    print(
        "Brier before/after: "
        f"{before['multiclass_brier']:.6f} -> {after['multiclass_brier']:.6f}"
    )
    print(
        "mean confidence before/after: "
        f"{before['mean_confidence']:.6f} -> {after['mean_confidence']:.6f}"
    )
    print(f"temperature_json: {repo_path(temperature_path)}")
    print(f"artifacts_dir: {repo_path(output_dir)}")


if __name__ == "__main__":
    raise SystemExit(main())
