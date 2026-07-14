"""Evaluate test-time augmentation policies for ResNet18 V2 without training."""

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
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.dataset import EXPECTED_CLASSES, OrderedImageFolder
from src.vision.evaluation import image_paths, load_checkpoint
from src.vision.model import create_model
from src.vision.tta import (
    POLICIES,
    TTASplitResult,
    aggregate_logits,
    available_policy_names,
    build_tta_tensor_transform,
    collect_tta_logits,
    evaluate_tta_logits,
    get_policy,
    repo_path,
    result_to_summary_row,
    select_policy,
)
from src.vision.train_v2 import prepare_v2_config, resolve_device


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vision_v2_resnet18.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "vision" / "resultados_2_mejoras" / "07_tta"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate TTA policies on validation and test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    """Run validation policy selection and one final test evaluation."""
    args = parse_args()
    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    config = prepare_v2_config(load_yaml_config(config_path), smoke_test=False)
    checkpoint_path = resolve_project_path(args.checkpoint or config["output"]["checkpoint_path"])
    class_names = list(config.get("classes", EXPECTED_CLASSES))
    device = resolve_device(args.device)
    model = load_model(
        checkpoint_path=checkpoint_path,
        config=config,
        class_count=len(class_names),
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_results = evaluate_validation_policies(
        model=model,
        config=config,
        class_names=class_names,
        device=device,
    )
    selected_validation, enabled_by_validation = select_policy(validation_results)

    write_validation_results(output_dir / "tta_validation_results.csv", validation_results)
    write_latency_csv(output_dir / "tta_latency.csv", validation_results)

    selected_policy = selected_validation.policy_name
    selected_test = evaluate_split_for_policy(
        model=model,
        config=config,
        class_names=class_names,
        device=device,
        policy_name=selected_policy,
        split="test",
        temperature=selected_validation.temperature,
    )
    append_latency_csv(output_dir / "tta_latency.csv", selected_test)
    write_test_predictions(
        output_path=output_dir / "tta_predictions.csv",
        result=selected_test,
        dataset=build_split_dataset(config=config, policy_name=selected_policy, split="test"),
        class_names=class_names,
    )
    selected_payload = build_selected_payload(
        selected_validation=selected_validation,
        validation_results=validation_results,
        enabled_by_validation=enabled_by_validation,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        class_names=class_names,
    )
    write_json(output_dir / "selected_tta_policy.json", selected_payload)
    write_json(
        output_dir / "tta_test_results.json",
        {
            "split": "test",
            "policy": selected_test.policy_name,
            "tta_enabled": bool(enabled_by_validation),
            "views": selected_test.view_count,
            "temperature": selected_test.temperature,
            "aggregation": "mean_logits",
            "metrics": selected_test.metrics,
            "latency_seconds_total": selected_test.latency_seconds_total,
            "latency_seconds_per_image": selected_test.latency_seconds_per_image,
            "evaluated_once_after_validation_selection": True,
            "generated_at_utc": utc_now(),
        },
    )
    write_plots(
        output_dir=output_dir,
        validation_results=validation_results,
        selected_validation=selected_validation,
        selected_test=selected_test,
        config=config,
        model=model,
        class_names=class_names,
        device=device,
    )
    print_summary(
        output_dir=output_dir,
        selected_validation=selected_validation,
        selected_test=selected_test,
        enabled_by_validation=enabled_by_validation,
    )
    return 0


def evaluate_validation_policies(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    class_names: Sequence[str],
    device: torch.device,
) -> list[TTASplitResult]:
    """Evaluate all candidate policies on validation only."""
    results: list[TTASplitResult] = []
    for policy_name in available_policy_names():
        results.append(
            evaluate_split_for_policy(
                model=model,
                config=config,
                class_names=class_names,
                device=device,
                policy_name=policy_name,
                split="validation",
                temperature=None,
            )
        )
    return results


def evaluate_split_for_policy(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    class_names: Sequence[str],
    device: torch.device,
    policy_name: str,
    split: str,
    temperature: float | None,
) -> TTASplitResult:
    """Evaluate one policy on one split using averaged logits."""
    dataset = build_split_dataset(config=config, policy_name=policy_name, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    logits, labels, elapsed_seconds = collect_tta_logits(
        model=model,
        loader=loader,
        device=device,
    )
    return evaluate_tta_logits(
        logits=logits,
        labels=labels,
        class_names=class_names,
        policy_name=policy_name,
        split=split,
        latency_seconds_total=elapsed_seconds,
        view_count=get_policy(policy_name).view_count,
        temperature=temperature,
    )


def build_split_dataset(
    *,
    config: dict[str, Any],
    policy_name: str,
    split: str,
) -> OrderedImageFolder:
    """Build a deterministic split dataset that returns stacked TTA views."""
    data_root = Path(str(config["data"]["root"]))
    return OrderedImageFolder(
        data_root / split,
        expected_classes=list(config.get("classes", EXPECTED_CLASSES)),
        transform=build_tta_tensor_transform(
            policy_name=policy_name,
            image_size=int(config["image_size"]),
            auto_crop=bool(config.get("preprocessing", {}).get("auto_crop", True)),
        ),
    )


def load_model(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    class_count: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load ResNet18 V2 weights without training."""
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


def build_selected_payload(
    *,
    selected_validation: TTASplitResult,
    validation_results: Sequence[TTASplitResult],
    enabled_by_validation: bool,
    checkpoint_path: Path,
    config_path: Path,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Build the selected policy JSON used by Streamlit."""
    baseline = next(result for result in validation_results if result.policy_name == "none")
    best_candidate = max(validation_results, key=lambda result: result_to_order_tuple(result))
    return {
        "selected_policy": selected_validation.policy_name,
        "best_validation_candidate": best_candidate.policy_name,
        "tta_enabled": bool(enabled_by_validation),
        "default_enabled": bool(enabled_by_validation),
        "reason": (
            "enabled: validation macro-F1 improved over no TTA"
            if enabled_by_validation
            else "disabled: no candidate improved validation macro-F1 over no TTA"
        ),
        "aggregation": "mean_logits",
        "temperature": selected_validation.temperature,
        "views": selected_validation.view_count,
        "policy_description": POLICIES[selected_validation.policy_name].description,
        "selection_rules": [
            "validation macro-F1",
            "validation recall intact",
            "validation recall broken",
            "lower validation latency",
            "fallback to none unless macro-F1 improves over no TTA",
        ],
        "validation_baseline_macro_f1": baseline.metrics["macro_f1"],
        "validation_selected_macro_f1": selected_validation.metrics["macro_f1"],
        "validation_selected_recall_intact": selected_validation.metrics["recall_intact"],
        "validation_selected_recall_broken": selected_validation.metrics["recall_broken"],
        "validation_selected_latency_seconds_per_image": (
            selected_validation.latency_seconds_per_image
        ),
        "checkpoint": repo_path(checkpoint_path, PROJECT_ROOT),
        "config": repo_path(config_path, PROJECT_ROOT),
        "class_names": list(class_names),
        "generated_at_utc": utc_now(),
    }


def result_to_order_tuple(result: TTASplitResult) -> tuple[float, float, float, float]:
    """Return the policy ordering tuple used for reporting the best candidate."""
    return (
        float(result.metrics["macro_f1"]),
        float(result.metrics["recall_intact"]),
        float(result.metrics["recall_broken"]),
        -float(result.latency_seconds_per_image),
    )


def write_validation_results(path: Path, results: Sequence[TTASplitResult]) -> None:
    """Write validation policy metrics to CSV."""
    pd.DataFrame([result_to_summary_row(result) for result in results]).to_csv(
        path,
        index=False,
    )


def write_latency_csv(path: Path, results: Sequence[TTASplitResult]) -> None:
    """Write latency rows for validation policy comparison."""
    rows = [latency_row(result) for result in results]
    pd.DataFrame(rows).to_csv(path, index=False)


def append_latency_csv(path: Path, result: TTASplitResult) -> None:
    """Append final selected test latency to the latency CSV."""
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    frame = pd.concat([existing, pd.DataFrame([latency_row(result)])], ignore_index=True)
    frame.to_csv(path, index=False)


def latency_row(result: TTASplitResult) -> dict[str, Any]:
    """Return one latency row."""
    baseline_factor = result.view_count
    return {
        "split": result.split,
        "policy": result.policy_name,
        "views": result.view_count,
        "latency_seconds_total": result.latency_seconds_total,
        "latency_seconds_per_image": result.latency_seconds_per_image,
        "expected_forward_pass_multiplier": baseline_factor,
    }


def write_test_predictions(
    *,
    output_path: Path,
    result: TTASplitResult,
    dataset: OrderedImageFolder,
    class_names: Sequence[str],
) -> None:
    """Write one final test prediction row per image."""
    paths = image_paths(dataset)
    rows: list[dict[str, Any]] = []
    probabilities = result.probabilities.detach().cpu().tolist()
    logits = result.logits.detach().cpu().tolist()
    for index, (true_index, pred_index, row_probabilities, row_logits) in enumerate(
        zip(result.y_true, result.y_pred, probabilities, logits, strict=True)
    ):
        row: dict[str, Any] = {
            "image_path": paths[index] if index < len(paths) else "",
            "true_label": class_names[int(true_index)],
            "predicted_label": class_names[int(pred_index)],
            "predicted_probability": float(row_probabilities[int(pred_index)]),
            "policy": result.policy_name,
            "views": result.view_count,
            "temperature": result.temperature,
            "aggregation": "mean_logits",
        }
        for class_index, class_name in enumerate(class_names):
            row[f"probability_{class_name}"] = float(row_probabilities[class_index])
            row[f"logit_{class_name}"] = float(row_logits[class_index])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_plots(
    *,
    output_dir: Path,
    validation_results: Sequence[TTASplitResult],
    selected_validation: TTASplitResult,
    selected_test: TTASplitResult,
    config: dict[str, Any],
    model: torch.nn.Module,
    class_names: Sequence[str],
    device: torch.device,
) -> None:
    """Write the required TTA PNG artifacts."""
    plot_validation_policies(
        validation_results=validation_results,
        output_path=output_dir / "r2_tta_politicas_validation.png",
    )
    plot_without_vs_selected(
        validation_results=validation_results,
        selected_validation=selected_validation,
        output_path=output_dir / "r2_sin_tta_vs_tta.png",
    )
    plot_orientation_stability(
        model=model,
        config=config,
        class_names=class_names,
        device=device,
        policy_name=selected_validation.policy_name,
        output_path=output_dir / "r2_estabilidad_orientacion.png",
    )
    plot_latency(
        validation_results=validation_results,
        selected_test=selected_test,
        output_path=output_dir / "r2_latencia_tta.png",
    )


def plot_validation_policies(
    *,
    validation_results: Sequence[TTASplitResult],
    output_path: Path,
) -> None:
    """Plot macro-F1 and critical recalls for validation policies."""
    policies = [result.policy_name for result in validation_results]
    x_positions = range(len(policies))
    figure, axis = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    width = 0.24
    axis.bar(
        [x - width for x in x_positions],
        [result.metrics["macro_f1"] for result in validation_results],
        width=width,
        label="macro-F1",
        color="#2563eb",
    )
    axis.bar(
        list(x_positions),
        [result.metrics["recall_intact"] for result in validation_results],
        width=width,
        label="recall intact",
        color="#0f766e",
    )
    axis.bar(
        [x + width for x in x_positions],
        [result.metrics["recall_broken"] for result in validation_results],
        width=width,
        label="recall broken",
        color="#d97706",
    )
    axis.set_xticks(list(x_positions), policies)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Politicas TTA en validation")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_without_vs_selected(
    *,
    validation_results: Sequence[TTASplitResult],
    selected_validation: TTASplitResult,
    output_path: Path,
) -> None:
    """Plot no-TTA versus selected policy on validation only."""
    baseline = next(result for result in validation_results if result.policy_name == "none")
    metrics = ["macro_f1", "recall_intact", "recall_broken"]
    x_positions = range(len(metrics))
    figure, axis = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
    axis.bar(
        [x - 0.18 for x in x_positions],
        [baseline.metrics[metric] for metric in metrics],
        width=0.36,
        label="sin TTA",
        color="#667085",
    )
    axis.bar(
        [x + 0.18 for x in x_positions],
        [selected_validation.metrics[metric] for metric in metrics],
        width=0.36,
        label=f"TTA {selected_validation.policy_name}",
        color="#2563eb",
    )
    axis.set_xticks(list(x_positions), ["macro-F1", "recall intact", "recall broken"])
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Sin TTA vs politica seleccionada en validation")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_orientation_stability(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    class_names: Sequence[str],
    device: torch.device,
    policy_name: str,
    output_path: Path,
) -> None:
    """Plot validation prediction stability across selected policy views."""
    policy = get_policy(policy_name)
    dataset = build_split_dataset(config=config, policy_name=policy_name, split="validation")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    view_predictions: list[torch.Tensor] = []
    aggregate_predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, _ in loader:
            batch_size, view_count = int(inputs.shape[0]), int(inputs.shape[1])
            flattened = inputs.reshape(batch_size * view_count, *inputs.shape[2:]).to(device)
            logits = model(flattened).detach().cpu().reshape(batch_size, view_count, -1)
            view_predictions.append(logits.argmax(dim=2))
            aggregate_predictions.append(aggregate_logits(logits).argmax(dim=1))
    if not view_predictions:
        return
    per_view = torch.cat(view_predictions, dim=0)
    aggregate = torch.cat(aggregate_predictions, dim=0)
    original = per_view[:, 0]
    agreement_original = [
        float(per_view[:, index].eq(original).float().mean().item())
        for index in range(per_view.shape[1])
    ]
    agreement_aggregate = [
        float(per_view[:, index].eq(aggregate).float().mean().item())
        for index in range(per_view.shape[1])
    ]
    x_positions = range(len(policy.views))
    figure, axis = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    axis.bar(
        [x - 0.18 for x in x_positions],
        agreement_original,
        width=0.36,
        label="acuerdo con original",
        color="#0f766e",
    )
    axis.bar(
        [x + 0.18 for x in x_positions],
        agreement_aggregate,
        width=0.36,
        label="acuerdo con promedio",
        color="#2563eb",
    )
    axis.set_xticks(list(x_positions), [view.name for view in policy.views], rotation=20)
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"Estabilidad de orientacion en validation ({policy_name})")
    axis.set_ylabel("proporcion de acuerdo")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_latency(
    *,
    validation_results: Sequence[TTASplitResult],
    selected_test: TTASplitResult,
    output_path: Path,
) -> None:
    """Plot validation latency by policy and final selected test latency."""
    labels = [f"val {result.policy_name}" for result in validation_results] + [
        f"test {selected_test.policy_name}"
    ]
    values = [result.latency_seconds_per_image for result in validation_results] + [
        selected_test.latency_seconds_per_image
    ]
    colors = ["#667085", "#2563eb", "#d97706"][: len(validation_results)] + ["#0f766e"]
    figure, axis = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    axis.bar(labels, values, color=colors)
    axis.set_title("Latencia por imagen")
    axis.set_ylabel("segundos")
    axis.grid(True, axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def utc_now() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def print_summary(
    *,
    output_dir: Path,
    selected_validation: TTASplitResult,
    selected_test: TTASplitResult,
    enabled_by_validation: bool,
) -> None:
    """Print the final TTA evaluation summary."""
    print(f"selected_policy: {selected_validation.policy_name}")
    print(f"tta_enabled: {enabled_by_validation}")
    print(f"validation_macro_f1: {selected_validation.metrics['macro_f1']:.6f}")
    print(f"validation_recall_intact: {selected_validation.metrics['recall_intact']:.6f}")
    print(f"validation_recall_broken: {selected_validation.metrics['recall_broken']:.6f}")
    print(f"test_macro_f1: {selected_test.metrics['macro_f1']:.6f}")
    print(f"test_recall_intact: {selected_test.metrics['recall_intact']:.6f}")
    print(f"test_recall_broken: {selected_test.metrics['recall_broken']:.6f}")
    print(f"latency_seconds_per_image: {selected_test.latency_seconds_per_image:.6f}")
    print(f"views: {selected_test.view_count}")
    print(f"temperature: {selected_test.temperature:.6f}")
    print(f"artifacts_dir: {repo_path(output_dir, PROJECT_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
