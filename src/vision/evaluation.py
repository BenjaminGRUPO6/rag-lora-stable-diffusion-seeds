from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Run inference over a dataloader and return labels, predictions and probabilities."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
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
    return y_true, y_pred, probabilities


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], list[int], list[int], list[list[float]]]:
    """Evaluate a model on one split."""
    y_true, y_pred, probabilities = predict(model=model, loader=loader, device=device)
    return compute_metrics(y_true, y_pred, class_names), y_true, y_pred, probabilities


def save_evaluation_outputs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
    dataset: Dataset,
    output_dir: str | Path,
    metrics_filename: str = "metrics_test.json",
    save_predictions: bool = True,
) -> dict[str, Any]:
    """Save metrics, report, confusion matrices and optional image predictions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, class_names)
    _write_json(output / metrics_filename, metrics)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output / "classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize="true",
    )
    _plot_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        output_path=output / "confusion_matrix.png",
        title="Confusion matrix",
        value_format="d",
    )
    _plot_confusion_matrix(
        matrix=normalized,
        class_names=class_names,
        output_path=output / "confusion_matrix_normalized.png",
        title="Normalized confusion matrix",
        value_format=".2f",
    )
    if save_predictions:
        save_predictions_csv(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_names=class_names,
            dataset=dataset,
            output_path=output / "test_predictions.csv",
        )
    return metrics


def save_predictions_csv(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
    dataset: Dataset,
    output_path: str | Path,
) -> None:
    """Write one prediction row per image with the predicted probability."""
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


def image_paths(dataset: Dataset) -> list[str]:
    """Return image paths in dataset iteration order for ImageFolder or Subset."""
    if isinstance(dataset, Subset):
        paths = image_paths(dataset.dataset)
        return [paths[int(index)] for index in dataset.indices]
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return []
    return [str(Path(sample[0])) for sample in samples]


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a training checkpoint on the requested device."""
    return torch.load(Path(path), map_location=device, weights_only=False)


def _plot_confusion_matrix(
    matrix: Any,
    class_names: Sequence[str],
    output_path: Path,
    title: str,
    value_format: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
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
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
