from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass(frozen=True)
class CalibrationMetrics:
    """Aggregate calibration and classification metrics for one split."""

    ece: float
    nll: float
    multiclass_brier: float
    confidence_accuracy_gap: float
    accuracy: float
    macro_f1: float
    mean_confidence: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable metrics dictionary."""
        return {
            "ece": self.ece,
            "nll": self.nll,
            "multiclass_brier": self.multiclass_brier,
            "confidence_accuracy_gap": self.confidence_accuracy_gap,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "mean_confidence": self.mean_confidence,
        }


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 100,
) -> float:
    """Optimize one positive temperature parameter by minimizing validation NLL."""
    prepared_logits = logits.detach().float()
    prepared_labels = labels.detach().long()
    if prepared_logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor.")
    if prepared_labels.ndim != 1 or prepared_labels.numel() != prepared_logits.shape[0]:
        raise ValueError("labels must be a 1D tensor with one item per logit row.")

    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp_min(1e-6)
        loss = criterion(prepared_logits / temperature, prepared_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp_min(1e-6).item())


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return probabilities from logits scaled by a positive temperature."""
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero.")
    return torch.softmax(logits.float() / float(temperature), dim=-1)


def compute_ece(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error using confidence bins."""
    return float(sum(bin_row["weighted_abs_gap"] for bin_row in calibration_bins(probabilities, labels, n_bins=n_bins)))


def calibration_bins(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return reliability statistics for equally spaced confidence bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    prepared_probabilities = probabilities.detach().float().cpu()
    prepared_labels = labels.detach().long().cpu()
    confidences, predictions = prepared_probabilities.max(dim=1)
    correct = predictions.eq(prepared_labels).float()
    total = max(int(prepared_labels.numel()), 1)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum().item())
        if count:
            accuracy = float(correct[mask].mean().item())
            confidence = float(confidences[mask].mean().item())
        else:
            accuracy = 0.0
            confidence = 0.0
        abs_gap = abs(accuracy - confidence)
        rows.append(
            {
                "bin_index": bin_index,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "proportion": float(count / total),
                "accuracy": accuracy,
                "confidence": confidence,
                "abs_gap": float(abs_gap),
                "weighted_abs_gap": float((count / total) * abs_gap),
            }
        )
    return rows


def evaluate_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    *,
    temperature: float = 1.0,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute calibration metrics without changing argmax classification."""
    prepared_logits = logits.detach().float().cpu()
    prepared_labels = labels.detach().long().cpu()
    probabilities = softmax_with_temperature(prepared_logits, temperature)
    predictions = prepared_logits.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    y_true = prepared_labels.numpy()
    y_pred = predictions.numpy()
    one_hot = torch.nn.functional.one_hot(prepared_labels, num_classes=len(class_names)).float()
    brier = torch.sum((probabilities - one_hot) ** 2, dim=1).mean()
    macro_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0,
    )[2]
    accuracy = float(accuracy_score(y_true, y_pred))
    return CalibrationMetrics(
        ece=compute_ece(probabilities, prepared_labels, n_bins=n_bins),
        nll=float(torch.nn.functional.cross_entropy(prepared_logits / temperature, prepared_labels).item()),
        multiclass_brier=float(brier.item()),
        confidence_accuracy_gap=float(confidences.mean().item() - accuracy),
        accuracy=accuracy,
        macro_f1=float(macro_f1),
        mean_confidence=float(confidences.mean().item()),
    )


def classes_unchanged_after_temperature(logits: torch.Tensor, temperature: float) -> bool:
    """Return true when positive temperature scaling preserves argmax classes."""
    before = logits.detach().float().argmax(dim=1)
    after = softmax_with_temperature(logits, temperature).argmax(dim=1)
    return bool(torch.equal(before.cpu(), after.cpu()))


def load_temperature(path: str | Path) -> float | None:
    """Load a positive temperature from JSON, returning None when unavailable."""
    temperature_path = Path(path)
    if not temperature_path.exists():
        return None
    payload = json.loads(temperature_path.read_text(encoding="utf-8"))
    value = float(payload["temperature"])
    if value <= 0.0:
        raise ValueError(f"Invalid non-positive temperature in {temperature_path}.")
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON using stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_bins_csv(
    path: str | Path,
    before: Sequence[dict[str, float | int]],
    after: Sequence[dict[str, float | int]],
) -> None:
    """Write before/after calibration bins to one CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bin_index",
        "bin_lower",
        "bin_upper",
        "count_before",
        "accuracy_before",
        "confidence_before",
        "abs_gap_before",
        "count_after",
        "accuracy_after",
        "confidence_after",
        "abs_gap_after",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for before_row, after_row in zip(before, after, strict=True):
            writer.writerow(
                {
                    "bin_index": int(before_row["bin_index"]),
                    "bin_lower": f"{float(before_row['bin_lower']):.6f}",
                    "bin_upper": f"{float(before_row['bin_upper']):.6f}",
                    "count_before": int(before_row["count"]),
                    "accuracy_before": f"{float(before_row['accuracy']):.12f}",
                    "confidence_before": f"{float(before_row['confidence']):.12f}",
                    "abs_gap_before": f"{float(before_row['abs_gap']):.12f}",
                    "count_after": int(after_row["count"]),
                    "accuracy_after": f"{float(after_row['accuracy']):.12f}",
                    "confidence_after": f"{float(after_row['confidence']):.12f}",
                    "abs_gap_after": f"{float(after_row['abs_gap']):.12f}",
                }
            )


def probabilities_to_numpy(probabilities: torch.Tensor) -> np.ndarray:
    """Return probabilities as a detached CPU NumPy array."""
    return probabilities.detach().float().cpu().numpy()
