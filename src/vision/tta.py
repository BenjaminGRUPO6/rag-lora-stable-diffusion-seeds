from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
from PIL import Image, ImageStat
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.vision.calibration import fit_temperature, softmax_with_temperature
from src.vision.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.vision.preprocessing import PreprocessingConfig, preprocess_image

if TYPE_CHECKING:
    from src.vision.inference import ImageInput, VisionInferenceEngine


@dataclass(frozen=True)
class TTAView:
    """One class-preserving test-time view."""

    name: str
    transform: Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class TTAPolicy:
    """A deterministic TTA policy made of class-preserving views."""

    name: str
    views: tuple[TTAView, ...]
    description: str

    @property
    def view_count(self) -> int:
        """Return the number of deterministic views in the policy."""
        return len(self.views)


@dataclass(frozen=True)
class TTASplitResult:
    """Evaluation result for one policy on one split."""

    policy_name: str
    split: str
    temperature: float
    metrics: dict[str, Any]
    y_true: list[int]
    y_pred: list[int]
    logits: torch.Tensor
    probabilities: torch.Tensor
    latency_seconds_total: float
    latency_seconds_per_image: float
    view_count: int


def identity(image: Image.Image) -> Image.Image:
    """Return the original RGB view."""
    return image.convert("RGB")


def horizontal_flip(image: Image.Image) -> Image.Image:
    """Return a horizontal flip, preserving visible defect type signals."""
    return image.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def rotate_degrees(degrees: float) -> Callable[[Image.Image], Image.Image]:
    """Build a small in-plane rotation that avoids color or morphology changes."""

    def _rotate(image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        fill = tuple(int(value) for value in ImageStat.Stat(rgb).median)
        return rgb.rotate(
            degrees,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=fill,
        )

    return _rotate


POLICIES: dict[str, TTAPolicy] = {
    "none": TTAPolicy(
        name="none",
        views=(TTAView("original", identity),),
        description="Vista original sin aumento en inferencia.",
    ),
    "light": TTAPolicy(
        name="light",
        views=(
            TTAView("original", identity),
            TTAView("horizontal_flip", horizontal_flip),
        ),
        description="Original y flip horizontal; no altera color, textura ni defectos visibles.",
    ),
    "standard": TTAPolicy(
        name="standard",
        views=(
            TTAView("original", identity),
            TTAView("horizontal_flip", horizontal_flip),
            TTAView("rotate_plus_5", rotate_degrees(5.0)),
            TTAView("rotate_minus_5", rotate_degrees(-5.0)),
        ),
        description=(
            "Original, flip horizontal y rotaciones leves +/-5 grados; evita "
            "recortes agresivos, cambios de color, blur, perspectiva o cualquier "
            "transformacion que pueda cambiar senales visuales de clase."
        ),
    ),
}


def get_policy(name: str) -> TTAPolicy:
    """Return a known policy by name."""
    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown TTA policy: {name}") from exc


def available_policy_names() -> list[str]:
    """Return candidate policies in evaluation order."""
    return ["none", "light", "standard"]


def aggregate_logits(view_logits: torch.Tensor) -> torch.Tensor:
    """Average raw logits over views.

    This implementation aggregates logits, not probabilities. The averaged logits
    are then temperature-scaled and converted to probabilities with softmax. This
    keeps the class decision explicit and lets validation fit a TTA-specific
    temperature because the averaged-logit distribution differs from single-view
    inference.
    """
    if view_logits.ndim != 3:
        raise ValueError("view_logits must have shape [batch, views, classes].")
    if view_logits.shape[1] <= 0:
        raise ValueError("view_logits must contain at least one view.")
    return view_logits.float().mean(dim=1)


def build_tta_tensor_transform(
    *,
    policy_name: str,
    image_size: int,
    auto_crop: bool,
) -> Callable[[Image.Image], torch.Tensor]:
    """Build a transform returning one stacked tensor per TTA policy."""
    policy = get_policy(policy_name)
    normalize = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    crop_config = PreprocessingConfig(output_size=image_size)

    def _transform(image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        base = preprocess_image(rgb, config=crop_config).crop if auto_crop else rgb
        tensors = [normalize(view.transform(base)) for view in policy.views]
        return torch.stack(tensors, dim=0)

    return _transform


def collect_tta_logits(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Collect averaged logits and labels from a loader yielding TTA view tensors."""
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    started_at = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels in loader:
            if inputs.ndim != 5:
                raise ValueError("TTA loader must return tensors shaped [batch, views, c, h, w].")
            batch_size, view_count = int(inputs.shape[0]), int(inputs.shape[1])
            flattened = inputs.reshape(batch_size * view_count, *inputs.shape[2:]).to(device)
            batch_logits = model(flattened).detach().cpu()
            batch_logits = batch_logits.reshape(batch_size, view_count, -1)
            logits.append(aggregate_logits(batch_logits))
            labels.append(batch_labels.detach().cpu().long())
    if not logits:
        raise ValueError("No logits collected; split is empty.")
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0), time.perf_counter() - started_at


def evaluate_tta_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    policy_name: str,
    split: str,
    latency_seconds_total: float,
    view_count: int,
    temperature: float | None = None,
) -> TTASplitResult:
    """Evaluate averaged TTA logits and return metrics plus probabilities."""
    prepared_logits = logits.detach().float().cpu()
    prepared_labels = labels.detach().long().cpu()
    if temperature is None:
        temperature = fit_temperature(prepared_logits, prepared_labels)
    probabilities = softmax_with_temperature(prepared_logits, temperature)
    y_true = [int(value) for value in prepared_labels.tolist()]
    y_pred = [int(value) for value in prepared_logits.argmax(dim=1).tolist()]
    metrics = compute_tta_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    sample_count = max(len(y_true), 1)
    return TTASplitResult(
        policy_name=policy_name,
        split=split,
        temperature=float(temperature),
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        logits=prepared_logits,
        probabilities=probabilities,
        latency_seconds_total=float(latency_seconds_total),
        latency_seconds_per_image=float(latency_seconds_total / sample_count),
        view_count=int(view_count),
    )


def compute_tta_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute aggregate and per-class metrics used for TTA policy selection."""
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
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
        "recall_intact": float(per_class.get("intact", {}).get("recall", 0.0)),
        "recall_broken": float(per_class.get("broken", {}).get("recall", 0.0)),
        "per_class": per_class,
    }


def policy_selection_key(result: TTASplitResult) -> tuple[float, float, float, float]:
    """Return the validation ordering key: macro-F1, intact, broken, lower latency."""
    return (
        float(result.metrics["macro_f1"]),
        float(result.metrics["recall_intact"]),
        float(result.metrics["recall_broken"]),
        -float(result.latency_seconds_per_image),
    )


def select_policy(validation_results: Sequence[TTASplitResult]) -> tuple[TTASplitResult, bool]:
    """Select the TTA policy using validation and keep no-TTA if macro-F1 does not improve."""
    if not validation_results:
        raise ValueError("At least one validation result is required.")
    by_name = {result.policy_name: result for result in validation_results}
    if "none" not in by_name:
        raise ValueError("Validation results must include the no-TTA 'none' policy.")
    best_candidate = max(validation_results, key=policy_selection_key)
    baseline = by_name["none"]
    improved = float(best_candidate.metrics["macro_f1"]) > float(baseline.metrics["macro_f1"])
    if best_candidate.policy_name == "none" or not improved:
        return baseline, False
    return best_candidate, True


def result_to_summary_row(result: TTASplitResult) -> dict[str, Any]:
    """Return a compact CSV-friendly row for one split result."""
    return {
        "split": result.split,
        "policy": result.policy_name,
        "views": result.view_count,
        "temperature": result.temperature,
        "accuracy": result.metrics["accuracy"],
        "macro_f1": result.metrics["macro_f1"],
        "macro_precision": result.metrics["macro_precision"],
        "macro_recall": result.metrics["macro_recall"],
        "recall_intact": result.metrics["recall_intact"],
        "recall_broken": result.metrics["recall_broken"],
        "latency_seconds_total": result.latency_seconds_total,
        "latency_seconds_per_image": result.latency_seconds_per_image,
    }


def predict_with_tta(
    *,
    engine: "VisionInferenceEngine",
    image: "ImageInput",
    policy_name: str,
    temperature: float,
) -> dict[str, Any]:
    """Run single-image inference with averaged TTA logits."""
    from src.vision.inference import load_rgb_image

    policy = get_policy(policy_name)
    pil_image = load_rgb_image(image)
    started_at = time.perf_counter()
    tensors = [
        engine.transform(view.transform(pil_image)).unsqueeze(0)
        for view in policy.views
    ]
    batch = torch.cat(tensors, dim=0).to(engine.device)
    with torch.no_grad():
        view_logits = engine.model(batch).detach().cpu().unsqueeze(0)
    averaged_logits = aggregate_logits(view_logits)[0]
    uncalibrated_probabilities_tensor = torch.softmax(averaged_logits, dim=0)
    probabilities_tensor = softmax_with_temperature(averaged_logits, temperature)
    prediction = prediction_dict_from_tensors(
        labels=engine.labels,
        logits_tensor=averaged_logits,
        probabilities_tensor=probabilities_tensor,
        uncalibrated_probabilities_tensor=uncalibrated_probabilities_tensor,
        temperature=temperature,
    )
    prediction["tta_enabled"] = True
    prediction["tta_policy"] = policy.name
    prediction["tta_views"] = policy.view_count
    prediction["tta_extra_seconds"] = max(time.perf_counter() - started_at, 0.0)
    prediction["aggregation"] = "mean_logits"
    return prediction


def prediction_dict_from_tensors(
    *,
    labels: Sequence[str],
    logits_tensor: torch.Tensor,
    probabilities_tensor: torch.Tensor,
    uncalibrated_probabilities_tensor: torch.Tensor,
    temperature: float,
) -> dict[str, Any]:
    """Build the inference dictionary shape from TTA tensors."""
    sorted_indices = torch.argsort(probabilities_tensor, descending=True).tolist()
    index = int(sorted_indices[0])
    second_index = int(sorted_indices[1]) if len(sorted_indices) >= 2 else None
    probabilities = {
        label: float(probabilities_tensor[label_index].item())
        for label_index, label in enumerate(labels)
    }
    uncalibrated_probabilities = {
        label: float(uncalibrated_probabilities_tensor[label_index].item())
        for label_index, label in enumerate(labels)
    }
    logits = {
        label: float(logits_tensor[label_index].item())
        for label_index, label in enumerate(labels)
    }
    second_confidence = (
        float(probabilities_tensor[second_index].item()) if second_index is not None else None
    )
    confidence = float(probabilities_tensor[index].item())
    return {
        "label": labels[index],
        "confidence": confidence,
        "probabilities": probabilities,
        "logits": logits,
        "top_3": [
            {
                "label": labels[int(label_index)],
                "probability": float(probabilities_tensor[int(label_index)].item()),
            }
            for label_index in sorted_indices[: min(3, len(labels))]
        ],
        "uncalibrated_confidence": float(uncalibrated_probabilities_tensor[index].item()),
        "uncalibrated_probabilities": uncalibrated_probabilities,
        "calibration_temperature": float(temperature),
        "calibration_applied": True,
        "second_class": labels[second_index] if second_index is not None else None,
        "second_confidence": second_confidence,
        "top1_top2_margin": (
            confidence - second_confidence if second_confidence is not None else confidence
        ),
    }


def repo_path(path: str | Path, project_root: str | Path) -> str:
    """Return a POSIX path relative to the repository when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
