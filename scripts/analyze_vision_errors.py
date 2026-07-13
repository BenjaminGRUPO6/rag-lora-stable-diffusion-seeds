from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from scripts.reconcile_vision_results import (
    sha256_file,
    validate_class_contract,
)
from src.vision.dataset import EXPECTED_CLASSES, OrderedImageFolder, build_transforms
from src.vision.evaluation import compute_metrics, image_paths, load_checkpoint
from src.vision.model import create_model
from src.vision.preprocessing import PreprocessingConfig, preprocess_image
from src.vision.train import resolve_device


DEFAULT_CONFIG = Path("configs/vision_config.yaml")
DEFAULT_CHECKPOINT = Path("models/vision/resnet18_baseline_best.pt")
DEFAULT_MANIFEST = Path("data/metadata/dataset_split.csv")
DEFAULT_OUTPUT_DIR = Path("results/vision/resultados_2_mejoras/04_analisis_errores")
DEFAULT_REVIEW_CSV = Path("data/metadata/vision_error_review.csv")
EXPECTED_VALIDATION_SAMPLES = 522
TARGET_CLASSES = ("intact", "broken")
REVIEW_COLUMNS = (
    "image_id",
    "true_label",
    "predicted_label",
    "confidence",
    "second_label",
    "second_probability",
    "quality_status",
    "suspected_label_issue",
    "reviewed_by",
    "notes",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for validation error analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze reconciled baseline errors on the validation split."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--margin-threshold", type=float, default=None)
    parser.add_argument("--gallery-limit", type=int, default=24)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def repo_path(path: Path) -> str:
    """Return a repository-relative POSIX path when possible."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_bool(value: object) -> bool:
    """Parse CSV boolean-like values."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalized_path(path: str | Path) -> str:
    """Normalize a path for stable cross-platform comparisons."""
    return Path(path).as_posix().lower()


def validate_manifest_validation_split(
    manifest_path: Path,
    expected_classes: Sequence[str],
    expected_samples: int = EXPECTED_VALIDATION_SAMPLES,
) -> dict[str, Any]:
    """Validate validation rows and prove they contain no synthetic data."""
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    if len(validation_rows) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} validation rows in {manifest_path}, "
            f"found {len(validation_rows)}."
        )
    synthetic_rows = [
        row for row in validation_rows if parse_bool(row.get("is_synthetic"))
    ]
    if synthetic_rows:
        raise ValueError(
            f"Validation split contains {len(synthetic_rows)} synthetic rows."
        )

    labels = [str(row.get("label", "")) for row in validation_rows]
    support = Counter(labels)
    expected_set = set(expected_classes)
    found_set = set(support)
    if found_set != expected_set:
        raise ValueError(
            f"Unexpected validation classes. Expected {sorted(expected_set)}, "
            f"found {sorted(found_set)}."
        )
    ordered_support = {
        class_name: int(support[class_name]) for class_name in expected_classes
    }
    return {
        "sample_count": len(validation_rows),
        "support": ordered_support,
        "synthetic_count": len(synthetic_rows),
        "processed_paths": sorted(
            normalized_path(row["processed_path"]) for row in validation_rows
        ),
    }


def build_validation_loader(
    config: dict[str, Any],
    class_names: Sequence[str],
) -> DataLoader:
    """Create the deterministic validation dataloader."""
    data_config = config["data"]
    dataset = OrderedImageFolder(
        Path(data_config["root"]) / "validation",
        expected_classes=class_names,
        transform=build_transforms(
            image_size=int(data_config["image_size"]),
            train=False,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def validate_validation_dataset(
    loader: DataLoader,
    manifest_validation: dict[str, Any],
    expected_samples: int,
) -> dict[str, Any]:
    """Validate physical validation files against the split manifest."""
    dataset = loader.dataset
    sample_count = len(dataset)  # type: ignore[arg-type]
    if sample_count != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} physical validation images, found {sample_count}."
        )
    sample_paths = sorted(
        normalized_path(Path(sample[0])) for sample in getattr(dataset, "samples", [])
    )
    if sample_paths != manifest_validation["processed_paths"]:
        raise ValueError("Physical validation files do not match manifest rows.")
    return {
        "deterministic_transforms": True,
        "sample_count": sample_count,
        "transform": str(getattr(dataset, "transform", "")),
    }


def evaluate_with_inference_mode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], list[int], list[int], list[list[float]], dict[str, bool]]:
    """Evaluate a model with eval mode and torch inference mode."""
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


def top_two(
    probabilities: Sequence[float],
    class_names: Sequence[str],
) -> tuple[str, float, str, float, float]:
    """Return top-1 label/probability, top-2 label/probability and their margin."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size != len(class_names):
        raise ValueError("Probability row length must match class names.")
    order = np.argsort(values)[::-1]
    top_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else top_index
    confidence = float(values[top_index])
    second_probability = float(values[second_index])
    return (
        str(class_names[top_index]),
        confidence,
        str(class_names[second_index]),
        second_probability,
        confidence - second_probability,
    )


def anonymized_image_id(index: int, image_path: str | Path) -> str:
    """Build a stable anonymized image identifier."""
    digest = hashlib.sha1(Path(image_path).as_posix().encode("utf-8")).hexdigest()[:8]
    return f"val_{index + 1:04d}_{digest}"


def quality_status(row: dict[str, Any]) -> str:
    """Condense visual quality metrics into a review status string."""
    warnings = str(row.get("quality_warnings", "")).strip()
    used_fallback = bool(row.get("used_fallback", False))
    crop_confidence = float(row.get("crop_confidence", 0.0))
    if used_fallback:
        return "fallback_crop"
    if crop_confidence < 0.70:
        return "low_crop_confidence"
    if warnings:
        return f"warning:{warnings}"
    return "ok"


def error_categories(
    true_label: str,
    predicted_label: str,
    confidence: float,
    margin: float,
    confidence_threshold: float,
    margin_threshold: float,
) -> list[str]:
    """Classify an image into non-mutually-exclusive error review categories."""
    categories: list[str] = []
    if true_label == "intact" and predicted_label != "intact":
        categories.append("true_intact_predicted_other")
    if true_label == "broken" and predicted_label != "broken":
        categories.append("true_broken_predicted_other")
    if predicted_label == "intact" and true_label != "intact":
        categories.append("predicted_intact_true_other")
    if predicted_label == "broken" and true_label != "broken":
        categories.append("predicted_broken_true_other")
    if confidence < confidence_threshold:
        categories.append("low_confidence")
    if margin < margin_threshold:
        categories.append("low_top1_top2_margin")
    return categories


def quality_metrics_for_image(image_path: Path) -> dict[str, Any]:
    """Calculate deterministic visual quality metrics for one image."""
    result = preprocess_image(image_path, PreprocessingConfig(output_size=224))
    quality = result.quality
    return {
        "used_fallback": bool(result.used_fallback),
        "fallback_reason": result.fallback_reason or "",
        "blur_score": float(quality.blur_score),
        "brightness_score": float(quality.brightness_score),
        "contrast_score": float(quality.contrast_score),
        "foreground_ratio": float(quality.foreground_ratio),
        "component_count": int(quality.component_count),
        "crop_confidence": float(quality.crop_confidence),
        "quality_warnings": ";".join(quality.warnings),
    }


def build_analysis_rows(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    paths: Sequence[str],
    class_names: Sequence[str],
    confidence_threshold: float,
    margin_threshold: float,
) -> list[dict[str, Any]]:
    """Build one analysis row per validation image."""
    rows: list[dict[str, Any]] = []
    for index, (true_index, pred_index, row_probabilities) in enumerate(
        zip(y_true, y_pred, probabilities, strict=True)
    ):
        image_path = Path(paths[index])
        predicted_label, confidence, second_label, second_probability, margin = top_two(
            row_probabilities,
            class_names,
        )
        true_label = str(class_names[int(true_index)])
        if predicted_label != str(class_names[int(pred_index)]):
            raise RuntimeError("Top-1 prediction mismatch.")
        quality = quality_metrics_for_image(image_path)
        row: dict[str, Any] = {
            "image_id": anonymized_image_id(index, image_path),
            "image_path": image_path.as_posix(),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "second_label": second_label,
            "second_probability": second_probability,
            "margin_top1_top2": margin,
            "is_correct": true_label == predicted_label,
        }
        row.update(quality)
        row["quality_status"] = quality_status(row)
        categories = error_categories(
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
            margin=margin,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
        row["error_categories"] = ";".join(categories)
        row["needs_human_review"] = bool(categories and not row["is_correct"])
        rows.append(row)
    return rows


def save_predictions_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Save validation predictions and analysis fields."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_confusion_outputs(
    rows: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save confusion matrix and top confusion tables."""
    true_labels = [str(row["true_label"]) for row in rows]
    predicted_labels = [str(row["predicted_label"]) for row in rows]
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(class_names))
    matrix_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
    matrix_frame.to_csv(output_dir / "r2_matriz_confusion_validation.csv")

    confusion_rows: list[dict[str, Any]] = []
    for true_label in class_names:
        total = int(matrix_frame.loc[true_label].sum())
        for predicted_label in class_names:
            if true_label == predicted_label:
                continue
            count = int(matrix_frame.loc[true_label, predicted_label])
            if count == 0:
                continue
            confusion_rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "rate_within_true_label": count / total if total else 0.0,
                }
            )
    top_frame = pd.DataFrame(confusion_rows).sort_values(
        ["count", "true_label", "predicted_label"],
        ascending=[False, True, True],
    )
    top_frame.to_csv(output_dir / "r2_top_confusiones.csv", index=False)
    return matrix_frame, top_frame


def build_quality_error_summary(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Summarize error rates by visual quality status."""
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "errors": 0})
    for row in rows:
        status = str(row["quality_status"])
        grouped[status]["total"] += 1
        if not bool(row["is_correct"]):
            grouped[status]["errors"] += 1
    summary_rows = []
    for status, counts in sorted(grouped.items()):
        total = int(counts["total"])
        errors = int(counts["errors"])
        summary_rows.append(
            {
                "quality_status": status,
                "total": total,
                "errors": errors,
                "error_rate": errors / total if total else 0.0,
            }
        )
    return pd.DataFrame(summary_rows)


def save_review_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Save the human-review CSV without asserting label corrections."""
    review_rows = []
    for row in rows:
        if not row["error_categories"]:
            continue
        review_rows.append(
            {
                "image_id": row["image_id"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "confidence": round(float(row["confidence"]), 6),
                "second_label": row["second_label"],
                "second_probability": round(float(row["second_probability"]), 6),
                "quality_status": row["quality_status"],
                "suspected_label_issue": "pending_human_review",
                "reviewed_by": "",
                "notes": row["error_categories"],
            }
        )
    frame = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def save_error_category_counts(rows: Sequence[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Save counts for every error-review category."""
    counter: Counter[str] = Counter()
    for row in rows:
        for category in str(row["error_categories"]).split(";"):
            if category:
                counter[category] += 1
    frame = pd.DataFrame(
        [{"category": category, "count": count} for category, count in counter.most_common()]
    )
    frame.to_csv(output_path, index=False)
    return frame


def build_recommendations(
    rows: Sequence[dict[str, Any]],
    top_confusions: pd.DataFrame,
    quality_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Create recommendations grounded only in observed validation results."""
    total = len(rows)
    errors = [row for row in rows if not bool(row["is_correct"])]
    low_confidence = [
        row for row in rows if "low_confidence" in str(row["error_categories"]).split(";")
    ]
    low_margin = [
        row for row in rows if "low_top1_top2_margin" in str(row["error_categories"]).split(";")
    ]
    category_counts = Counter()
    for row in rows:
        for category in str(row["error_categories"]).split(";"):
            if category:
                category_counts[category] += 1

    recommendations: list[dict[str, str]] = []
    if errors:
        recommendations.append(
            {
                "priority": "high",
                "basis": f"{len(errors)} validation errors observed out of {total} images.",
                "recommendation": "Prioritize human review of validation errors before changing labels or model settings.",
            }
        )
    if category_counts.get("true_intact_predicted_other", 0) or category_counts.get(
        "predicted_intact_true_other", 0
    ):
        recommendations.append(
            {
                "priority": "high",
                "basis": (
                    f"Intact-related review categories total "
                    f"{category_counts.get('true_intact_predicted_other', 0) + category_counts.get('predicted_intact_true_other', 0)} cases."
                ),
                "recommendation": "Review intact boundary cases as visual ambiguities, not automatic label errors.",
            }
        )
    if category_counts.get("true_broken_predicted_other", 0) or category_counts.get(
        "predicted_broken_true_other", 0
    ):
        recommendations.append(
            {
                "priority": "high",
                "basis": (
                    f"Broken-related review categories total "
                    f"{category_counts.get('true_broken_predicted_other', 0) + category_counts.get('predicted_broken_true_other', 0)} cases."
                ),
                "recommendation": "Review broken boundary cases and document visual criteria before any dataset edit.",
            }
        )
    if low_confidence or low_margin:
        recommendations.append(
            {
                "priority": "medium",
                "basis": (
                    f"{len(low_confidence)} low-confidence cases and "
                    f"{len(low_margin)} low-margin cases on validation."
                ),
                "recommendation": "Use low-confidence and low-margin cases as a review queue for decision-boundary analysis.",
            }
        )
    if not quality_summary.empty:
        worst = quality_summary.sort_values(["error_rate", "errors"], ascending=False).iloc[0]
        if int(worst["errors"]) > 0:
            recommendations.append(
                {
                    "priority": "medium",
                    "basis": (
                        f"Quality status '{worst['quality_status']}' has "
                        f"{int(worst['errors'])}/{int(worst['total'])} errors."
                    ),
                    "recommendation": "Inspect whether visual quality warnings explain errors before changing preprocessing.",
                }
            )

    top_confusion_records = top_confusions.head(5).to_dict(orient="records")
    return {
        "scope": "validation_only",
        "label_change_policy": "No label is declared incorrect; all suspected issues require human review.",
        "recommendations": recommendations,
        "observed_top_confusions": top_confusion_records,
    }


def _save_figure(figure: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_confusions_intact_broken(
    matrix_frame: pd.DataFrame,
    top_confusions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot validation confusion matrix and top intact/broken confusions."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    matrix = matrix_frame.to_numpy()
    image = axes[0].imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Validation confusion matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_xticks(range(len(matrix_frame.columns)), matrix_frame.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(matrix_frame.index)), matrix_frame.index)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axes[0].text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    target_confusions = top_confusions[
        top_confusions["true_label"].isin(TARGET_CLASSES)
        | top_confusions["predicted_label"].isin(TARGET_CLASSES)
    ].head(8)
    if target_confusions.empty:
        axes[1].text(0.5, 0.5, "No intact/broken confusions", ha="center", va="center")
        axes[1].axis("off")
    else:
        labels = [
            f"{row.true_label} -> {row.predicted_label}"
            for row in target_confusions.itertuples(index=False)
        ]
        counts = [int(row.count) for row in target_confusions.itertuples(index=False)]
        axes[1].barh(labels[::-1], counts[::-1], color="#1f77b4")
        axes[1].set_title("Top intact/broken confusions")
        axes[1].set_xlabel("Count")
    _save_figure(figure, output_path)


def plot_confidence_and_margin(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Plot confidence and top-1/top-2 margin for correct vs error cases."""
    correct = [row for row in rows if bool(row["is_correct"])]
    errors = [row for row in rows if not bool(row["is_correct"])]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(
        [float(row["confidence"]) for row in correct],
        bins=18,
        alpha=0.70,
        label="correct",
        color="#2ca02c",
    )
    axes[0].hist(
        [float(row["confidence"]) for row in errors],
        bins=18,
        alpha=0.70,
        label="errors",
        color="#d62728",
    )
    axes[0].set_title("Confidence distribution")
    axes[0].set_xlabel("Top-1 confidence")
    axes[0].set_ylabel("Images")
    axes[0].set_xlim(0, 1)
    axes[0].legend()

    axes[1].hist(
        [float(row["margin_top1_top2"]) for row in correct],
        bins=18,
        alpha=0.70,
        label="correct",
        color="#2ca02c",
    )
    axes[1].hist(
        [float(row["margin_top1_top2"]) for row in errors],
        bins=18,
        alpha=0.70,
        label="errors",
        color="#d62728",
    )
    axes[1].set_title("Top-1 minus top-2 margin")
    axes[1].set_xlabel("Margin")
    axes[1].set_ylabel("Images")
    axes[1].set_xlim(0, 1)
    axes[1].legend()
    _save_figure(figure, output_path)


def plot_quality_vs_error(quality_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot error rate by visual quality status."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    if quality_summary.empty:
        axis.text(0.5, 0.5, "No quality rows", ha="center", va="center")
        axis.axis("off")
    else:
        frame = quality_summary.sort_values("error_rate", ascending=False)
        axis.bar(
            frame["quality_status"],
            frame["error_rate"],
            color="#9467bd",
        )
        axis.set_ylim(0, min(1.0, max(0.1, float(frame["error_rate"].max()) + 0.05)))
        axis.set_title("Validation error rate by visual quality")
        axis.set_xlabel("Quality status")
        axis.set_ylabel("Error rate")
        axis.tick_params(axis="x", rotation=35)
        for index, row in enumerate(frame.itertuples(index=False)):
            axis.text(
                index,
                float(row.error_rate) + 0.01,
                f"{int(row.errors)}/{int(row.total)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    _save_figure(figure, output_path)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _wrapped_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def save_error_gallery(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    limit: int,
) -> None:
    """Create a composite PNG gallery for selected validation cases."""
    selected = sorted(
        rows,
        key=lambda row: (bool(row["is_correct"]), -float(row["confidence"]), row["image_id"]),
    )[:limit]
    cell_width = 310
    cell_height = 365
    columns = 4
    rows_count = max(1, int(np.ceil(len(selected) / columns)))
    header_height = 44
    canvas = Image.new("RGB", (columns * cell_width, header_height + rows_count * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    text_font = _load_font(13)
    small_font = _load_font(11)
    draw.text((12, 10), title, fill="black", font=title_font)
    if not selected:
        draw.text((12, header_height + 20), "No cases", fill="black", font=text_font)
    for index, row in enumerate(selected):
        column = index % columns
        row_number = index // columns
        left = column * cell_width
        top = header_height + row_number * cell_height
        draw.rectangle((left, top, left + cell_width - 1, top + cell_height - 1), outline="#dddddd")
        image_box = (left + 14, top + 12, left + 296, top + 230)
        try:
            with Image.open(Path(str(row["image_path"]))) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail(
                    (image_box[2] - image_box[0], image_box[3] - image_box[1]),
                    Image.Resampling.LANCZOS,
                )
                image_left = image_box[0] + ((image_box[2] - image_box[0]) - thumbnail.width) // 2
                image_top = image_box[1] + ((image_box[3] - image_box[1]) - thumbnail.height) // 2
                canvas.paste(thumbnail, (image_left, image_top))
        except OSError:
            draw.text((image_box[0], image_box[1]), "Image unavailable", fill="#b00020", font=text_font)

        text_lines = [
            f"ID: {row['image_id']}",
            f"True: {row['true_label']} | Pred: {row['predicted_label']}",
            f"Conf: {float(row['confidence']):.3f} | 2nd: {row['second_label']} {float(row['second_probability']):.3f}",
            f"Quality: {row['quality_status']}",
        ]
        text_y = top + 240
        for line in text_lines:
            for wrapped in _wrapped_text(draw, line, text_font, cell_width - 26):
                draw.text((left + 14, text_y), wrapped, fill="black", font=text_font)
                text_y += 18
        categories = str(row["error_categories"]).replace(";", ", ")
        for wrapped in _wrapped_text(draw, categories, small_font, cell_width - 26)[:2]:
            draw.text((left + 14, text_y + 3), wrapped, fill="#555555", font=small_font)
            text_y += 15
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_plots_and_galleries(
    rows: Sequence[dict[str, Any]],
    matrix_frame: pd.DataFrame,
    top_confusions: pd.DataFrame,
    quality_summary: pd.DataFrame,
    output_dir: Path,
    gallery_limit: int,
) -> list[Path]:
    """Generate requested PNG artifacts."""
    intact_rows = [
        row
        for row in rows
        if row["true_label"] == "intact" and row["predicted_label"] != "intact"
        or row["predicted_label"] == "intact" and row["true_label"] != "intact"
    ]
    broken_rows = [
        row
        for row in rows
        if row["true_label"] == "broken" and row["predicted_label"] != "broken"
        or row["predicted_label"] == "broken" and row["true_label"] != "broken"
    ]
    png_paths = [
        output_dir / "r2_errores_intact.png",
        output_dir / "r2_errores_broken.png",
        output_dir / "r2_confusiones_intact_broken.png",
        output_dir / "r2_confianza_correctos_vs_errores.png",
        output_dir / "r2_calidad_vs_error.png",
    ]
    save_error_gallery(
        intact_rows,
        png_paths[0],
        title="Validation intact-related errors",
        limit=gallery_limit,
    )
    save_error_gallery(
        broken_rows,
        png_paths[1],
        title="Validation broken-related errors",
        limit=gallery_limit,
    )
    plot_confusions_intact_broken(matrix_frame, top_confusions, png_paths[2])
    plot_confidence_and_margin(rows, png_paths[3])
    plot_quality_vs_error(quality_summary, png_paths[4])
    return png_paths


def summary_payload(
    *,
    rows: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    class_names: Sequence[str],
    confidence_threshold: float,
    margin_threshold: float,
    matrix_frame: pd.DataFrame,
    top_confusions: pd.DataFrame,
    quality_summary: pd.DataFrame,
    png_paths: Sequence[Path],
    generated_at_utc: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    validation_manifest: dict[str, Any],
    evaluation_flags: dict[str, bool],
) -> dict[str, Any]:
    """Build the machine-readable analysis summary."""
    category_counts: Counter[str] = Counter()
    for row in rows:
        for category in str(row["error_categories"]).split(";"):
            if category:
                category_counts[category] += 1
    return {
        "experiment_id": "04_analisis_errores",
        "split": "validation",
        "generated_at_utc": generated_at_utc,
        "checkpoint": repo_path(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "classes": list(class_names),
        "thresholds": {
            "confidence": confidence_threshold,
            "margin_top1_top2": margin_threshold,
        },
        "validation": {
            "sample_count": validation_manifest["sample_count"],
            "support": validation_manifest["support"],
            "synthetic_count": validation_manifest["synthetic_count"],
        },
        "metrics": metrics,
        "evaluation": evaluation_flags,
        "error_counts": {
            "total_errors": sum(1 for row in rows if not bool(row["is_correct"])),
            "by_category": dict(category_counts),
        },
        "confusion_matrix": matrix_frame.to_dict(),
        "top_confusions": top_confusions.head(10).to_dict(orient="records"),
        "quality_vs_error": quality_summary.to_dict(orient="records"),
        "png": [repo_path(path) for path in png_paths],
        "policy_notes": [
            "Validation split only; test is not used for development decisions.",
            "No labels are modified or declared incorrect by this script.",
            "spotted is treated only as a visual category.",
        ],
    }


def run_analysis(
    *,
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    review_csv_path: Path = DEFAULT_REVIEW_CSV,
    device_name: str | None = None,
    confidence_threshold: float | None = None,
    margin_threshold: float | None = None,
    gallery_limit: int = 24,
) -> dict[str, Any]:
    """Run the validation-only error analysis and write all artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    class_names = list(config.get("classes", EXPECTED_CLASSES))
    device = resolve_device(device_name)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    checkpoint_mapping = {
        str(class_name): int(index)
        for class_name, index in dict(checkpoint.get("class_to_idx", {})).items()
    }
    validate_class_contract(class_names, checkpoint_mapping)
    validation_manifest = validate_manifest_validation_split(
        manifest_path=manifest_path,
        expected_classes=class_names,
        expected_samples=EXPECTED_VALIDATION_SAMPLES,
    )
    loader = build_validation_loader(config, class_names)
    dataset_validation = validate_validation_dataset(
        loader,
        manifest_validation=validation_manifest,
        expected_samples=EXPECTED_VALIDATION_SAMPLES,
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
    paths = image_paths(loader.dataset)
    confidence_cutoff = float(
        confidence_threshold
        if confidence_threshold is not None
        else config.get("inference", {}).get("confidence_threshold", 0.60)
    )
    margin_cutoff = float(
        margin_threshold
        if margin_threshold is not None
        else config.get("inference", {}).get("margin_threshold", 0.15)
    )
    rows = build_analysis_rows(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        paths=paths,
        class_names=class_names,
        confidence_threshold=confidence_cutoff,
        margin_threshold=margin_cutoff,
    )
    predictions_path = output_dir / "r2_predicciones_validation.csv"
    save_predictions_csv(rows, predictions_path)
    matrix_frame, top_confusions = save_confusion_outputs(rows, class_names, output_dir)
    quality_summary = build_quality_error_summary(rows)
    quality_summary.to_csv(output_dir / "r2_errores_por_calidad.csv", index=False)
    save_error_category_counts(rows, output_dir / "r2_categorias_error.csv")
    review_frame = save_review_csv(rows, review_csv_path)
    png_paths = save_plots_and_galleries(
        rows=rows,
        matrix_frame=matrix_frame,
        top_confusions=top_confusions,
        quality_summary=quality_summary,
        output_dir=output_dir,
        gallery_limit=gallery_limit,
    )
    recommendations = build_recommendations(rows, top_confusions, quality_summary)
    write_json(output_dir / "recommendations.json", recommendations)
    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary = summary_payload(
        rows=rows,
        metrics=metrics,
        class_names=class_names,
        confidence_threshold=confidence_cutoff,
        margin_threshold=margin_cutoff,
        matrix_frame=matrix_frame,
        top_confusions=top_confusions,
        quality_summary=quality_summary,
        png_paths=png_paths,
        generated_at_utc=generated_at_utc,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        validation_manifest=validation_manifest,
        evaluation_flags={**evaluation_flags, **dataset_validation},
    )
    summary["outputs"] = {
        "predictions": repo_path(predictions_path),
        "review_csv": repo_path(review_csv_path),
        "recommendations": repo_path(output_dir / "recommendations.json"),
        "summary": repo_path(output_dir / "r2_analisis_errores_resumen.json"),
    }
    summary["review_rows"] = int(len(review_frame))
    write_json(output_dir / "r2_analisis_errores_resumen.json", summary)
    return summary


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    summary = run_analysis(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        review_csv_path=args.review_csv,
        device_name=args.device,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
        gallery_limit=int(args.gallery_limit),
    )
    print(
        yaml.safe_dump(
            {
                "validation_samples": summary["validation"]["sample_count"],
                "accuracy": summary["metrics"]["accuracy"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "total_errors": summary["error_counts"]["total_errors"],
                "error_categories": summary["error_counts"]["by_category"],
                "top_confusions": summary["top_confusions"][:5],
                "review_rows": summary["review_rows"],
                "summary": summary["outputs"]["summary"],
                "png": summary["png"],
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
