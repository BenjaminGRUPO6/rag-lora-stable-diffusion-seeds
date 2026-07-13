"""Evaluate deterministic preprocessing and visual quality on validation samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.dataset import EXPECTED_CLASSES
from src.vision.preprocessing import PreprocessingConfig, PreprocessingResult, preprocess_image


EXPERIMENT_ID = "03_recorte_y_calidad"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "vision"
    / "resultados_2_mejoras"
    / EXPERIMENT_ID
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_IMAGES_PER_CLASS = 20
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SelectedImage:
    """One validation image selected for preprocessing review."""

    image_id: str
    true_class: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate automatic crop and visual quality.")
    parser.add_argument("--validation-dir", type=Path, default=Path("data/processed/validation"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images-per-class", type=int, default=DEFAULT_IMAGES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    """Run preprocessing evaluation and write review artifacts."""
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    validation_dir = resolve_project_path(args.validation_dir)
    selected = select_validation_images(
        validation_dir=validation_dir,
        class_names=EXPECTED_CLASSES,
        images_per_class=int(args.images_per_class),
        seed=int(args.seed),
    )
    config = PreprocessingConfig(output_size=int(args.image_size))
    evaluated = evaluate_images(selected, config)
    artifacts = write_outputs(
        selected=evaluated,
        output_dir=output_dir,
        validation_dir=validation_dir,
        seed=int(args.seed),
        images_per_class=int(args.images_per_class),
        config=config,
    )
    summary = json.loads((output_dir / "r2_recorte_y_calidad_resumen.json").read_text(encoding="utf-8"))
    print(f"Crops evaluados: {summary['crops_evaluated']}")
    print(f"Fallback: {summary['fallback_count']}")
    print(f"Calidad media: {summary['mean_quality']}")
    print(f"Warnings: {summary['warning_counts']}")
    print(f"PNG: {', '.join(summary['png_artifacts'])}")
    print(f"Artefactos: {', '.join(artifacts)}")
    return 0


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def repo_path(path: Path) -> str:
    """Return a repository-relative POSIX path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def select_validation_images(
    *,
    validation_dir: Path,
    class_names: tuple[str, ...],
    images_per_class: int,
    seed: int,
) -> list[SelectedImage]:
    """Select a reproducible stratified validation sample."""
    rng = random.Random(seed)
    selected: list[SelectedImage] = []
    for class_name in class_names:
        class_dir = validation_dir / class_name
        candidates = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(candidates) < images_per_class:
            raise ValueError(
                f"No hay suficientes imagenes de validation para {class_name}: "
                f"{len(candidates)} disponibles, {images_per_class} requeridas."
            )
        for image_path in sorted(rng.sample(candidates, images_per_class)):
            selected.append(
                SelectedImage(
                    image_id=image_path.stem,
                    true_class=class_name,
                    image_path=image_path,
                )
            )
    return selected


def evaluate_images(
    selected: list[SelectedImage],
    config: PreprocessingConfig,
) -> list[tuple[SelectedImage, PreprocessingResult]]:
    """Run preprocessing for each selected image."""
    results: list[tuple[SelectedImage, PreprocessingResult]] = []
    for item in selected:
        results.append((item, preprocess_image(item.image_path, config=config)))
    return results


def write_outputs(
    *,
    selected: list[tuple[SelectedImage, PreprocessingResult]],
    output_dir: Path,
    validation_dir: Path,
    seed: int,
    images_per_class: int,
    config: PreprocessingConfig,
) -> list[str]:
    """Write CSV, JSON, review template and PNG panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "r2_recorte_y_calidad.csv"
    review_csv = output_dir / "r2_revision_humana_template.csv"
    summary_json = output_dir / "r2_recorte_y_calidad_resumen.json"
    png_paths = [
        output_dir / "r2_galeria_recortes.png",
        output_dir / "r2_original_vs_recorte.png",
        output_dir / "r2_panel_calidad.png",
        output_dir / "r2_distribucion_calidad.png",
        output_dir / "r2_casos_fallback.png",
    ]

    rows = build_metric_rows(selected)
    write_metrics_csv(metrics_csv, rows)
    write_review_template(review_csv, rows)
    plot_crop_gallery(selected, png_paths[0])
    plot_original_vs_crop(selected, png_paths[1])
    plot_quality_panel(rows, png_paths[2])
    plot_quality_distribution(rows, png_paths[3])
    plot_fallback_cases(selected, png_paths[4])
    summary = build_summary(
        rows=rows,
        validation_dir=validation_dir,
        seed=seed,
        images_per_class=images_per_class,
        config=config,
        artifacts=[metrics_csv, review_csv, summary_json, *png_paths],
        png_artifacts=png_paths,
    )
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [repo_path(path) for path in [metrics_csv, review_csv, summary_json, *png_paths]]


def build_metric_rows(
    selected: list[tuple[SelectedImage, PreprocessingResult]],
) -> list[dict[str, Any]]:
    """Build flat metric rows for CSV and plots."""
    rows: list[dict[str, Any]] = []
    for index, (item, result) in enumerate(selected, start=1):
        quality = result.quality
        rows.append(
            {
                "case_id": f"crop_quality_{index:03d}",
                "image_id": item.image_id,
                "image_path": repo_path(item.image_path),
                "true_class": item.true_class,
                "used_fallback": result.used_fallback,
                "fallback_reason": result.fallback_reason or "",
                "bbox": json.dumps(result.bbox, ensure_ascii=False),
                "component_count": result.component_count,
                "blur_score": quality.blur_score,
                "brightness_score": quality.brightness_score,
                "contrast_score": quality.contrast_score,
                "foreground_ratio": quality.foreground_ratio,
                "crop_confidence": quality.crop_confidence,
                "warnings": ";".join(quality.warnings),
            }
        )
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write preprocessing metrics."""
    fieldnames = [
        "case_id",
        "image_id",
        "image_path",
        "true_class",
        "used_fallback",
        "fallback_reason",
        "bbox",
        "component_count",
        "blur_score",
        "brightness_score",
        "contrast_score",
        "foreground_ratio",
        "crop_confidence",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_review_template(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write human review template with required review fields."""
    fieldnames = [
        "case_id",
        "image_id",
        "image_path",
        "true_class",
        "used_fallback",
        "fallback_reason",
        "crop_acceptable",
        "object_centered",
        "single_seed",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "true_class": row["true_class"],
                    "used_fallback": row["used_fallback"],
                    "fallback_reason": row["fallback_reason"],
                    "crop_acceptable": "",
                    "object_centered": "",
                    "single_seed": "",
                    "notes": "",
                }
            )


def build_summary(
    *,
    rows: list[dict[str, Any]],
    validation_dir: Path,
    seed: int,
    images_per_class: int,
    config: PreprocessingConfig,
    artifacts: list[Path],
    png_artifacts: list[Path],
) -> dict[str, Any]:
    """Build machine-readable evaluation summary."""
    warning_counts = Counter(
        warning
        for row in rows
        for warning in str(row["warnings"]).split(";")
        if warning
    )
    fallback_count = sum(1 for row in rows if bool(row["used_fallback"]))
    quality_keys = [
        "blur_score",
        "brightness_score",
        "contrast_score",
        "foreground_ratio",
        "crop_confidence",
    ]
    mean_quality = {
        key: round(statistics.fmean(float(row[key]) for row in rows), 6) if rows else 0.0
        for key in quality_keys
    }
    per_class = {}
    for class_name in EXPECTED_CLASSES:
        class_rows = [row for row in rows if row["true_class"] == class_name]
        per_class[class_name] = {
            "count": len(class_rows),
            "fallback_count": sum(1 for row in class_rows if bool(row["used_fallback"])),
            "mean_crop_confidence": (
                round(statistics.fmean(float(row["crop_confidence"]) for row in class_rows), 6)
                if class_rows
                else 0.0
            ),
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "split": "validation",
        "validation_dir": repo_path(validation_dir),
        "seed": seed,
        "images_per_class": images_per_class,
        "crops_evaluated": len(rows),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(rows), 6) if rows else 0.0,
        "mean_quality": mean_quality,
        "warning_counts": dict(sorted(warning_counts.items())),
        "per_class": per_class,
        "config": asdict(config),
        "artifacts": [repo_path(path) for path in artifacts],
        "png_artifacts": [repo_path(path) for path in png_artifacts],
        "review_template": "crop_acceptable, object_centered, single_seed, notes",
        "note": "Las heuristicas son control de calidad visual y no constituyen diagnostico.",
    }


def plot_crop_gallery(
    selected: list[tuple[SelectedImage, PreprocessingResult]],
    path: Path,
) -> None:
    """Plot a gallery of automatic crops."""
    examples = balanced_examples(selected, per_class=5)
    fig, axes = plt.subplots(5, 5, figsize=(9, 9), facecolor="white")
    for axis, (item, result) in zip(axes.ravel(), examples):
        axis.imshow(result.crop)
        suffix = " F" if result.used_fallback else ""
        axis.set_title(f"{item.true_class} {item.image_id}{suffix}", fontsize=7)
        axis.axis("off")
    for axis in axes.ravel()[len(examples) :]:
        axis.axis("off")
    fig.suptitle("03_recorte_y_calidad - galeria de recortes", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_original_vs_crop(
    selected: list[tuple[SelectedImage, PreprocessingResult]],
    path: Path,
) -> None:
    """Plot original images next to automatic crops."""
    examples = selected[:12]
    fig, axes = plt.subplots(len(examples), 2, figsize=(6.5, 2.0 * len(examples)), facecolor="white")
    if len(examples) == 1:
        axes = axes.reshape(1, 2)
    for row_index, (item, result) in enumerate(examples):
        axes[row_index, 0].imshow(result.original)
        axes[row_index, 0].set_title(f"Original: {item.true_class}/{item.image_id}", fontsize=8)
        axes[row_index, 1].imshow(result.crop)
        title = "Recorte automatico"
        if result.used_fallback:
            title = f"Fallback: {result.fallback_reason}"
        axes[row_index, 1].set_title(title, fontsize=8)
        for axis in axes[row_index]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_quality_panel(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot mean quality and fallback rate by class."""
    classes = list(EXPECTED_CLASSES)
    confidence = [mean_for_class(rows, class_name, "crop_confidence") for class_name in classes]
    foreground = [mean_for_class(rows, class_name, "foreground_ratio") for class_name in classes]
    fallback = [fallback_rate_for_class(rows, class_name) for class_name in classes]
    x_positions = range(len(classes))
    fig, axis = plt.subplots(figsize=(9, 4.8), facecolor="white")
    axis.bar([x - 0.25 for x in x_positions], confidence, width=0.25, label="confianza crop", color="#2563eb")
    axis.bar(x_positions, foreground, width=0.25, label="foreground ratio", color="#0f766e")
    axis.bar([x + 0.25 for x in x_positions], fallback, width=0.25, label="fallback rate", color="#b45309")
    axis.set_title("03_recorte_y_calidad - panel de calidad")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(classes, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_quality_distribution(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot distributions of visual quality scores."""
    metrics = [
        ("crop_confidence", "Confianza de recorte"),
        ("foreground_ratio", "Foreground ratio"),
        ("brightness_score", "Brillo"),
        ("contrast_score", "Contraste"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), facecolor="white")
    for axis, (key, title) in zip(axes.ravel(), metrics):
        axis.hist([float(row[key]) for row in rows], bins=12, color="#334155", alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel(key)
        axis.set_ylabel("imagenes")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("03_recorte_y_calidad - distribucion de calidad", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_fallback_cases(
    selected: list[tuple[SelectedImage, PreprocessingResult]],
    path: Path,
) -> None:
    """Plot fallback cases or an explicit empty-state panel."""
    fallback_examples = [(item, result) for item, result in selected if result.used_fallback][:8]
    if not fallback_examples:
        fig, axis = plt.subplots(figsize=(6, 3), facecolor="white")
        axis.text(0.5, 0.5, "Sin casos fallback en la muestra", ha="center", va="center", fontsize=13)
        axis.set_title("03_recorte_y_calidad - casos fallback")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(fallback_examples), 2, figsize=(6.5, 2.0 * len(fallback_examples)), facecolor="white")
    if len(fallback_examples) == 1:
        axes = axes.reshape(1, 2)
    for row_index, (item, result) in enumerate(fallback_examples):
        axes[row_index, 0].imshow(result.original)
        axes[row_index, 0].set_title(f"{item.true_class}/{item.image_id}", fontsize=8)
        axes[row_index, 1].imshow(result.crop)
        axes[row_index, 1].set_title(str(result.fallback_reason), fontsize=8)
        for axis in axes[row_index]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def balanced_examples(
    selected: list[tuple[SelectedImage, PreprocessingResult]],
    *,
    per_class: int,
) -> list[tuple[SelectedImage, PreprocessingResult]]:
    """Return a balanced set of examples by class."""
    examples: list[tuple[SelectedImage, PreprocessingResult]] = []
    for class_name in EXPECTED_CLASSES:
        examples.extend([(item, result) for item, result in selected if item.true_class == class_name][:per_class])
    return examples


def mean_for_class(rows: list[dict[str, Any]], class_name: str, key: str) -> float:
    """Return a class mean for a numeric metric."""
    values = [float(row[key]) for row in rows if row["true_class"] == class_name]
    return statistics.fmean(values) if values else 0.0


def fallback_rate_for_class(rows: list[dict[str, Any]], class_name: str) -> float:
    """Return fallback rate for one class."""
    class_rows = [row for row in rows if row["true_class"] == class_name]
    if not class_rows:
        return 0.0
    return sum(1 for row in class_rows if bool(row["used_fallback"])) / len(class_rows)


if __name__ == "__main__":
    raise SystemExit(main())
