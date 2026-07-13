"""Compare direct CLI inference with the Streamlit pipeline inference path."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.analyze_seed import (
    DEFAULT_RAG_CONFIG,
    DEFAULT_VISION_CONFIG,
    analyze_seed,
    default_checkpoint_path,
    load_yaml_config,
    resolve_device,
)
from src.vision.inference import IMAGENET_MEAN, IMAGENET_STD, VisionInferenceEngine


EXPERIMENT_ID = "02_paridad_inferencia"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "vision"
    / "resultados_2_mejoras"
    / EXPERIMENT_ID
)
CLASS_ORDER = ("intact", "spotted", "immature", "broken", "skin_damaged")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_SEED = 42
DEFAULT_IMAGES_PER_CLASS = 5
PROBABILITY_ATOL = 1e-6
LOGIT_ATOL = 1e-6


@dataclass(frozen=True)
class SelectedImage:
    """One validation image selected for parity comparison."""

    image_id: str
    true_class: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the parity comparison."""
    parser = argparse.ArgumentParser(description="Compare vision inference paths.")
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_VISION_CONFIG)
    parser.add_argument("--rag-config", type=Path, default=DEFAULT_RAG_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--validation-dir", type=Path, default=Path("data/processed/validation"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--images-per-class", type=int, default=DEFAULT_IMAGES_PER_CLASS)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    """Run the comparison and persist parity artifacts."""
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    validation_dir = resolve_project_path(args.validation_dir)
    vision_config_path = resolve_project_path(args.vision_config)
    rag_config_path = resolve_project_path(args.rag_config)
    vision_config = load_yaml_config(vision_config_path)
    checkpoint = resolve_project_path(args.checkpoint or default_checkpoint_path(vision_config))
    device = resolve_device(args.device)
    engine = VisionInferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint,
        device=device,
        config=vision_config,
    )

    selected = select_validation_images(
        validation_dir=validation_dir,
        class_names=tuple(engine.labels),
        images_per_class=int(args.images_per_class),
        seed=int(args.seed),
    )
    rows = compare_paths(
        selected_images=selected,
        engine=engine,
        vision_config_path=vision_config_path,
        rag_config_path=rag_config_path,
        labels=engine.labels,
        device_name=str(device),
    )
    artifacts = write_outputs(
        rows=rows,
        selected_images=selected,
        output_dir=output_dir,
        labels=engine.labels,
        seed=int(args.seed),
        images_per_class=int(args.images_per_class),
        validation_dir=validation_dir,
        checkpoint=checkpoint,
        vision_config_path=vision_config_path,
    )
    summary = json.loads((output_dir / "r2_paridad_resumen.json").read_text(encoding="utf-8"))
    print(f"Clase coincide: {summary['class_match_count']}/{summary['image_count']}")
    print(f"Diferencia media probabilidades: {summary['mean_probability_abs_diff']:.12f}")
    print(f"Diferencia maxima probabilidades: {summary['max_probability_abs_diff']:.12f}")
    print(f"Artefactos: {', '.join(artifacts)}")
    return 0 if summary["parity_passed"] else 1


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
    class_names: tuple[str, ...] = CLASS_ORDER,
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> list[SelectedImage]:
    """Select N images per class from validation using a reproducible seed."""
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


def compare_paths(
    *,
    selected_images: list[SelectedImage],
    engine: VisionInferenceEngine,
    vision_config_path: Path,
    rag_config_path: Path,
    labels: list[str],
    device_name: str,
) -> list[dict[str, Any]]:
    """Run every image through direct engine and Streamlit pipeline paths."""
    rows: list[dict[str, Any]] = []
    for index, selected in enumerate(selected_images, start=1):
        cli_prediction = engine.predict_dict(selected.image_path)
        streamlit_result = analyze_seed(
            image=selected.image_path,
            vision_config_path=vision_config_path,
            rag_config_path=rag_config_path,
            inference_engine=engine,
            retriever=lambda query, top_k=None: [],
            top_k=0,
            device_name=device_name,
        )
        streamlit_prediction = {
            "label": streamlit_result["prediction"],
            "confidence": streamlit_result["confidence"],
            "probabilities": streamlit_result["probabilities"],
            "logits": streamlit_result["logits"],
            "top_3": streamlit_result["top_3"],
        }
        probability_diffs = {
            label: abs(
                float(cli_prediction["probabilities"][label])
                - float(streamlit_prediction["probabilities"][label])
            )
            for label in labels
        }
        logit_diffs = {
            label: abs(
                float(cli_prediction["logits"][label])
                - float(streamlit_prediction["logits"][label])
            )
            for label in labels
        }
        rows.append(
            {
                "case_id": f"parity_{index:02d}",
                "image_id": selected.image_id,
                "image_path": repo_path(selected.image_path),
                "true_class": selected.true_class,
                "cli_label": cli_prediction["label"],
                "streamlit_label": streamlit_prediction["label"],
                "class_match": cli_prediction["label"] == streamlit_prediction["label"],
                "cli_confidence": float(cli_prediction["confidence"]),
                "streamlit_confidence": float(streamlit_prediction["confidence"]),
                "confidence_abs_diff": abs(
                    float(cli_prediction["confidence"]) - float(streamlit_prediction["confidence"])
                ),
                "max_probability_abs_diff": max(probability_diffs.values()),
                "mean_probability_abs_diff": statistics.fmean(probability_diffs.values()),
                "max_logit_abs_diff": max(logit_diffs.values()),
                "cli_probabilities": cli_prediction["probabilities"],
                "streamlit_probabilities": streamlit_prediction["probabilities"],
                "probability_abs_diffs": probability_diffs,
                "cli_logits": cli_prediction["logits"],
                "streamlit_logits": streamlit_prediction["logits"],
                "logit_abs_diffs": logit_diffs,
                "cli_top_3": cli_prediction["top_3"],
                "streamlit_top_3": streamlit_prediction["top_3"],
                "top_3_match": cli_prediction["top_3"] == streamlit_prediction["top_3"],
            }
        )
    return rows


def write_outputs(
    *,
    rows: list[dict[str, Any]],
    selected_images: list[SelectedImage],
    output_dir: Path,
    labels: list[str],
    seed: int,
    images_per_class: int,
    validation_dir: Path,
    checkpoint: Path,
    vision_config_path: Path,
) -> list[str]:
    """Write CSV, summary JSON and PNG plots for the parity experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "r2_paridad_inferencia.csv"
    summary_path = output_dir / "r2_paridad_resumen.json"
    scatter_path = output_dir / "r2_cli_vs_streamlit_probabilidades.png"
    diff_path = output_dir / "r2_diferencia_probabilidades.png"
    class_path = output_dir / "r2_paridad_por_clase.png"

    write_csv(csv_path, rows)
    summary = build_summary(
        rows=rows,
        selected_images=selected_images,
        labels=labels,
        seed=seed,
        images_per_class=images_per_class,
        validation_dir=validation_dir,
        checkpoint=checkpoint,
        vision_config_path=vision_config_path,
        artifacts=[csv_path, summary_path, scatter_path, diff_path, class_path],
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_cli_vs_streamlit_probabilities(rows, labels, scatter_path)
    plot_probability_differences(rows, labels, diff_path)
    plot_class_parity(rows, labels, class_path)
    return [repo_path(path) for path in (csv_path, summary_path, scatter_path, diff_path, class_path)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write parity rows to CSV with JSON-encoded nested values."""
    fieldnames = [
        "case_id",
        "image_id",
        "image_path",
        "true_class",
        "cli_label",
        "streamlit_label",
        "class_match",
        "cli_confidence",
        "streamlit_confidence",
        "confidence_abs_diff",
        "max_probability_abs_diff",
        "mean_probability_abs_diff",
        "max_logit_abs_diff",
        "top_3_match",
        "cli_top_3",
        "streamlit_top_3",
        "cli_probabilities",
        "streamlit_probabilities",
        "probability_abs_diffs",
        "cli_logits",
        "streamlit_logits",
        "logit_abs_diffs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def csv_value(value: Any) -> str:
    """Format values for CSV output."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.12f}"
    return "" if value is None else str(value)


def build_summary(
    *,
    rows: list[dict[str, Any]],
    selected_images: list[SelectedImage],
    labels: list[str],
    seed: int,
    images_per_class: int,
    validation_dir: Path,
    checkpoint: Path,
    vision_config_path: Path,
    artifacts: list[Path],
) -> dict[str, Any]:
    """Build a machine-readable summary of parity criteria and observed diffs."""
    probability_diffs = [
        float(diff)
        for row in rows
        for diff in dict(row["probability_abs_diffs"]).values()
    ]
    logit_diffs = [
        float(diff)
        for row in rows
        for diff in dict(row["logit_abs_diffs"]).values()
    ]
    class_match_count = sum(1 for row in rows if row["class_match"])
    top_3_match_count = sum(1 for row in rows if row["top_3_match"])
    max_probability_abs_diff = max(probability_diffs) if probability_diffs else 0.0
    max_logit_abs_diff = max(logit_diffs) if logit_diffs else 0.0
    per_class = {}
    for label in labels:
        class_rows = [row for row in rows if row["true_class"] == label]
        class_probability_diffs = [
            float(diff)
            for row in class_rows
            for diff in dict(row["probability_abs_diffs"]).values()
        ]
        per_class[label] = {
            "image_count": len(class_rows),
            "class_match_count": sum(1 for row in class_rows if row["class_match"]),
            "class_match_rate": (
                sum(1 for row in class_rows if row["class_match"]) / len(class_rows)
                if class_rows
                else 0.0
            ),
            "max_probability_abs_diff": (
                max(class_probability_diffs) if class_probability_diffs else 0.0
            ),
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": seed,
        "split": "validation",
        "validation_dir": repo_path(validation_dir),
        "images_per_class": images_per_class,
        "image_count": len(selected_images),
        "class_order": labels,
        "class_match_count": class_match_count,
        "class_match_rate": class_match_count / len(rows) if rows else 0.0,
        "top_3_match_count": top_3_match_count,
        "top_3_match_rate": top_3_match_count / len(rows) if rows else 0.0,
        "mean_probability_abs_diff": statistics.fmean(probability_diffs)
        if probability_diffs
        else 0.0,
        "max_probability_abs_diff": max_probability_abs_diff,
        "mean_logit_abs_diff": statistics.fmean(logit_diffs) if logit_diffs else 0.0,
        "max_logit_abs_diff": max_logit_abs_diff,
        "probability_tolerance": PROBABILITY_ATOL,
        "logit_tolerance": LOGIT_ATOL,
        "parity_passed": (
            class_match_count == len(rows)
            and max_probability_abs_diff <= PROBABILITY_ATOL
            and max_logit_abs_diff <= LOGIT_ATOL
        ),
        "per_class": per_class,
        "inference_contract": {
            "engine": "src.vision.inference_engine.VisionInferenceEngine",
            "checkpoint": repo_path(checkpoint),
            "vision_config": repo_path(vision_config_path),
            "rgb_conversion": True,
            "resize": [int(load_yaml_config(vision_config_path).get("data", {}).get("image_size", 224))] * 2,
            "crop": None,
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "softmax": "torch.softmax(logits, dim=0)",
            "probability_tolerance_note": (
                "Se exige igualdad de clase en 100% y diferencias absolutas de "
                f"probabilidad <= {PROBABILITY_ATOL:g}; se registra la diferencia maxima."
            ),
        },
        "selected_images": [asdict(selected) | {"image_path": repo_path(selected.image_path)} for selected in selected_images],
        "artifacts": [repo_path(path) for path in artifacts],
    }


def plot_cli_vs_streamlit_probabilities(rows: list[dict[str, Any]], labels: list[str], path: Path) -> None:
    """Plot all CLI probabilities against Streamlit pipeline probabilities."""
    cli_values = [
        float(row["cli_probabilities"][label])
        for row in rows
        for label in labels
    ]
    streamlit_values = [
        float(row["streamlit_probabilities"][label])
        for row in rows
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(6.5, 6.0), facecolor="white")
    ax.scatter(cli_values, streamlit_values, s=22, alpha=0.75, color="#2563eb")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#111827", linewidth=1.0)
    ax.set_title("02_paridad_inferencia - CLI vs Streamlit probabilidades")
    ax.set_xlabel("Probabilidad motor CLI")
    ax.set_ylabel("Probabilidad pipeline Streamlit")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_probability_differences(rows: list[dict[str, Any]], labels: list[str], path: Path) -> None:
    """Plot maximum probability difference for each evaluated image."""
    diffs = [float(row["max_probability_abs_diff"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.0, 4.8), facecolor="white")
    ax.bar(range(1, len(rows) + 1), diffs, color="#0f766e")
    ax.axhline(PROBABILITY_ATOL, color="#b91c1c", linewidth=1.2, label="tolerancia")
    ax.set_title("02_paridad_inferencia - diferencia de probabilidades")
    ax.set_xlabel("Imagen evaluada")
    ax.set_ylabel("Diferencia absoluta maxima")
    ax.set_xticks(range(1, len(rows) + 1))
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_class_parity(rows: list[dict[str, Any]], labels: list[str], path: Path) -> None:
    """Plot class prediction parity rate by true class."""
    rates = []
    for label in labels:
        class_rows = [row for row in rows if row["true_class"] == label]
        rates.append(
            100.0 * sum(1 for row in class_rows if row["class_match"]) / len(class_rows)
            if class_rows
            else 0.0
        )
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
    ax.bar(labels, rates, color="#7c3aed")
    ax.set_title("02_paridad_inferencia - paridad por clase")
    ax.set_xlabel("Clase visual")
    ax.set_ylabel("Coincidencia de clase (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
