from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.analyze_seed import resolve_device
from src.vision.gradcam import GradCamResult, find_last_convolutional_layer, generate_gradcam_with_fallback
from src.vision.inference import VisionInferenceEngine
from src.vision.preprocessing import PreprocessingConfig, preprocess_image
from src.vision.visualization import (
    build_combined_gradcam_image,
    build_image_grid,
    build_probability_panel_image,
    heatmap_to_image,
)


DEFAULT_PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "production_vision_model.yaml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "vision" / "resultados_2_mejoras" / "09_gradcam_interfaz"
)
EXPECTED_CLASSES = ["intact", "spotted", "immature", "broken", "skin_damaged"]


@dataclass(frozen=True)
class EvidenceSample:
    """Metadata for one generated Grad-CAM evidence sample."""

    image_path: str
    true_label: str
    predicted_label: str
    confidence: float
    gradcam_status: str
    gradcam_layer: str
    gradcam_intensity: float
    combined_path: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Grad-CAM evidence PNGs for the app.")
    parser.add_argument("--production-config", type=Path, default=DEFAULT_PRODUCTION_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    """Generate Grad-CAM evidence without training."""
    args = parse_args()
    summary = generate_evidence(
        production_config_path=args.production_config,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def generate_evidence(
    *,
    production_config_path: Path = DEFAULT_PRODUCTION_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Generate the requested Grad-CAM galleries and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    production = load_yaml(production_config_path)
    architecture = str(production.get("architecture") or "resnet18")
    checkpoint = PROJECT_ROOT / Path(str(production.get("checkpoint_path") or ""))
    class_names = [str(item) for item in production.get("class_names") or EXPECTED_CLASSES]
    image_size = int(production.get("image_size") or 224)
    device = resolve_device(device_name)
    engine = VisionInferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint,
        device=device,
        config={
            "model": {"architecture": architecture, "dropout": 0.20},
            "data": {"image_size": image_size},
        },
        temperature_path=production.get("calibration_path"),
    )
    gradcam_layer, _ = find_last_convolutional_layer(engine.model)
    predictions = pd.read_csv(resolve_predictions_path(architecture))
    selected_groups = select_evidence_rows(predictions, class_names)
    generated: dict[str, list[EvidenceSample]] = {}
    individual_dir = output_dir / "samples"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for group_name, rows in selected_groups.items():
        generated[group_name] = [
            render_sample(
                row=row,
                engine=engine,
                output_dir=individual_dir,
                image_size=image_size,
            )
            for row in rows
        ]

    build_image_grid(
        [sample_to_item(sample) for sample in generated["correctos"]],
        output_dir / "r2_gradcam_correctos.png",
        title="Resultados 2 - Grad-CAM en ejemplos correctos",
        columns=2,
    )
    build_image_grid(
        [sample_to_item(sample) for sample in generated["errores"]],
        output_dir / "r2_gradcam_errores.png",
        title="Resultados 2 - Grad-CAM en errores de clasificacion",
        columns=2,
    )
    build_image_grid(
        [sample_to_item(sample) for sample in generated["intact_broken"]],
        output_dir / "r2_gradcam_intact_broken.png",
        title="Resultados 2 - Grad-CAM para intact y broken",
        columns=2,
    )
    demo_sample = generated["correctos"][0] if generated["correctos"] else generated["cinco_clases"][0]
    demo_payload = load_sample_payload(individual_dir / Path(demo_sample.combined_path).name)
    demo_panel = build_probability_panel_image(
        original=demo_payload["original"],
        crop=demo_payload["crop"],
        overlay=demo_payload["overlay"],
        probabilities=demo_payload["probabilities"],
        title="Demo visual Streamlit - Resultados 2",
        metadata={
            "modelo": str(production.get("model_name") or architecture),
            "clase": demo_sample.predicted_label,
            "confianza": f"{demo_sample.confidence:.3f}",
            "capa": demo_sample.gradcam_layer,
        },
    )
    demo_panel.save(output_dir / "r2_panel_visual_demo.png")
    create_r1_vs_r2_dashboard(output_dir / "r1_vs_r2_dashboard.png")

    manifest = {
        "model_name": str(production.get("model_name") or architecture),
        "architecture": architecture,
        "checkpoint_path": relative_to_project(checkpoint),
        "gradcam_layer": gradcam_layer,
        "explanation_scope": "Grad-CAM es aproximado; no es prueba causal ni diagnostico.",
        "output_dir": relative_to_project(output_dir),
        "png": [
            "r2_gradcam_correctos.png",
            "r2_gradcam_errores.png",
            "r2_gradcam_intact_broken.png",
            "r2_panel_visual_demo.png",
            "r1_vs_r2_dashboard.png",
        ],
        "samples": {
            key: [asdict(sample) for sample in samples]
            for key, samples in generated.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML object."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuracion invalida: {path}")
    return loaded


def resolve_predictions_path(architecture: str) -> Path:
    """Return the local predictions CSV for a supported architecture."""
    if architecture.lower() == "efficientnet_b0":
        return PROJECT_ROOT / "results" / "vision" / "resultados_2_mejoras" / "08_comparacion_modelos" / "efficientnet_predictions_test.csv"
    return PROJECT_ROOT / "results" / "vision" / "resultados_2_mejoras" / "05_resnet18_v2" / "predictions_test.csv"


def select_evidence_rows(
    predictions: pd.DataFrame,
    class_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Select correct, error and class-coverage rows from predictions."""
    frame = predictions.copy()
    frame["is_correct"] = frame["true_label"].astype(str) == frame["predicted_label"].astype(str)
    frame = frame.sort_values("predicted_probability", ascending=False)
    correctos = coverage_by_true_label(frame[frame["is_correct"]], class_names, limit=5)
    errores = frame[~frame["is_correct"]].head(5).to_dict("records")
    intact_broken = coverage_by_true_label(
        frame[frame["true_label"].isin(["intact", "broken"])],
        ["intact", "broken"],
        limit=6,
    )
    cinco_clases = coverage_by_true_label(frame, class_names, limit=5)
    return {
        "correctos": correctos,
        "errores": errores,
        "intact_broken": intact_broken,
        "cinco_clases": cinco_clases,
    }


def coverage_by_true_label(
    frame: pd.DataFrame,
    labels: list[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Pick up to one high-confidence row per label, then fill by confidence."""
    selected: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for label in labels:
        rows = frame[frame["true_label"].astype(str) == label]
        if rows.empty:
            continue
        row = rows.iloc[0].to_dict()
        selected.append(row)
        used_paths.add(str(row["image_path"]))
    for row in frame.to_dict("records"):
        if len(selected) >= limit:
            break
        path = str(row["image_path"])
        if path in used_paths:
            continue
        selected.append(row)
        used_paths.add(path)
    return selected[:limit]


def render_sample(
    *,
    row: dict[str, Any],
    engine: VisionInferenceEngine,
    output_dir: Path,
    image_size: int,
) -> EvidenceSample:
    """Generate Grad-CAM artifacts for one selected prediction row."""
    image_path = PROJECT_ROOT / Path(str(row["image_path"]))
    original = Image.open(image_path).convert("RGB")
    preprocessing = preprocess_image(
        original,
        config=PreprocessingConfig(output_size=image_size),
    )
    predicted_label = str(row["predicted_label"])
    target_index = engine.labels.index(predicted_label) if predicted_label in engine.labels else None
    gradcam = generate_gradcam_with_fallback(
        model=engine.model,
        image=preprocessing.crop,
        transform=engine.transform,
        device=engine.device,
        target_class_index=target_index,
    )
    heatmap = heatmap_to_image(gradcam.heatmap, preprocessing.crop.size)
    probabilities = extract_probabilities(row, engine.labels)
    metadata = {
        "real": str(row["true_label"]),
        "pred": predicted_label,
        "conf": f"{float(row['predicted_probability']):.3f}",
        "intensidad": f"{gradcam.intensity:.3f}",
        "capa": gradcam.target_layer_name,
    }
    title = f"{Path(str(row['image_path'])).name} - Grad-CAM aproximado"
    combined = build_combined_gradcam_image(
        original=original,
        crop=preprocessing.crop,
        heatmap=heatmap,
        overlay=gradcam.overlay,
        title=title,
        metadata=metadata,
    )
    stem = Path(str(row["image_path"])).stem
    safe_name = f"{str(row['true_label'])}_{stem}_{predicted_label}.png"
    combined_path = output_dir / safe_name
    combined.save(combined_path)
    payload_path = combined_path.with_suffix(".json")
    payload_path.write_text(
        json.dumps(
            {
                "probabilities": probabilities,
                "gradcam": gradcam_metadata(gradcam),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    original.save(combined_path.with_name(f"{combined_path.stem}_original.png"))
    preprocessing.crop.save(combined_path.with_name(f"{combined_path.stem}_crop.png"))
    gradcam.overlay.save(combined_path.with_name(f"{combined_path.stem}_overlay.png"))
    heatmap.save(combined_path.with_name(f"{combined_path.stem}_heatmap.png"))
    return EvidenceSample(
        image_path=relative_to_project(image_path),
        true_label=str(row["true_label"]),
        predicted_label=predicted_label,
        confidence=float(row["predicted_probability"]),
        gradcam_status=gradcam.status,
        gradcam_layer=gradcam.target_layer_name,
        gradcam_intensity=gradcam.intensity,
        combined_path=relative_to_project(combined_path),
    )


def gradcam_metadata(gradcam: GradCamResult) -> dict[str, Any]:
    """Return JSON-safe Grad-CAM metadata."""
    return {
        "status": gradcam.status,
        "message": gradcam.message,
        "target_class_index": gradcam.target_class_index,
        "target_layer_name": gradcam.target_layer_name,
        "intensity": gradcam.intensity,
    }


def extract_probabilities(row: dict[str, Any], labels: list[str]) -> dict[str, float]:
    """Extract per-class probabilities from a prediction CSV row."""
    values: dict[str, float] = {}
    for label in labels:
        key = f"probability_{label}"
        if key in row:
            values[label] = float(row[key])
    return values


def sample_to_item(sample: EvidenceSample) -> dict[str, Any]:
    """Load a combined sample image for gallery composition."""
    return {"combined": Image.open(PROJECT_ROOT / sample.combined_path).convert("RGB")}


def load_sample_payload(combined_path: Path) -> dict[str, Any]:
    """Load images and probabilities for the visual demo panel."""
    json_path = combined_path.with_suffix(".json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "original": Image.open(combined_path.with_name(f"{combined_path.stem}_original.png")).convert("RGB"),
        "crop": Image.open(combined_path.with_name(f"{combined_path.stem}_crop.png")).convert("RGB"),
        "overlay": Image.open(combined_path.with_name(f"{combined_path.stem}_overlay.png")).convert("RGB"),
        "probabilities": payload.get("probabilities") or {},
    }


def create_r1_vs_r2_dashboard(output_path: Path) -> None:
    """Create or copy a compact R1-vs-R2 dashboard PNG."""
    candidates = [
        PROJECT_ROOT / "results" / "vision" / "resultados_2_mejoras" / "08_comparacion_modelos" / "r1_vs_r2_modelos_dashboard.png",
        PROJECT_ROOT / "results" / "vision" / "resultados_2_mejoras" / "05_resnet18_v2" / "r1_vs_r2_f1_resnet18.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            with Image.open(candidate) as image:
                image.convert("RGB").save(output_path)
            return
    canvas = Image.new("RGB", (900, 320), "white")
    canvas.save(output_path)


def relative_to_project(path: Path) -> str:
    """Return a POSIX path relative to the repository when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
