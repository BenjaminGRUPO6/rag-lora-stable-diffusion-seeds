"""Utilities to consolidate visual evidence for the trained SD 1.5 LoRA.

The functions in this module read existing local evidence only. They do not
load Stable Diffusion, open safetensors tensors, train, or generate images.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "lora_sd15.yaml"
METADATA_PATH = REPO_ROOT / "data" / "lora" / "train" / "metadata.jsonl"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "06_entrenamiento_lora_sd15_colab.ipynb"
LEGACY_RESULTS_DIR = REPO_ROOT / "results" / "lora"
VISION_LORA_RESULTS_DIR = (
    REPO_ROOT / "results" / "vision" / "resultados_2_mejoras" / "10_lora_generativo"
)
ADAPTER_SEARCH_ROOT = REPO_ROOT / "models" / "lora"

CLASS_NAMES = ("intact", "broken", "spotted", "immature", "skin_damaged")
PRIVATE_PATH_PATTERN = re.compile(
    r"([A-Za-z]:\\Users\\[^\\/\s]+)|(/Users/[^/\s]+)|(/home/[^/\s]+)"
)
MANDATORY_EXPLANATION = (
    "El LoRA genera imágenes sintéticas de semillas. No clasifica la imagen "
    "cargada y no modifica la confianza del clasificador ResNet18."
)
DISPLAY_CLASS_NAMES = {
    "intact": "intact",
    "broken": "broken",
    "spotted": "spotted (categoria visual)",
    "immature": "immature",
    "skin_damaged": "skin_damaged",
}


@dataclass(frozen=True)
class EvidenceValue:
    """A verified value and the repository-local source that supports it."""

    value: Any
    source: str


def repo_path(path: Path) -> str:
    """Return a repository-relative POSIX path."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix() if not path.is_absolute() else path.name


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object from disk."""

    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty dict when absent."""

    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, data: Any) -> None:
    """Write a stable UTF-8 JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_metadata_records(path: Path = METADATA_PATH) -> list[dict[str, Any]]:
    """Read LoRA metadata JSONL records."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"metadata.jsonl line {line_number} is not a JSON object")
        records.append(record)
    return records


def class_from_record(record: dict[str, Any]) -> str | None:
    """Extract the visual class from metadata when it is explicit."""

    file_name = str(record.get("file_name", ""))
    text = str(record.get("text", ""))
    parts = [part.lower() for part in Path(file_name).parts]
    stem = Path(file_name).stem.lower()
    for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
        if class_name in parts or stem.startswith(f"{class_name}_"):
            return class_name
    for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(class_name)}\b", text.lower()):
            return class_name
    return None


def iter_strings(value: Any) -> list[str]:
    """Collect string leaves from a nested serializable value."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for key, item in value.items():
            strings.extend(iter_strings(str(key)))
            strings.extend(iter_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(iter_strings(item))
        return strings
    return []


def contains_private_path(value: Any) -> bool:
    """Return true when a private absolute user path appears in a value."""

    return any(PRIVATE_PATH_PATTERN.search(text) for text in iter_strings(value))


def config_parameters(config_path: Path = CONFIG_PATH) -> dict[str, EvidenceValue]:
    """Extract display-safe LoRA parameters verified in configs/lora_sd15.yaml."""

    config = load_yaml(config_path)
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    mapping = {
        "base_model": model.get("base_model"),
        "trigger_word": model.get("trigger_word"),
        "resolution": training.get("resolution"),
        "rank": training.get("rank"),
        "learning_rate": training.get("learning_rate"),
        "max_train_steps_initial": training.get("max_train_steps_initial"),
        "max_train_steps_full": training.get("max_train_steps_full"),
        "train_batch_size": training.get("train_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "mixed_precision": training.get("mixed_precision"),
        "train_text_encoder": training.get("train_text_encoder"),
        "seed": training.get("seed"),
    }
    return {
        key: EvidenceValue(value=value, source=repo_path(config_path))
        for key, value in mapping.items()
        if value is not None
    }


def notebook_summary(path: Path = NOTEBOOK_PATH) -> dict[str, Any]:
    """Summarize notebook execution evidence without executing it."""

    if not path.exists():
        return {
            "exists": False,
            "total_cells": 0,
            "code_cells": 0,
            "executed_code_cells": 0,
            "cells_with_outputs": 0,
            "source": repo_path(path),
        }
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
    code_cells = 0
    executed_code_cells = 0
    cells_with_outputs = 0
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") == "code":
            code_cells += 1
            if cell.get("execution_count") is not None:
                executed_code_cells += 1
        if cell.get("outputs"):
            cells_with_outputs += 1
    return {
        "exists": True,
        "total_cells": len(cells),
        "code_cells": code_cells,
        "executed_code_cells": executed_code_cells,
        "cells_with_outputs": cells_with_outputs,
        "source": repo_path(path),
    }


def find_safetensors(search_root: Path = ADAPTER_SEARCH_ROOT) -> list[Path]:
    """Find local safetensors adapters without loading their contents."""

    if not search_root.exists():
        return []
    return sorted(search_root.rglob("*.safetensors"), key=repo_path)


def adapter_info(paths: list[Path] | None = None) -> dict[str, Any]:
    """Return adapter file evidence based on filesystem metadata only."""

    adapters = find_safetensors() if paths is None else paths
    if not adapters:
        return {"exists": False, "status": "EVIDENCE_MISSING", "candidates": []}
    adapter = adapters[0]
    stat = adapter.stat()
    return {
        "exists": True,
        "status": "FOUND",
        "adapter_name": adapter.parent.name,
        "file_name": adapter.name,
        "path": repo_path(adapter),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "last_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "candidate_count": len(adapters),
    }


def metadata_summary(records: list[dict[str, Any]], metadata_path: Path = METADATA_PATH) -> dict[str, Any]:
    """Summarize LoRA training metadata and referenced image availability."""

    distribution: Counter[str] = Counter()
    existing_images = 0
    base_dir = metadata_path.parent
    for record in records:
        class_name = class_from_record(record)
        if class_name:
            distribution[class_name] += 1
        file_name = record.get("file_name")
        if isinstance(file_name, str) and (base_dir / file_name).exists():
            existing_images += 1
    captions = [str(record.get("text") or "") for record in records]
    return {
        "metadata_path": repo_path(metadata_path),
        "metadata_records": len(records),
        "referenced_images_existing": existing_images,
        "class_distribution": {
            class_name: distribution.get(class_name, 0) for class_name in CLASS_NAMES
        },
        "metadata_private_paths_detected": contains_private_path(records),
        "caption_examples": captions[:5],
    }


def select_sample_rows(
    records: list[dict[str, Any]],
    metadata_path: Path = METADATA_PATH,
    per_class_limit: int = 1,
) -> list[dict[str, str]]:
    """Select deterministic existing image samples from metadata."""

    selected: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for record in records:
        class_name = class_from_record(record)
        if class_name is None or selected[class_name] >= per_class_limit:
            continue
        file_name = record.get("file_name")
        if not isinstance(file_name, str):
            continue
        image_path = metadata_path.parent / file_name
        if not image_path.exists():
            continue
        rows.append(
            {
                "class_name": class_name,
                "image_role": "training_metadata_sample",
                "image_path": repo_path(image_path),
                "prompt": str(record.get("text") or ""),
                "seed": "",
                "evidence_status": "FOUND",
            }
        )
        selected[class_name] += 1
    for class_name in CLASS_NAMES:
        if selected[class_name] == 0:
            rows.append(
                {
                    "class_name": class_name,
                    "image_role": "training_metadata_sample",
                    "image_path": "",
                    "prompt": "",
                    "seed": "",
                    "evidence_status": "EVIDENCE_MISSING",
                }
            )
    return rows


def find_comparison_images() -> list[Path]:
    """Find existing base/adapted/comparison images, excluding training samples."""

    search_roots = [LEGACY_RESULTS_DIR, ADAPTER_SEARCH_ROOT]
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    hint_pattern = re.compile(r"(base|lora|comparison|compare|adaptado|adapted|vs)", re.I)
    found: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in image_extensions:
                continue
            relative = repo_path(path)
            if "results/lora/samples/" in relative:
                continue
            if hint_pattern.search(relative):
                found.append(path)
    return sorted(found, key=repo_path)


def comparison_rows(images: list[Path]) -> list[dict[str, str]]:
    """Build a base-vs-adapted manifest from existing comparison images only."""

    rows: list[dict[str, str]] = []
    for class_name in CLASS_NAMES:
        class_images = [path for path in images if class_name in repo_path(path).lower()]
        base = next((path for path in class_images if "base" in repo_path(path).lower()), None)
        adapted = next(
            (
                path
                for path in class_images
                if "lora" in repo_path(path).lower() or "adapt" in repo_path(path).lower()
            ),
            None,
        )
        comparison = next(
            (
                path
                for path in class_images
                if re.search(r"(comparison|compare|vs)", repo_path(path), re.I)
            ),
            None,
        )
        rows.append(
            {
                "class_name": class_name,
                "prompt": "",
                "seed": "",
                "base_image": repo_path(base) if base else "",
                "adapted_image": repo_path(adapted) if adapted else "",
                "comparison_image": repo_path(comparison) if comparison else "",
                "evidence_status": "FOUND" if any([base, adapted, comparison]) else "EVIDENCE_MISSING",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows using a fixed header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evidence_value_payload(values: dict[str, EvidenceValue]) -> dict[str, dict[str, Any]]:
    """Serialize evidence values with source metadata."""

    return {
        key: {"value": evidence.value, "source": evidence.source}
        for key, evidence in sorted(values.items())
    }


def build_lora_evidence() -> dict[str, Any]:
    """Build the LoRA visual evidence payload without writing files."""

    records = read_metadata_records(METADATA_PATH)
    parameters = config_parameters(CONFIG_PATH)
    adapter = adapter_info()
    notebook = notebook_summary(NOTEBOOK_PATH)
    dataset = metadata_summary(records, METADATA_PATH)
    samples = select_sample_rows(records, METADATA_PATH)
    comparisons = comparison_rows(find_comparison_images())

    found: list[str] = []
    missing: list[str] = []
    if CONFIG_PATH.exists():
        found.append(f"Config: {repo_path(CONFIG_PATH)}")
    else:
        missing.append(f"Config file missing: {repo_path(CONFIG_PATH)}")
    if METADATA_PATH.exists():
        found.append(f"Metadata records: {len(records)} from {repo_path(METADATA_PATH)}")
    else:
        missing.append(f"Metadata missing: {repo_path(METADATA_PATH)}")
    if notebook["exists"]:
        found.append(
            "Notebook: "
            f"{notebook['source']}; executed code cells {notebook['executed_code_cells']}; "
            f"cells with outputs {notebook['cells_with_outputs']}"
        )
    else:
        missing.append(f"Notebook missing: {repo_path(NOTEBOOK_PATH)}")
    if adapter["exists"]:
        found.append(
            f"LoRA adapter file: {adapter['file_name']} "
            f"({adapter['size_bytes']} bytes) at {adapter['path']}"
        )
    else:
        missing.append("LoRA adapter safetensors not found.")
    if all(row["evidence_status"] == "EVIDENCE_MISSING" for row in comparisons):
        missing.append("Base-vs-adapted comparison images are not available.")
    else:
        found.append("At least one base-vs-adapted comparison image exists.")
    if notebook["cells_with_outputs"] == 0:
        missing.append("Notebook outputs/logs are missing; hardware is not verified.")
    missing.append("Parameter not verified: hardware")

    required_parameters = {
        "base_model",
        "resolution",
        "rank",
        "learning_rate",
        "max_train_steps_full",
        "trigger_word",
    }
    for parameter in sorted(required_parameters - set(parameters)):
        missing.append(f"Parameter not verified: {parameter}")

    comparison_status = (
        "EVIDENCE_MISSING"
        if all(row["evidence_status"] == "EVIDENCE_MISSING" for row in comparisons)
        else "FOUND"
    )
    status = "PARTIAL" if missing else "COMPLETE"
    payload = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mandatory_explanation": MANDATORY_EXPLANATION,
        "no_retraining_performed": True,
        "no_mass_generation_performed": True,
        "stable_diffusion_loaded": False,
        "lora_inference_required": False,
        "classification_impact_claim": "NOT_CLAIMED",
        "spotted_semantics": "visual_category_only",
        "synthetic_train_policy": "Synthetic images may enter train only after human review.",
        "sources": {
            "config": repo_path(CONFIG_PATH),
            "metadata": repo_path(METADATA_PATH),
            "notebook": repo_path(NOTEBOOK_PATH),
            "legacy_lora_results": repo_path(LEGACY_RESULTS_DIR),
        },
        "model": {
            "base_model": evidence_value_payload(parameters).get("base_model"),
            "adapter": adapter,
            "adapter_name": adapter.get("adapter_name") if adapter.get("exists") else None,
        },
        "training": {
            "parameters": evidence_value_payload(parameters),
            "hardware": {"value": None, "source": "EVIDENCE_MISSING"},
            "steps": {
                "initial": evidence_value_payload(parameters).get("max_train_steps_initial"),
                "full": evidence_value_payload(parameters).get("max_train_steps_full"),
            },
        },
        "dataset": dataset,
        "samples": samples,
        "comparison": {
            "status": comparison_status,
            "rows": comparisons,
        },
        "evidence_found": found,
        "evidence_missing": sorted(dict.fromkeys(missing)),
        "limitations": [
            "This evidence card does not evaluate classifier performance.",
            "No Stable Diffusion inference was executed by this consolidation.",
            "Base-vs-adapted visual comparisons remain missing unless existing files are provided.",
            "`spotted` is a visual category and is not presented as a fungal diagnosis.",
        ],
    }
    if contains_private_path(payload):
        raise ValueError("Private absolute path detected in LoRA evidence payload")
    return payload


def load_lora_visual_evidence(
    evidence_path: Path = VISION_LORA_RESULTS_DIR / "lora_evidence.json",
) -> dict[str, Any]:
    """Load consolidated LoRA visual evidence for Streamlit display."""

    if not evidence_path.exists():
        return {
            "available": False,
            "message": f"Evidencia LoRA no consolidada: {repo_path(evidence_path)}",
        }
    payload = read_json(evidence_path)
    payload["available"] = True
    if contains_private_path(payload):
        raise ValueError("Private absolute path detected in LoRA visual evidence")
    return payload


def write_model_card(path: Path, evidence: dict[str, Any]) -> None:
    """Write a Markdown model card for the consolidated LoRA evidence."""

    parameters = evidence.get("training", {}).get("parameters", {})
    dataset = evidence.get("dataset", {})
    adapter = evidence.get("model", {}).get("adapter", {})
    lines = [
        "# Modelo generativo LoRA SD 1.5",
        "",
        f"Status: **{evidence.get('status', 'UNKNOWN')}**",
        "",
        MANDATORY_EXPLANATION,
        "",
        "## Que hace",
        "- Genera imagenes sinteticas de semillas cuando se carga externamente en un pipeline SD 1.5.",
        "- Registra evidencia de configuracion, metadata y adaptador local ya entrenado.",
        "",
        "## Que no hace",
        "- No clasifica imagenes cargadas en Streamlit.",
        "- No modifica probabilidades ni confianza de ResNet18.",
        "- No ejecuta generacion automatica ni masiva en esta etapa.",
        "",
        "## Evidencia verificada",
        f"- Modelo base: `{_value(parameters, 'base_model')}`",
        f"- Trigger word: `{_value(parameters, 'trigger_word')}`",
        f"- Resolucion: `{_value(parameters, 'resolution')}`",
        f"- Rank: `{_value(parameters, 'rank')}`",
        f"- Learning rate: `{_value(parameters, 'learning_rate')}`",
        f"- Pasos inicial/full: `{_value(parameters, 'max_train_steps_initial')}` / "
        f"`{_value(parameters, 'max_train_steps_full')}`",
        f"- Hardware: `EVIDENCE_MISSING`",
        f"- Adaptador: `{adapter.get('adapter_name') or 'EVIDENCE_MISSING'}`",
        f"- Archivo adaptador: `{adapter.get('file_name') or 'EVIDENCE_MISSING'}`",
        f"- Metadata: `{dataset.get('metadata_records', 0)}` registros; "
        f"`{dataset.get('referenced_images_existing', 0)}` imagenes existentes.",
        "",
        "## Clases visuales",
    ]
    distribution = dataset.get("class_distribution", {})
    for class_name in CLASS_NAMES:
        label = DISPLAY_CLASS_NAMES[class_name]
        lines.append(f"- {label}: `{distribution.get(class_name, 0)}`")
    lines.extend(["", "## Evidencia faltante"])
    for item in evidence.get("evidence_missing", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Aviso de uso",
            "- Las imagenes sinteticas solo pueden incorporarse a `train` despues de revision humana.",
            "- `spotted` se conserva como categoria visual, no como diagnostico de hongo.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _value(parameters: dict[str, Any], key: str) -> Any:
    """Return a display value from a serialized parameter map."""

    value = parameters.get(key, {})
    if isinstance(value, dict):
        return value.get("value", "EVIDENCE_MISSING")
    return "EVIDENCE_MISSING"


def draw_all_pngs(output_dir: Path, evidence: dict[str, Any]) -> list[str]:
    """Generate the required PNG evidence panels using existing values only."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        draw_model_card_png(output_dir / "r2_lora_model_card.png", evidence),
        draw_base_vs_adapted_png(output_dir / "r2_lora_base_vs_adaptado.png", evidence),
        draw_classes_png(output_dir / "r2_lora_clases.png", evidence),
        draw_flow_png(output_dir / "r2_lora_flujo.png", evidence),
    ]
    return [repo_path(path) for path in paths]


def new_canvas(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a white RGB drawing canvas."""

    image = Image.new("RGB", size, "white")
    return image, ImageDraw.Draw(image)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a portable TrueType font when available."""

    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    text_font: ImageFont.ImageFont,
    fill: str = "#1f2933",
    line_gap: int = 6,
) -> int:
    """Draw wrapped text and return the next y position."""

    x, y = xy
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textlength(candidate, font=text_font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    line_height = int(text_font.size * 1.25) if hasattr(text_font, "size") else 18
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height + line_gap
    return y


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: list[str],
) -> None:
    """Draw a simple bordered information block."""

    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=8, outline="#b8c2cc", width=2, fill="#f8fafc")
    draw.text((x1 + 18, y1 + 14), title, font=font(22, bold=True), fill="#102a43")
    y = y1 + 52
    for line in body:
        y = draw_wrapped(draw, (x1 + 18, y), line, x2 - x1 - 36, font(17), "#334e68", 2)
        y += 5
        if y > y2 - 20:
            break


def draw_model_card_png(path: Path, evidence: dict[str, Any]) -> Path:
    """Draw the LoRA model card PNG."""

    image, draw = new_canvas((1400, 900))
    draw.text((48, 36), "R2 - Modelo generativo LoRA", font=font(42, True), fill="#102a43")
    draw.text((48, 92), str(evidence.get("status", "")), font=font(24, True), fill="#9f1239")
    y = draw_wrapped(draw, (48, 140), MANDATORY_EXPLANATION, 1260, font(24), "#243b53", 10)
    parameters = evidence.get("training", {}).get("parameters", {})
    adapter = evidence.get("model", {}).get("adapter", {})
    draw_card(
        draw,
        (48, y + 24, 660, y + 310),
        "Evidencia de entrenamiento",
        [
            f"Base: {_value(parameters, 'base_model')}",
            f"Resolucion: {_value(parameters, 'resolution')} px",
            f"Rank: {_value(parameters, 'rank')}",
            f"Learning rate: {_value(parameters, 'learning_rate')}",
            f"Pasos full: {_value(parameters, 'max_train_steps_full')}",
            "Hardware: EVIDENCE_MISSING",
        ],
    )
    draw_card(
        draw,
        (710, y + 24, 1352, y + 310),
        "Adaptador",
        [
            f"Nombre: {adapter.get('adapter_name') or 'EVIDENCE_MISSING'}",
            f"Archivo: {adapter.get('file_name') or 'EVIDENCE_MISSING'}",
            f"Tamano: {adapter.get('size_mb', 'EVIDENCE_MISSING')} MB",
            "La app no carga safetensors al iniciar.",
            "No se afirma mejora sobre ResNet18.",
        ],
    )
    draw_card(
        draw,
        (48, y + 350, 1352, y + 610),
        "Limitaciones",
        [
            "No hay comparaciones base-vs-adaptado verificadas.",
            "No se ejecuto Stable Diffusion en esta consolidacion.",
            "Las imagenes sinteticas requieren revision humana antes de entrenamiento.",
            "spotted se reporta solo como categoria visual.",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_base_vs_adapted_png(path: Path, evidence: dict[str, Any]) -> Path:
    """Draw the base-vs-adapted evidence status PNG."""

    image, draw = new_canvas((1400, 760))
    draw.text((48, 36), "R2 - Base vs LoRA adaptado", font=font(42, True), fill="#102a43")
    draw_wrapped(
        draw,
        (48, 100),
        "No se encontraron imagenes comparativas existentes. Estado: EVIDENCE_MISSING. "
        "No se ejecuto Stable Diffusion automaticamente.",
        1280,
        font(25),
        "#334e68",
        8,
    )
    headers = ["Clase", "Prompt", "Seed", "Base", "Adaptado", "Estado"]
    rows = evidence.get("comparison", {}).get("rows", [])
    x_positions = [52, 270, 520, 650, 840, 1060]
    y = 210
    draw.rectangle((48, y, 1352, y + 56), fill="#e0f2fe", outline="#7dd3fc")
    for x, header in zip(x_positions, headers):
        draw.text((x, y + 15), header, font=font(20, True), fill="#0f172a")
    y += 56
    for row in rows:
        draw.rectangle((48, y, 1352, y + 72), outline="#cbd5e1", fill="#ffffff")
        values = [
            row.get("class_name", ""),
            row.get("prompt", "") or "EVIDENCE_MISSING",
            row.get("seed", "") or "EVIDENCE_MISSING",
            "FOUND" if row.get("base_image") else "MISSING",
            "FOUND" if row.get("adapted_image") else "MISSING",
            row.get("evidence_status", ""),
        ]
        for x, value in zip(x_positions, values):
            draw_wrapped(draw, (x, y + 13), str(value), 190, font(16), "#334155", 0)
        y += 72
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_classes_png(path: Path, evidence: dict[str, Any]) -> Path:
    """Draw class distribution and sample thumbnails from existing metadata."""

    image, draw = new_canvas((1400, 900))
    draw.text((48, 36), "R2 - Clases usadas por LoRA", font=font(42, True), fill="#102a43")
    dataset = evidence.get("dataset", {})
    distribution = dataset.get("class_distribution", {})
    samples = evidence.get("samples", [])
    max_count = max([int(distribution.get(class_name, 0)) for class_name in CLASS_NAMES] or [1])
    y = 130
    for class_name in CLASS_NAMES:
        count = int(distribution.get(class_name, 0))
        label = DISPLAY_CLASS_NAMES[class_name]
        draw.text((60, y), label, font=font(22, True), fill="#102a43")
        bar_width = int(520 * (count / max_count)) if max_count else 0
        draw.rectangle((310, y + 4, 310 + bar_width, y + 34), fill="#2563eb")
        draw.text((850, y), str(count), font=font(22), fill="#334155")
        y += 58
    draw.text((60, 470), "Muestras existentes de metadata", font=font(28, True), fill="#102a43")
    x = 60
    y = 530
    for row in samples[:5]:
        image_path = REPO_ROOT / str(row.get("image_path", ""))
        draw.rounded_rectangle((x, y, x + 230, y + 260), radius=8, outline="#cbd5e1", width=2)
        if image_path.exists():
            with Image.open(image_path) as sample:
                thumbnail = sample.convert("RGB")
                thumbnail.thumbnail((190, 170))
                image.paste(thumbnail, (x + 20, y + 20))
        draw_wrapped(draw, (x + 16, y + 202), row.get("class_name", ""), 198, font(18, True))
        draw_wrapped(draw, (x + 16, y + 230), row.get("image_role", ""), 198, font(14), "#52606d")
        x += 260
    draw_wrapped(
        draw,
        (60, 820),
        "Estas muestras provienen de metadata de entrenamiento, no de una generacion nueva.",
        1240,
        font(20),
        "#52606d",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def draw_flow_png(path: Path, evidence: dict[str, Any]) -> Path:
    """Draw the LoRA evidence flow PNG."""

    image, draw = new_canvas((1400, 720))
    draw.text((48, 36), "R2 - Flujo LoRA generativo", font=font(42, True), fill="#102a43")
    steps = [
        ("Config", "configs/lora_sd15.yaml"),
        ("Metadata", "1000 registros verificados"),
        ("Adaptador", "safetensors existente; no se carga"),
        ("Streamlit", "visualiza evidencia; no clasifica"),
        ("Revision humana", "requerida antes de train"),
    ]
    x = 70
    y = 210
    for index, (title, body) in enumerate(steps):
        draw.rounded_rectangle((x, y, x + 220, y + 170), radius=10, fill="#f8fafc", outline="#94a3b8", width=2)
        draw.text((x + 20, y + 22), title, font=font(24, True), fill="#102a43")
        draw_wrapped(draw, (x + 20, y + 64), body, 180, font(17), "#334155", 4)
        if index < len(steps) - 1:
            draw.line((x + 230, y + 85, x + 282, y + 85), fill="#64748b", width=4)
            draw.polygon(
                [(x + 282, y + 85), (x + 264, y + 75), (x + 264, y + 95)],
                fill="#64748b",
            )
        x += 260
    draw_wrapped(draw, (70, 470), MANDATORY_EXPLANATION, 1260, font(26), "#334e68", 10)
    draw_wrapped(
        draw,
        (70, 590),
        "Salida de esta etapa: JSON, model card, manifiesto CSV y PNGs de evidencia. "
        "No hay generacion masiva ni reentrenamiento.",
        1260,
        font(20),
        "#52606d",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def write_visual_evidence_bundle(output_dir: Path = VISION_LORA_RESULTS_DIR) -> dict[str, Any]:
    """Write JSON, Markdown, CSV and PNG evidence artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_lora_evidence()
    evidence["generated_pngs"] = draw_all_pngs(output_dir, evidence)
    evidence_path = output_dir / "lora_evidence.json"
    write_json(evidence_path, evidence)
    write_model_card(output_dir / "lora_model_card.md", evidence)
    write_csv(
        output_dir / "lora_samples_manifest.csv",
        evidence["samples"],
        ["class_name", "image_role", "image_path", "prompt", "seed", "evidence_status"],
    )
    write_csv(
        output_dir / "lora_base_vs_adaptado_manifest.csv",
        evidence["comparison"]["rows"],
        ["class_name", "prompt", "seed", "base_image", "adapted_image", "comparison_image", "evidence_status"],
    )
    if contains_private_path(evidence):
        raise ValueError("Private absolute path detected in written LoRA evidence")
    return evidence
