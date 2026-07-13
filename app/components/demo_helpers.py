from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


DISCLAIMER = (
    "Herramienta de apoyo visual. No constituye diagnóstico fitosanitario "
    "ni reemplaza una evaluación especializada."
)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PRIVATE_SOURCE_KEYS = {"local_path", "path", "source_metadata"}
SELECTED_LORA_PARAMETER_KEYS = {
    "base_model",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_train_steps_full",
    "max_train_steps_initial",
    "mixed_precision",
    "rank",
    "resolution",
    "seed",
    "train_batch_size",
    "train_text_encoder",
    "trigger_word",
}


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be accepted by the demo."""


def validate_uploaded_image(file_name: str, data: bytes) -> Image.Image:
    """Validate an uploaded JPG/JPEG/PNG image and return it as RGB PIL image."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise ImageValidationError(f"Formato no permitido. Use: {allowed}.")
    if not data:
        raise ImageValidationError("El archivo cargado esta vacio.")

    try:
        with Image.open(BytesIO(data)) as opened:
            opened.verify()
        with Image.open(BytesIO(data)) as reopened:
            return reopened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("La imagen no es valida o esta danada.") from exc


def top_probabilities(probabilities: dict[str, float], limit: int = 3) -> list[dict[str, float | str]]:
    """Return the highest probabilities sorted descending."""
    ordered = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
    return [
        {"class": str(label), "probability": round(float(probability), 6)}
        for label, probability in ordered[:limit]
    ]


def source_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return display-safe source rows with title, page, URL and fragment only."""
    rows: list[dict[str, Any]] = []
    for source in sources:
        rows.append(
            {
                "title": str(source.get("title") or source.get("document_title") or "Sin titulo"),
                "page": source.get("page"),
                "url": extract_source_url(source),
                "score": source.get("score"),
                "fragment": compact_text(
                    str(
                        source.get("fragment")
                        or source.get("text")
                        or source.get("content")
                        or source.get("snippet")
                        or ""
                    )
                ),
            }
        )
    return rows


def build_download_payload(result: dict[str, Any], observations: str) -> dict[str, Any]:
    """Build a privacy-aware JSON payload for download."""
    probabilities = result.get("probabilities") or {}
    retrieved_sources = result.get("retrieved_sources") or []
    return {
        "disclaimer": DISCLAIMER,
        "observations": [observations.strip()] if observations.strip() else [],
        "prediction": {
            "label": str(result.get("prediction") or ""),
            "confidence": round(float(result.get("confidence") or 0.0), 6),
            "top_probabilities": top_probabilities(probabilities),
            "uncertainty_status": str(result.get("uncertainty_status") or ""),
        },
        "retrieved_information": source_rows(retrieved_sources),
        "preliminary_report": sanitize_report(result.get("preliminary_report") or {}),
        "limitations": [str(item) for item in result.get("limitations") or []],
        "processing_times": {
            str(key): round(float(value), 6)
            for key, value in (result.get("processing_times") or {}).items()
        },
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    """Render a download payload as Markdown."""
    prediction = payload.get("prediction") or {}
    lines = [
        "# SeedCare-RAG - informe preliminar",
        "",
        str(payload.get("disclaimer") or DISCLAIMER),
        "",
        "## Resultado visual",
        f"- Clase estimada: {prediction.get('label', '')}",
        f"- Confianza: {float(prediction.get('confidence') or 0.0):.4f}",
        f"- Estado de incertidumbre: {prediction.get('uncertainty_status', '')}",
        "",
        "## Top 3 probabilidades",
    ]
    for item in prediction.get("top_probabilities") or []:
        lines.append(f"- {item.get('class')}: {float(item.get('probability') or 0.0):.4f}")

    lines.extend(["", "## Informacion recuperada"])
    rows = payload.get("retrieved_information") or []
    if rows:
        for index, row in enumerate(rows, start=1):
            page = row.get("page")
            page_text = f", pagina {page}" if page not in (None, "") else ""
            url = row.get("url") or "URL no disponible"
            lines.extend(
                [
                    f"{index}. {row.get('title', 'Sin titulo')}{page_text}",
                    f"   - URL: {url}",
                    f"   - Fragmento: {row.get('fragment', '')}",
                ]
            )
    else:
        lines.append("- No se recuperaron documentos para este analisis.")

    report = payload.get("preliminary_report") or {}
    lines.extend(["", "## Informe preliminar"])
    summary = report.get("resumen_visual") or report.get("informe_generado")
    if summary:
        lines.append(str(summary))
    else:
        lines.append("No hay informe textual disponible.")

    lines.extend(["", "## Limitaciones"])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")

    lines.extend(["", "## Tiempos de procesamiento"])
    for key, value in (payload.get("processing_times") or {}).items():
        lines.append(f"- {key}: {float(value):.6f} s")

    return "\n".join(lines).strip() + "\n"


def load_lora_evidence(results_dir: str | Path = Path("results/lora")) -> dict[str, Any]:
    """Load selected LoRA evidence without exposing weights or private paths."""
    root = Path(results_dir)
    if not root.exists():
        return {"available": False, "message": "No hay evidencia LoRA local en results/lora."}

    run_manifest = read_json(root / "run_manifest.json")
    training_summary = read_json(root / "training_summary.json")
    dataset_summary = read_json(root / "dataset_summary.json")
    evidence_inventory = read_json(root / "evidence_inventory.json")

    dataset = run_manifest.get("dataset") if isinstance(run_manifest.get("dataset"), dict) else {}
    class_distribution = (
        dataset_summary.get("class_distribution")
        or dataset.get("class_distribution")
        or training_summary.get("class_distribution")
        or {}
    )
    parameters = selected_lora_parameters(
        training_summary.get("training_parameters")
        or run_manifest.get("confirmed_parameters")
        or {}
    )
    missing_evidence = (
        run_manifest.get("missing_evidence")
        or evidence_inventory.get("missing")
        or []
    )
    notes = dataset_summary.get("notes") if isinstance(dataset_summary.get("notes"), list) else []
    samples_dir = root / "samples"
    sample_names = sorted(path.name for path in samples_dir.glob("*.jpg")) if samples_dir.exists() else []

    return {
        "available": True,
        "status": str(run_manifest.get("status") or training_summary.get("status") or "SIN ESTADO"),
        "no_retraining_performed": bool(
            run_manifest.get("no_retraining_performed")
            or training_summary.get("no_retraining_performed")
        ),
        "dataset_images": dataset_summary.get("metadata_records")
        or dataset.get("metadata_records")
        or training_summary.get("dataset_images"),
        "class_distribution": class_distribution,
        "parameters": parameters,
        "missing_evidence": [str(item) for item in missing_evidence],
        "notes": [str(item) for item in notes],
        "sample_names": sample_names,
    }


def selected_lora_parameters(raw_parameters: Any) -> dict[str, Any]:
    """Return only selected, display-safe LoRA parameters."""
    if not isinstance(raw_parameters, dict):
        return {}
    selected: dict[str, Any] = {}
    for key, value in raw_parameters.items():
        if key not in SELECTED_LORA_PARAMETER_KEYS:
            continue
        if isinstance(value, dict) and "value" in value:
            selected[key] = value["value"]
        else:
            selected[key] = value
    return selected


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty dict when absent."""
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def sanitize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep only report fields that do not expose local source paths."""
    allowed_keys = {
        "categoria_estimada",
        "confianza",
        "estado_incertidumbre",
        "observaciones",
        "resumen_visual",
        "informe_generado",
        "informacion_documental",
        "posibles_factores_descritos_por_las_fuentes",
        "prevencion_o_manejo",
    }
    return {
        key: sanitize_value(value)
        for key, value in report.items()
        if key in allowed_keys
    }


def sanitize_value(value: Any) -> Any:
    """Recursively remove private path keys from report values."""
    if isinstance(value, dict):
        return {
            str(key): sanitize_value(item)
            for key, item in value.items()
            if str(key) not in PRIVATE_SOURCE_KEYS
        }
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def extract_source_url(source: dict[str, Any]) -> str:
    """Return an HTTP(S) source URL when present."""
    for key in ("url", "source_url", "source"):
        candidate = str(source.get(key) or "").strip()
        if candidate.startswith(("http://", "https://")):
            return candidate
    return ""


def compact_text(text: str, limit: int = 500) -> str:
    """Normalize whitespace and truncate long text for display."""
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[: limit - 3].rstrip()}..."


def sanitize_error_message(message: str, repo_root: str | Path | None = None) -> str:
    """Remove absolute local paths from an error message before display."""
    sanitized = message
    roots = []
    if repo_root is not None:
        roots.append(Path(repo_root))
    try:
        roots.append(Path.home())
    except RuntimeError:
        pass
    for root in roots:
        root_text = str(root)
        sanitized = sanitized.replace(root_text, ".")
        sanitized = sanitized.replace(root_text.replace("\\", "/"), ".")
    return sanitized


def is_memory_error(exc: BaseException) -> bool:
    """Return true for Python, CUDA or torch memory allocation failures."""
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text or "memory allocation" in text
