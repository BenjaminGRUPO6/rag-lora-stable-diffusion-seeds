from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.analyze_seed import (
    DEFAULT_RAG_CONFIG,
    DEFAULT_VISION_CONFIG,
    analyze_seed,
    default_checkpoint_path,
    load_yaml_config,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "end_to_end"
EXPECTED_CLASSES = {"intact", "spotted", "immature", "broken", "skin_damaged"}
FORBIDDEN_DIAGNOSTIC_PHRASES = (
    "spotted es hongo",
    "spotted confirma hongo",
    "enfermedad confirmada",
    "diagnostico fitosanitario definitivo",
)


@dataclass(frozen=True)
class FunctionalValidation:
    """Validation summary for one controlled end-to-end run."""

    passed: bool
    image_path: str
    checkpoint_available: bool
    tensor_created: bool
    model_loaded: bool
    prediction: str
    confidence: float
    probabilities_sum: float
    class_valid: bool
    confidence_valid: bool
    probabilities_valid: bool
    report_structured: bool
    no_private_paths: bool
    no_forbidden_diagnostic: bool
    rag_status: str
    sources_count: int
    sources_real_when_available: bool
    output_json: str
    output_markdown: str
    errors: list[str]


def parse_args() -> argparse.Namespace:
    """Parse functional test arguments."""
    parser = argparse.ArgumentParser(description="Run a controlled SeedCare-RAG functional test.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    """Run the functional test and persist JSON plus Markdown outputs."""
    args = parse_args()
    summary = run_functional_test(
        image_path=args.image,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0 if summary.passed else 1


def run_functional_test(
    image_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device_name: str = "cpu",
) -> FunctionalValidation:
    """Execute one local image analysis and validate its structured output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_image = image_path or find_first_local_image(PROJECT_ROOT / "data" / "processed" / "test")
    checkpoint = default_checkpoint_path(load_yaml_config(PROJECT_ROOT / DEFAULT_VISION_CONFIG))
    errors: list[str] = []
    tensor_created = validate_image_readable(selected_image)
    result: dict[str, Any] = {}
    if tensor_created:
        result = analyze_seed(
            image=selected_image,
            vision_config_path=PROJECT_ROOT / DEFAULT_VISION_CONFIG,
            rag_config_path=PROJECT_ROOT / DEFAULT_RAG_CONFIG,
            index_dir=PROJECT_ROOT / "vector_db",
            checkpoint_path=PROJECT_ROOT / checkpoint,
            device_name=device_name,
        )

    probabilities = result.get("probabilities") or {}
    probabilities_sum = float(sum(float(value) for value in probabilities.values()))
    prediction = str(result.get("prediction") or "")
    confidence = float(result.get("confidence") or 0.0)
    retrieved_sources = result.get("retrieved_sources") or []
    report = result.get("preliminary_report") or {}
    payload_text = json.dumps(result, ensure_ascii=False)
    repo_text = str(PROJECT_ROOT)

    checks = {
        "checkpoint_available": (PROJECT_ROOT / checkpoint).exists(),
        "tensor_created": tensor_created,
        "model_loaded": bool(result),
        "class_valid": prediction in EXPECTED_CLASSES,
        "confidence_valid": 0.0 <= confidence <= 1.0,
        "probabilities_valid": abs(probabilities_sum - 1.0) <= 0.001,
        "report_structured": isinstance(report, dict) and bool(report.get("resumen_visual")),
        "no_private_paths": repo_text not in payload_text and str(Path.home()) not in payload_text,
        "no_forbidden_diagnostic": not contains_forbidden_diagnostic(payload_text),
        "sources_real_when_available": sources_real_when_available(retrieved_sources),
    }
    for key, passed in checks.items():
        if not passed:
            errors.append(key)

    json_path = output_dir / "functional_test.json"
    markdown_path = output_dir / "functional_test.md"
    summary = FunctionalValidation(
        passed=not errors,
        image_path=selected_image.relative_to(PROJECT_ROOT).as_posix(),
        checkpoint_available=checks["checkpoint_available"],
        tensor_created=checks["tensor_created"],
        model_loaded=checks["model_loaded"],
        prediction=prediction,
        confidence=confidence,
        probabilities_sum=probabilities_sum,
        class_valid=checks["class_valid"],
        confidence_valid=checks["confidence_valid"],
        probabilities_valid=checks["probabilities_valid"],
        report_structured=checks["report_structured"],
        no_private_paths=checks["no_private_paths"],
        no_forbidden_diagnostic=checks["no_forbidden_diagnostic"],
        rag_status=str(result.get("rag_status") or "not_run"),
        sources_count=len(retrieved_sources),
        sources_real_when_available=checks["sources_real_when_available"],
        output_json=relative_to_project(json_path),
        output_markdown=relative_to_project(markdown_path),
        errors=errors,
    )
    json_path.write_text(
        json.dumps({"summary": asdict(summary), "result": result}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def find_first_local_image(root: Path) -> Path:
    """Return the first local processed test image."""
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        candidates = sorted(root.rglob(suffix))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No hay imagenes de prueba en {root.as_posix()}")


def relative_to_project(path: Path) -> str:
    """Return a POSIX relative path for generated artifacts."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate_image_readable(path: Path) -> bool:
    """Validate that PIL can read the image and create RGB pixels."""
    with Image.open(path) as image:
        image.convert("RGB")
    return True


def contains_forbidden_diagnostic(text: str) -> bool:
    """Return true when output contains forbidden affirmative diagnostic wording."""
    normalized = text.lower()
    return any(phrase in normalized for phrase in FORBIDDEN_DIAGNOSTIC_PHRASES)


def sources_real_when_available(sources: list[dict[str, Any]]) -> bool:
    """Validate that returned sources include real metadata fields when present."""
    if not sources:
        return True
    for source in sources:
        if not str(source.get("document_id") or "").strip():
            return False
        if not str(source.get("title") or "").strip():
            return False
        if not str(source.get("text") or "").strip():
            return False
    return True


def render_markdown(summary: FunctionalValidation) -> str:
    """Render the functional validation summary as Markdown."""
    return "\n".join(
        [
            "# Functional Test",
            "",
            f"- passed: {summary.passed}",
            f"- image_path: {summary.image_path}",
            f"- checkpoint_available: {summary.checkpoint_available}",
            f"- tensor_created: {summary.tensor_created}",
            f"- model_loaded: {summary.model_loaded}",
            f"- prediction: {summary.prediction}",
            f"- confidence: {summary.confidence:.6f}",
            f"- probabilities_sum: {summary.probabilities_sum:.6f}",
            f"- rag_status: {summary.rag_status}",
            f"- sources_count: {summary.sources_count}",
            f"- no_private_paths: {summary.no_private_paths}",
            f"- no_forbidden_diagnostic: {summary.no_forbidden_diagnostic}",
            f"- errors: {', '.join(summary.errors) if summary.errors else 'none'}",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
