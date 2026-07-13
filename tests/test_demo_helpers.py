from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.components.demo_helpers import (
    DISCLAIMER,
    ImageValidationError,
    build_download_payload,
    build_markdown_report,
    load_lora_evidence,
    sanitize_error_message,
    source_rows,
    top_probabilities,
    validate_uploaded_image,
)


def make_png_bytes() -> bytes:
    """Create a small valid PNG image in memory."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_uploaded_image_accepts_png() -> None:
    image = validate_uploaded_image("seed.png", make_png_bytes())

    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_validate_uploaded_image_rejects_invalid_extension() -> None:
    with pytest.raises(ImageValidationError, match="Formato no permitido"):
        validate_uploaded_image("seed.webp", make_png_bytes())


def test_validate_uploaded_image_rejects_invalid_bytes() -> None:
    with pytest.raises(ImageValidationError, match="no es valida"):
        validate_uploaded_image("seed.jpg", b"not an image")


def test_top_probabilities_sorts_and_limits() -> None:
    result = top_probabilities(
        {"spotted": 0.2, "broken": 0.7, "intact": 0.1, "immature": 0.4},
        limit=3,
    )

    assert result == [
        {"class": "broken", "probability": 0.7},
        {"class": "immature", "probability": 0.4},
        {"class": "spotted", "probability": 0.2},
    ]


def test_source_rows_hide_local_paths_and_keep_urls() -> None:
    rows = source_rows(
        [
            {
                "title": "Manual tecnico",
                "page": 4,
                "source_url": "https://example.org/manual.pdf",
                "local_path": r"C:\private\manual.pdf",
                "text": " fragmento   recuperado ",
            },
            {
                "title": "Solo local",
                "local_path": r"C:\private\local.pdf",
                "text": "sin url",
            },
        ]
    )

    assert rows[0]["url"] == "https://example.org/manual.pdf"
    assert rows[0]["fragment"] == "fragmento recuperado"
    assert rows[1]["url"] == ""
    assert "private" not in str(rows)


def test_download_payload_and_markdown_are_privacy_aware() -> None:
    result = {
        "prediction": "spotted",
        "confidence": 0.81,
        "probabilities": {"spotted": 0.81, "intact": 0.12, "broken": 0.07},
        "uncertainty_status": "certain",
        "retrieved_sources": [
            {
                "title": "Fuente",
                "page": 2,
                "source_url": "https://example.org/source.pdf",
                "local_path": r"D:\private\source.pdf",
                "text": "Evidencia seleccionada.",
            }
        ],
        "preliminary_report": {
            "resumen_visual": "Clasificacion visual preliminar.",
            "referencias": [r"D:\private\source.pdf"],
            "informacion_documental": [
                {
                    "title": "Fuente",
                    "fragment": "Evidencia seleccionada.",
                    "local_path": r"D:\private\source.pdf",
                }
            ],
        },
        "limitations": ["spotted describe alteraciones visibles; no confirma hongo."],
        "processing_times": {"total_seconds": 0.5},
    }

    payload = build_download_payload(result, "manchas visibles")
    markdown = build_markdown_report(payload)

    assert payload["disclaimer"] == DISCLAIMER
    assert payload["prediction"]["top_probabilities"][0]["class"] == "spotted"
    assert "private" not in str(payload)
    assert "private" not in markdown
    assert "no confirma hongo" in markdown


def test_load_lora_evidence_selects_available_public_fields(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "lora"
    samples_dir = results_dir / "samples"
    samples_dir.mkdir(parents=True)
    (samples_dir / "intact_00001.jpg").write_bytes(make_png_bytes())
    (results_dir / "training_summary.json").write_text(
        """
        {
          "status": "PARTIAL",
          "no_retraining_performed": true,
          "dataset_images": 10,
          "training_parameters": {
            "rank": {"value": 8, "source": "configs/lora_sd15.yaml"},
            "hardware": {"value": "gpu"},
            "output_dir": {"value": "D:/private/models"}
          }
        }
        """,
        encoding="utf-8",
    )
    (results_dir / "dataset_summary.json").write_text(
        """
        {
          "metadata_records": 10,
          "class_distribution": {"intact": 10},
          "notes": ["spotted se registra como categoria visual, no como diagnostico de hongo."]
        }
        """,
        encoding="utf-8",
    )

    evidence = load_lora_evidence(results_dir)

    assert evidence["available"] is True
    assert evidence["parameters"] == {"rank": 8}
    assert evidence["class_distribution"] == {"intact": 10}
    assert evidence["sample_names"] == ["intact_00001.jpg"]
    assert "private" not in str(evidence)


def test_sanitize_error_message_replaces_private_roots(tmp_path: Path) -> None:
    message = f"Fallo en {tmp_path / 'models' / 'checkpoint.pt'}"

    sanitized = sanitize_error_message(message, tmp_path)

    assert str(tmp_path) not in sanitized
    assert sanitized.startswith("Fallo en .")
