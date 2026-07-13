from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.compare_inference_paths import SelectedImage, compare_paths, select_validation_images


class FakeEngine:
    """Small deterministic engine for parity script tests."""

    labels = ["intact", "spotted", "immature", "broken", "skin_damaged"]

    def predict_dict(self, image: object) -> dict[str, Any]:
        return {
            "label": "intact",
            "confidence": 0.7,
            "probabilities": {
                "intact": 0.7,
                "spotted": 0.1,
                "immature": 0.08,
                "broken": 0.07,
                "skin_damaged": 0.05,
            },
            "logits": {
                "intact": 2.0,
                "spotted": 0.1,
                "immature": 0.0,
                "broken": -0.1,
                "skin_damaged": -0.2,
            },
            "top_3": [
                {"label": "intact", "probability": 0.7},
                {"label": "spotted", "probability": 0.1},
                {"label": "immature", "probability": 0.08},
            ],
        }


def test_select_validation_images_is_reproducible(tmp_path: Path) -> None:
    validation_dir = tmp_path / "validation"
    for class_name in ("intact", "spotted"):
        class_dir = validation_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(8):
            (class_dir / f"{class_name}_{index}.jpg").write_text("", encoding="utf-8")

    first = select_validation_images(
        validation_dir=validation_dir,
        class_names=("intact", "spotted"),
        images_per_class=5,
        seed=42,
    )
    second = select_validation_images(
        validation_dir=validation_dir,
        class_names=("intact", "spotted"),
        images_per_class=5,
        seed=42,
    )

    assert [item.image_path.name for item in first] == [item.image_path.name for item in second]
    assert [item.true_class for item in first].count("intact") == 5
    assert [item.true_class for item in first].count("spotted") == 5


def test_select_validation_images_requires_enough_images(tmp_path: Path) -> None:
    class_dir = tmp_path / "validation" / "intact"
    class_dir.mkdir(parents=True)
    (class_dir / "one.jpg").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No hay suficientes imagenes"):
        select_validation_images(
            validation_dir=tmp_path / "validation",
            class_names=("intact",),
            images_per_class=5,
            seed=42,
        )


def test_compare_paths_matches_direct_engine_and_pipeline(tmp_path: Path) -> None:
    vision_config = tmp_path / "vision.yaml"
    rag_config = tmp_path / "rag.yaml"
    image_path = tmp_path / "seed.jpg"
    vision_config.write_text(
        "\n".join(
            [
                "data:",
                "  image_size: 224",
                "inference:",
                "  confidence_threshold: 0.60",
                "  margin_threshold: 0.15",
            ]
        ),
        encoding="utf-8",
    )
    rag_config.write_text("rag:\n  top_k: 1\n", encoding="utf-8")
    image_path.write_text("", encoding="utf-8")

    rows = compare_paths(
        selected_images=[
            SelectedImage(
                image_id="seed",
                true_class="intact",
                image_path=image_path,
            )
        ],
        engine=FakeEngine(),  # type: ignore[arg-type]
        vision_config_path=vision_config,
        rag_config_path=rag_config,
        labels=FakeEngine.labels,
        device_name="cpu",
    )

    assert rows[0]["class_match"] is True
    assert rows[0]["top_3_match"] is True
    assert rows[0]["max_probability_abs_diff"] == pytest.approx(0.0)
    assert rows[0]["max_logit_abs_diff"] == pytest.approx(0.0)
