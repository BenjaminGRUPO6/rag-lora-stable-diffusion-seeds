from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.vision.preprocessing import (
    PreprocessingConfig,
    connected_components,
    preprocess_image,
)


def make_seed_image(mode: str = "RGB", size: tuple[int, int] = (96, 80)) -> Image.Image:
    """Create a synthetic soybean-like object on a uniform background."""
    image = Image.new("RGB", size, (245, 245, 240))
    draw = ImageDraw.Draw(image)
    draw.ellipse((28, 18, 68, 58), fill=(142, 110, 62))
    if mode == "RGBA":
        return image.convert("RGBA")
    return image.convert(mode)


def test_preprocess_rgb_image_returns_crop_mask_and_quality() -> None:
    result = preprocess_image(make_seed_image("RGB"), PreprocessingConfig(output_size=64))

    assert result.original.mode == "RGB"
    assert result.crop.mode == "RGB"
    assert result.crop.size == (64, 64)
    assert result.mask.mode == "L"
    assert result.used_fallback is False
    assert result.bbox is not None
    assert 0.0 <= result.quality.crop_confidence <= 1.0


def test_preprocess_rgba_image_converts_to_rgb() -> None:
    result = preprocess_image(make_seed_image("RGBA"), PreprocessingConfig(output_size=48))

    assert result.original.mode == "RGB"
    assert result.crop.mode == "RGB"
    assert result.crop.size == (48, 48)


def test_small_image_uses_fallback() -> None:
    image = Image.new("RGB", (5, 5), (245, 245, 245))
    image.putpixel((2, 2), (90, 70, 50))

    result = preprocess_image(image, PreprocessingConfig(output_size=32))

    assert result.used_fallback is True
    assert result.fallback_reason in {"no_object_detected", "object_too_small"}
    assert result.crop.size == (32, 32)


def test_empty_uniform_background_uses_fallback() -> None:
    result = preprocess_image(
        Image.new("RGB", (64, 64), (250, 250, 250)),
        PreprocessingConfig(output_size=32),
    )

    assert result.used_fallback is True
    assert result.fallback_reason == "no_object_detected"
    assert "fallback_preprocessing:no_object_detected" in result.quality.warnings


def test_foreground_on_uniform_background_is_detected() -> None:
    result = preprocess_image(make_seed_image("RGB"), PreprocessingConfig(output_size=64))

    assert result.used_fallback is False
    assert result.quality.foreground_ratio > 0.05


def test_connected_components_detects_multiple_objects() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:4, 1:4] = True
    mask[7:10, 7:10] = True

    components = connected_components(mask)

    assert len(components) == 2
    assert sorted(component.area for component in components) == [9, 9]


def test_too_many_components_uses_fallback() -> None:
    image = Image.new("RGB", (120, 120), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for x, y in [(15, 15), (75, 15), (15, 75), (75, 75)]:
        draw.ellipse((x, y, x + 18, y + 18), fill=(120, 90, 50))

    result = preprocess_image(
        image,
        PreprocessingConfig(
            output_size=64,
            max_component_count=2,
            min_component_area_ratio=0.005,
        ),
    )

    assert result.used_fallback is True
    assert result.fallback_reason == "too_many_components"
    assert result.component_count > 2


def test_windows_style_path_input(tmp_path: Path) -> None:
    image_path = tmp_path / "nested" / "seed_image.png"
    image_path.parent.mkdir()
    make_seed_image("RGB").save(image_path)

    result = preprocess_image(str(image_path), PreprocessingConfig(output_size=40))

    assert result.crop.size == (40, 40)
    assert result.original.mode == "RGB"


def test_invalid_expansion_ratio_is_rejected() -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(bbox_expansion_ratio=0.20)
