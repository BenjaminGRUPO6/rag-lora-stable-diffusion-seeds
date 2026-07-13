from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualQuality:
    """Deterministic visual quality summary for a preprocessing result."""

    blur_score: float
    brightness_score: float
    contrast_score: float
    foreground_ratio: float
    component_count: int
    crop_confidence: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def assess_visual_quality(
    *,
    image: Image.Image,
    mask: Image.Image | None,
    component_count: int,
    used_fallback: bool,
    fallback_reason: str | None = None,
) -> VisualQuality:
    """Calculate lightweight visual quality metrics without diagnostic claims."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    blur_score = _laplacian_variance(gray)
    brightness_score = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))
    contrast_score = float(np.clip(gray.std() / 80.0, 0.0, 1.0))
    foreground_ratio = _foreground_ratio(mask)
    warnings = build_quality_warnings(
        blur_score=blur_score,
        brightness_score=brightness_score,
        contrast_score=contrast_score,
        foreground_ratio=foreground_ratio,
        component_count=component_count,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
    )
    crop_confidence = estimate_crop_confidence(
        foreground_ratio=foreground_ratio,
        component_count=component_count,
        used_fallback=used_fallback,
        warnings=warnings,
    )
    return VisualQuality(
        blur_score=round(blur_score, 6),
        brightness_score=round(brightness_score, 6),
        contrast_score=round(contrast_score, 6),
        foreground_ratio=round(foreground_ratio, 6),
        component_count=int(component_count),
        crop_confidence=round(crop_confidence, 6),
        warnings=warnings,
    )


def build_quality_warnings(
    *,
    blur_score: float,
    brightness_score: float,
    contrast_score: float,
    foreground_ratio: float,
    component_count: int,
    used_fallback: bool,
    fallback_reason: str | None,
) -> list[str]:
    """Return human-review warnings for image quality and crop reliability."""
    warnings: list[str] = []
    if used_fallback:
        reason = fallback_reason or "fallback"
        warnings.append(f"fallback_preprocessing:{reason}")
    if blur_score < 70.0:
        warnings.append("possible_blur")
    if brightness_score < 0.18:
        warnings.append("low_brightness")
    if brightness_score > 0.92:
        warnings.append("high_brightness")
    if contrast_score < 0.08:
        warnings.append("low_contrast")
    if foreground_ratio < 0.03:
        warnings.append("low_foreground_ratio")
    if foreground_ratio > 0.88:
        warnings.append("high_foreground_ratio")
    if component_count > 1:
        warnings.append("multiple_candidate_components")
    return warnings


def estimate_crop_confidence(
    *,
    foreground_ratio: float,
    component_count: int,
    used_fallback: bool,
    warnings: list[str],
) -> float:
    """Estimate crop confidence from deterministic segmentation signals."""
    confidence = 1.0
    if used_fallback:
        confidence -= 0.45
    if foreground_ratio < 0.05:
        confidence -= 0.30
    elif foreground_ratio < 0.12:
        confidence -= 0.12
    elif foreground_ratio > 0.80:
        confidence -= 0.18
    if component_count > 1:
        confidence -= min(0.30, 0.04 * float(component_count - 1))
    confidence -= min(0.25, 0.03 * float(len(warnings)))
    return float(np.clip(confidence, 0.0, 1.0))


def _foreground_ratio(mask: Image.Image | None) -> float:
    if mask is None:
        return 0.0
    values = np.asarray(mask.convert("L"), dtype=np.uint8)
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values > 0) / values.size)


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1] * 4.0
    laplacian = (
        center
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(np.var(laplacian))
