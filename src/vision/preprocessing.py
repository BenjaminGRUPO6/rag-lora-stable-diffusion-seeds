from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.vision.image_quality import VisualQuality, assess_visual_quality


ImageInput = str | Path | Image.Image
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for deterministic automatic seed cropping."""

    output_size: int = 224
    bbox_expansion_ratio: float = 0.12
    adaptive_window_ratio: float = 0.08
    min_component_area_ratio: float = 0.01
    max_component_count: int = 30
    min_object_side_pixels: int = 8
    morphology_iterations: int = 1

    def __post_init__(self) -> None:
        if not 0.08 <= self.bbox_expansion_ratio <= 0.15:
            raise ValueError("bbox_expansion_ratio debe estar entre 0.08 y 0.15.")
        if self.output_size <= 0:
            raise ValueError("output_size debe ser positivo.")


@dataclass(frozen=True)
class Component:
    """Connected foreground component."""

    label: int
    area: int
    bbox: BBox


@dataclass(frozen=True)
class PreprocessingResult:
    """Original, crop, mask and quality output from automatic preprocessing."""

    original: Image.Image
    crop: Image.Image
    mask: Image.Image
    bbox: BBox | None
    component_count: int
    used_fallback: bool
    fallback_reason: str | None
    quality: VisualQuality

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata summary without image pixels."""
        return {
            "bbox": self.bbox,
            "component_count": self.component_count,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "quality": self.quality.to_dict(),
        }


def preprocess_image(
    image: ImageInput,
    config: PreprocessingConfig | None = None,
    *,
    compute_quality: bool = True,
) -> PreprocessingResult:
    """Correct orientation, segment a candidate object and produce a safe crop."""
    active_config = config or PreprocessingConfig()
    original = load_oriented_rgb_image(image)
    if original.width <= 0 or original.height <= 0:
        return _fallback_result(
            original=Image.new("RGB", (active_config.output_size, active_config.output_size), "white"),
            mask=Image.new("L", (active_config.output_size, active_config.output_size), 0),
            config=active_config,
            reason="invalid_dimensions",
            component_count=0,
            compute_quality=compute_quality,
        )

    mask_array = detect_candidate_mask(original, active_config)
    components = connected_components(mask_array)
    significant = significant_components(
        components,
        image_size=original.size,
        min_area_ratio=active_config.min_component_area_ratio,
    )
    component_count = len(significant)
    mask_image = mask_to_image(mask_array)

    if not significant:
        return _fallback_result(
            original=original,
            mask=mask_image,
            config=active_config,
            reason="no_object_detected",
            component_count=component_count,
            compute_quality=compute_quality,
        )
    if component_count > active_config.max_component_count:
        return _fallback_result(
            original=original,
            mask=mask_image,
            config=active_config,
            reason="too_many_components",
            component_count=component_count,
            compute_quality=compute_quality,
        )

    component = max(significant, key=lambda item: item.area)
    if object_too_small(component, original.size, active_config):
        return _fallback_result(
            original=original,
            mask=mask_image,
            config=active_config,
            reason="object_too_small",
            component_count=component_count,
            compute_quality=compute_quality,
        )

    crop, crop_mask, crop_bbox = crop_square_with_padding(
        original=original,
        mask=mask_image,
        bbox=component.bbox,
        config=active_config,
    )
    if crop.width != active_config.output_size or crop.height != active_config.output_size:
        return _fallback_result(
            original=original,
            mask=mask_image,
            config=active_config,
            reason="invalid_crop",
            component_count=component_count,
            compute_quality=compute_quality,
        )

    quality = (
        assess_visual_quality(
            image=crop,
            mask=crop_mask,
            component_count=component_count,
            used_fallback=False,
        )
        if compute_quality
        else _skipped_quality(component_count=component_count)
    )
    return PreprocessingResult(
        original=original,
        crop=crop,
        mask=crop_mask,
        bbox=crop_bbox,
        component_count=component_count,
        used_fallback=False,
        fallback_reason=None,
        quality=quality,
    )


def load_oriented_rgb_image(image: ImageInput) -> Image.Image:
    """Load an image, apply EXIF orientation and return RGB pixels."""
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(Path(image)) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def detect_candidate_mask(image: Image.Image, config: PreprocessingConfig) -> np.ndarray:
    """Detect foreground candidates using border-color distance and adaptive thresholding."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if rgb.size == 0:
        return np.zeros((image.height, image.width), dtype=bool)

    border_color = estimate_border_color(rgb)
    color_delta = np.linalg.norm(rgb - border_color.reshape(1, 1, 3), axis=2) / 441.67295593
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    ) / 255.0
    border_gray = float(
        (0.299 * border_color[0] + 0.587 * border_color[1] + 0.114 * border_color[2])
        / 255.0
    )
    luminance_delta = np.abs(gray - border_gray)
    delta = np.maximum(color_delta, luminance_delta)

    window = adaptive_window(image.size, config.adaptive_window_ratio)
    local_mean = box_mean(delta, window)
    local_sq_mean = box_mean(delta * delta, window)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
    border_delta = border_values(delta)
    noise_floor = float(np.median(border_delta) + 2.5 * median_abs_deviation(border_delta))
    global_threshold = max(0.07, noise_floor)
    adaptive_threshold = local_mean + 0.35 * local_std
    mask = (delta > global_threshold) | (delta > adaptive_threshold)
    mask = close_binary(mask, iterations=config.morphology_iterations)
    mask = open_binary(mask, iterations=config.morphology_iterations)
    return mask


def connected_components(mask: np.ndarray) -> list[Component]:
    """Return 8-connected components from a binary mask."""
    if mask.ndim != 2 or mask.size == 0:
        return []
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[Component] = []
    label = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            label += 1
            area, bbox = flood_component(mask, visited, x, y)
            components.append(Component(label=label, area=area, bbox=bbox))
    return components


def flood_component(mask: np.ndarray, visited: np.ndarray, start_x: int, start_y: int) -> tuple[int, BBox]:
    """Flood-fill one 8-connected component."""
    height, width = mask.shape
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    visited[start_y, start_x] = True
    area = 0
    min_x = max_x = start_x
    min_y = max_y = start_y
    while queue:
        x, y = queue.popleft()
        area += 1
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((nx, ny))
    return area, (min_x, min_y, max_x + 1, max_y + 1)


def significant_components(
    components: list[Component],
    *,
    image_size: tuple[int, int],
    min_area_ratio: float,
) -> list[Component]:
    """Filter tiny connected components from candidate objects."""
    width, height = image_size
    min_area = max(4, int(round(width * height * min_area_ratio)))
    return [component for component in components if component.area >= min_area]


def object_too_small(
    component: Component,
    image_size: tuple[int, int],
    config: PreprocessingConfig,
) -> bool:
    """Return true when the principal component is too small for reliable cropping."""
    width, height = image_size
    x0, y0, x1, y1 = component.bbox
    component_width = x1 - x0
    component_height = y1 - y0
    area_ratio = component.area / float(width * height)
    return (
        area_ratio < config.min_component_area_ratio
        or component_width < config.min_object_side_pixels
        or component_height < config.min_object_side_pixels
    )


def crop_square_with_padding(
    *,
    original: Image.Image,
    mask: Image.Image,
    bbox: BBox,
    config: PreprocessingConfig,
) -> tuple[Image.Image, Image.Image, BBox]:
    """Expand a bbox, crop it as a square and pad outside image boundaries."""
    expanded = expanded_square_bbox(
        bbox=bbox,
        image_size=original.size,
        expansion_ratio=config.bbox_expansion_ratio,
    )
    crop = crop_with_padding(
        original,
        expanded,
        fill=border_fill_color(original),
    )
    crop_mask = crop_with_padding(mask, expanded, fill=0)
    crop = crop.resize((config.output_size, config.output_size), Image.Resampling.BICUBIC)
    crop_mask = crop_mask.resize((config.output_size, config.output_size), Image.Resampling.NEAREST)
    return crop.convert("RGB"), crop_mask.convert("L"), expanded


def expanded_square_bbox(
    *,
    bbox: BBox,
    image_size: tuple[int, int],
    expansion_ratio: float,
) -> BBox:
    """Return a square expanded bbox in source-image coordinates."""
    image_width, image_height = image_size
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    expanded_width = width * (1.0 + 2.0 * expansion_ratio)
    expanded_height = height * (1.0 + 2.0 * expansion_ratio)
    side = int(np.ceil(max(expanded_width, expanded_height)))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(np.floor(center_x - side / 2.0))
    top = int(np.floor(center_y - side / 2.0))
    right = left + side
    bottom = top + side
    if side <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("Crop invalido.")
    return (left, top, right, bottom)


def crop_with_padding(
    image: Image.Image,
    bbox: BBox,
    *,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    """Crop a bbox and pad missing regions when it extends outside the image."""
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("Crop invalido.")
    mode = image.mode
    canvas = Image.new(mode, (width, height), fill)
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        return canvas
    pasted = image.crop(source_box)
    canvas.paste(pasted, (source_box[0] - left, source_box[1] - top))
    return canvas


def fallback_crop(original: Image.Image, config: PreprocessingConfig) -> Image.Image:
    """Return a square padded resized version of the full original image."""
    side = max(original.width, original.height, 1)
    fill = border_fill_color(original)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(original.convert("RGB"), ((side - original.width) // 2, (side - original.height) // 2))
    return canvas.resize((config.output_size, config.output_size), Image.Resampling.BICUBIC)


def fallback_mask(original: Image.Image, config: PreprocessingConfig) -> Image.Image:
    """Return a zero mask matching the configured crop size."""
    _ = original
    return Image.new("L", (config.output_size, config.output_size), 0)


def estimate_border_color(rgb: np.ndarray) -> np.ndarray:
    """Estimate background color from image border pixels."""
    values = border_pixels(rgb)
    if values.size == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.median(values.reshape(-1, 3), axis=0).astype(np.float32)


def border_fill_color(image: Image.Image) -> tuple[int, int, int]:
    """Return an RGB padding color estimated from image borders."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = estimate_border_color(rgb)
    return tuple(int(np.clip(round(channel), 0, 255)) for channel in color)


def border_pixels(rgb: np.ndarray) -> np.ndarray:
    """Return outer-border RGB pixels from an image array."""
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        return np.empty((0, 3), dtype=np.float32)
    height, width, _ = rgb.shape
    border = max(1, min(height, width) // 20)
    top = rgb[:border, :, :]
    bottom = rgb[height - border :, :, :]
    left = rgb[:, :border, :]
    right = rgb[:, width - border :, :]
    return np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )


def border_values(values: np.ndarray) -> np.ndarray:
    """Return outer-border scalar values."""
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return np.array([0.0], dtype=np.float32)
    height, width = values.shape
    border = max(1, min(height, width) // 20)
    return np.concatenate(
        [
            values[:border, :].ravel(),
            values[height - border :, :].ravel(),
            values[:, :border].ravel(),
            values[:, width - border :].ravel(),
        ]
    )


def adaptive_window(image_size: tuple[int, int], ratio: float) -> int:
    """Return an odd local-window size for adaptive thresholding."""
    width, height = image_size
    window = max(3, int(round(min(width, height) * ratio)))
    if window % 2 == 0:
        window += 1
    return window


def box_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Calculate a reflected-border box mean using integral images."""
    radius = window // 2
    padded = np.pad(values, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return total / float(window * window)


def median_abs_deviation(values: np.ndarray) -> float:
    """Return median absolute deviation for a scalar array."""
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def dilate_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask with a 3x3 structuring element."""
    result = mask.astype(bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.zeros_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                result |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return result


def erode_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode a binary mask with a 3x3 structuring element."""
    result = mask.astype(bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        result = eroded
    return result


def close_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Close small gaps in a binary mask."""
    return erode_binary(dilate_binary(mask, iterations), iterations)


def open_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Remove isolated binary mask noise."""
    return dilate_binary(erode_binary(mask, iterations), iterations)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    """Convert a binary mask to an 8-bit PIL image."""
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _fallback_result(
    *,
    original: Image.Image,
    mask: Image.Image,
    config: PreprocessingConfig,
    reason: str,
    component_count: int,
    compute_quality: bool = True,
) -> PreprocessingResult:
    crop = fallback_crop(original, config)
    empty_mask = fallback_mask(original, config)
    quality = (
        assess_visual_quality(
            image=crop,
            mask=empty_mask,
            component_count=component_count,
            used_fallback=True,
            fallback_reason=reason,
        )
        if compute_quality
        else _skipped_quality(
            component_count=component_count,
            warnings=[f"fallback_preprocessing:{reason}"],
        )
    )
    return PreprocessingResult(
        original=original,
        crop=crop,
        mask=mask.resize((config.output_size, config.output_size), Image.Resampling.NEAREST).convert("L"),
        bbox=None,
        component_count=component_count,
        used_fallback=True,
        fallback_reason=reason,
        quality=quality,
    )


def _skipped_quality(
    *,
    component_count: int,
    warnings: list[str] | None = None,
) -> VisualQuality:
    """Return a cheap placeholder when quality metrics are disabled for training."""
    return VisualQuality(
        blur_score=0.0,
        brightness_score=0.0,
        contrast_score=0.0,
        foreground_ratio=0.0,
        component_count=int(component_count),
        crop_confidence=0.0,
        warnings=warnings or ["quality_not_computed"],
    )
