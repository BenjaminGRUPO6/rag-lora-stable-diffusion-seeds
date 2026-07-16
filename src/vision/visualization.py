from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RGBColor = tuple[int, int, int]


def normalize_intensity(values: np.ndarray) -> np.ndarray:
    """Return a finite 0-1 map suitable for visual display."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def heatmap_to_image(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Convert a normalized heatmap into an RGB PIL image."""
    normalized = normalize_intensity(heatmap)
    image = Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L")
    image = image.resize(size, Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32) / 255.0
    red = np.clip(255.0 * np.minimum(1.0, values * 1.7), 0, 255)
    green = np.clip(255.0 * (1.0 - np.abs(values - 0.55) * 1.8), 0, 255)
    blue = np.clip(255.0 * np.maximum(0.0, 1.0 - values * 1.8), 0, 255)
    rgb = np.stack([red, green, blue], axis=2).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.42,
) -> Image.Image:
    """Blend a heatmap over an RGB image."""
    base = image.convert("RGB")
    colorized = heatmap_to_image(heatmap, base.size)
    return Image.blend(base, colorized, alpha=max(0.0, min(1.0, alpha)))


def fallback_heatmap(size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Return a neutral Grad-CAM fallback map."""
    width, height = size
    return np.zeros((max(1, height), max(1, width)), dtype=np.float32)


def build_combined_gradcam_image(
    *,
    original: Image.Image,
    crop: Image.Image,
    heatmap: Image.Image,
    overlay: Image.Image,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> Image.Image:
    """Build a compact four-panel visual explanation image."""
    panel_size = (224, 224)
    images = [
        original.convert("RGB").resize(panel_size, Image.Resampling.BICUBIC),
        crop.convert("RGB").resize(panel_size, Image.Resampling.BICUBIC),
        heatmap.convert("RGB").resize(panel_size, Image.Resampling.BICUBIC),
        overlay.convert("RGB").resize(panel_size, Image.Resampling.BICUBIC),
    ]
    labels = ["Original", "Crop", "Grad-CAM", "Overlay"]
    padding = 18
    label_height = 30
    footer_height = 70
    width = panel_size[0] * 4 + padding * 5
    height = panel_size[1] + label_height + footer_height + padding * 3
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((padding, padding), title, fill=(17, 24, 39), font=font)
    top = padding + label_height
    for index, image in enumerate(images):
        left = padding + index * (panel_size[0] + padding)
        canvas.paste(image, (left, top))
        draw.text((left, top + panel_size[1] + 6), labels[index], fill=(31, 41, 55), font=font)
    if metadata:
        footer = " | ".join(f"{key}: {value}" for key, value in metadata.items())
        draw.text((padding, height - footer_height + 18), footer[:160], fill=(75, 85, 99), font=font)
    return canvas


def build_probability_panel_image(
    *,
    original: Image.Image,
    crop: Image.Image,
    overlay: Image.Image,
    probabilities: dict[str, float],
    title: str,
    metadata: dict[str, Any],
) -> Image.Image:
    """Build a visual demo panel with images and horizontal probability bars."""
    thumb_size = (210, 210)
    width = 960
    height = 420
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((24, 18), title, fill=(17, 24, 39), font=font)
    images = [
        ("Original", original),
        ("Crop", crop),
        ("Grad-CAM", overlay),
    ]
    for index, (label, image) in enumerate(images):
        left = 24 + index * 230
        top = 58
        canvas.paste(image.convert("RGB").resize(thumb_size, Image.Resampling.BICUBIC), (left, top))
        draw.text((left, top + thumb_size[1] + 8), label, fill=(31, 41, 55), font=font)

    bar_left = 720
    bar_top = 74
    bar_width = 200
    ordered = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
    draw.text((bar_left, 48), "Probabilidades calibradas", fill=(31, 41, 55), font=font)
    for index, (label, value) in enumerate(ordered[:5]):
        y = bar_top + index * 42
        draw.text((bar_left, y), str(label), fill=(31, 41, 55), font=font)
        draw.rectangle((bar_left, y + 14, bar_left + bar_width, y + 26), fill=(229, 231, 235))
        draw.rectangle(
            (bar_left, y + 14, bar_left + int(bar_width * float(value)), y + 26),
            fill=(37, 99, 235),
        )
        draw.text((bar_left + bar_width + 8, y + 10), f"{float(value):.3f}", fill=(31, 41, 55), font=font)

    footer = " | ".join(f"{key}: {value}" for key, value in metadata.items())
    draw.text((24, 360), footer[:180], fill=(75, 85, 99), font=font)
    draw.text(
        (24, 382),
        "Grad-CAM es una explicacion aproximada de atencion del modelo, no una prueba causal ni diagnostico.",
        fill=(75, 85, 99),
        font=font,
    )
    return canvas


def build_image_grid(
    items: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    columns: int = 3,
) -> None:
    """Save a grid of Grad-CAM combined images."""
    if not items:
        empty = Image.new("RGB", (900, 260), "white")
        draw = ImageDraw.Draw(empty)
        draw.text((24, 24), title, fill=(17, 24, 39), font=ImageFont.load_default())
        draw.text((24, 60), "No hay ejemplos disponibles.", fill=(75, 85, 99), font=ImageFont.load_default())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.save(output_path)
        return

    thumbs = [
        item["combined"].convert("RGB").resize((420, 170), Image.Resampling.BICUBIC)
        for item in items
    ]
    rows = int(np.ceil(len(thumbs) / float(columns)))
    padding = 18
    header = 46
    width = columns * 420 + padding * (columns + 1)
    height = header + rows * 170 + padding * (rows + 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 18), title, fill=(17, 24, 39), font=ImageFont.load_default())
    for index, thumb in enumerate(thumbs):
        row = index // columns
        column = index % columns
        left = padding + column * (420 + padding)
        top = header + padding + row * (170 + padding)
        canvas.paste(thumb, (left, top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
