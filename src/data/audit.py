from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageRecord:
    relative_path: str
    category: str
    extension: str
    width: int | None
    height: int | None
    mode: str | None
    size_bytes: int
    sha256: str | None
    perceptual_hash: str | None
    valid: bool
    error: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category_from_relative(relative_path: Path) -> str:
    """Return the first directory name as the visual category."""
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"


def _iter_image_paths(root: Path) -> list[Path]:
    """Find supported image files under root without modifying them."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def audit_images(root: Path) -> list[ImageRecord]:
    """Audit supported image files and return structured records."""
    records: list[ImageRecord] = []
    for path in _iter_image_paths(root):
        relative = path.relative_to(root)
        category = _category_from_relative(relative)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                phash = str(imagehash.phash(image.convert("RGB")))
            records.append(
                ImageRecord(
                    str(relative),
                    category,
                    path.suffix.lower(),
                    width,
                    height,
                    mode,
                    path.stat().st_size,
                    _sha256(path),
                    phash,
                    True,
                    None,
                )
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            records.append(
                ImageRecord(
                    str(relative),
                    category,
                    path.suffix.lower(),
                    None,
                    None,
                    None,
                    path.stat().st_size,
                    None,
                    None,
                    False,
                    str(exc),
                )
            )
    return records


def summarize(records: list[ImageRecord]) -> dict:
    """Summarize audited image records for JSON reporting."""
    valid = [r for r in records if r.valid]
    sha_counts = Counter(r.sha256 for r in valid if r.sha256)
    phash_groups: dict[str, list[str]] = defaultdict(list)
    for record in valid:
        if record.perceptual_hash:
            phash_groups[record.perceptual_hash].append(record.relative_path)
    return {
        'total_files': len(records),
        'valid_images': len(valid),
        'corrupted_images': len(records) - len(valid),
        'exact_duplicate_groups': sum(1 for count in sha_counts.values() if count > 1),
        'same_phash_groups': sum(1 for paths in phash_groups.values() if len(paths) > 1),
        'category_distribution': dict(Counter(r.category for r in valid)),
        'extension_distribution': dict(Counter(r.extension for r in valid)),
        'width_min': min((r.width for r in valid if r.width is not None), default=None),
        'width_max': max((r.width for r in valid if r.width is not None), default=None),
        'height_min': min((r.height for r in valid if r.height is not None), default=None),
        'height_max': max((r.height for r in valid if r.height is not None), default=None),
    }


def audit_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit a dataset directory and split valid and corrupted image rows.

    The function reads image metadata, exact hashes and perceptual hashes from
    files with supported extensions. It does not write to, move or delete any
    dataset file.
    """
    records = audit_images(root)
    rows = [asdict(record) for record in records]
    columns = [field.name for field in fields(ImageRecord)]
    images = pd.DataFrame(
        [row for row in rows if row["valid"]],
        columns=columns,
    )
    corrupted = pd.DataFrame(
        [row for row in rows if not row["valid"]],
        columns=columns,
    )
    return images.reset_index(drop=True), corrupted.reset_index(drop=True)


def summarize_images(images: pd.DataFrame, corrupted: pd.DataFrame) -> dict:
    """Build a dataset audit summary from valid and corrupted image tables."""
    sha_counts = Counter(images["sha256"].dropna()) if "sha256" in images else Counter()
    phash_counts = (
        Counter(images["perceptual_hash"].dropna())
        if "perceptual_hash" in images
        else Counter()
    )

    summary = {
        "total_valid": int(len(images)),
        "total_corrupted": int(len(corrupted)),
        "total_files": int(len(images) + len(corrupted)),
        "exact_duplicate_groups": sum(1 for count in sha_counts.values() if count > 1),
        "same_phash_groups": sum(1 for count in phash_counts.values() if count > 1),
        "category_distribution": _value_counts(images, "category"),
        "extension_distribution": _value_counts(images, "extension"),
        "resolution_distribution": _resolution_counts(images),
        "width_min": _numeric_min(images, "width"),
        "width_max": _numeric_max(images, "width"),
        "height_min": _numeric_min(images, "height"),
        "height_max": _numeric_max(images, "height"),
    }
    return summary


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return deterministic value counts for a DataFrame column."""
    if column not in frame or frame.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().sort_index().items()
    }


def _resolution_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count image resolutions as WIDTHxHEIGHT strings."""
    if frame.empty or "width" not in frame or "height" not in frame:
        return {}
    resolutions = frame.dropna(subset=["width", "height"]).copy()
    if resolutions.empty:
        return {}
    labels = resolutions.apply(
        lambda row: f"{int(row['width'])}x{int(row['height'])}",
        axis=1,
    )
    return {str(key): int(value) for key, value in labels.value_counts().sort_index().items()}


def _numeric_min(frame: pd.DataFrame, column: str) -> int | None:
    """Return the integer minimum for a numeric column, if present."""
    if column not in frame or frame[column].dropna().empty:
        return None
    return int(frame[column].min())


def _numeric_max(frame: pd.DataFrame, column: str) -> int | None:
    """Return the integer maximum for a numeric column, if present."""
    if column not in frame or frame[column].dropna().empty:
        return None
    return int(frame[column].max())
