from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageRecord:
    path: str
    category: str
    extension: str
    width: int
    height: int
    mode: str
    size_bytes: int
    sha256: str
    phash: str
    is_small: bool


def iter_image_paths(dataset_dir: Path) -> Iterable[Path]:
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path, dataset_dir: Path, min_size: int = 256) -> ImageRecord:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        phash = str(imagehash.phash(image.convert("RGB")))
        mode = image.mode
    relative = path.relative_to(dataset_dir)
    category = relative.parts[0] if len(relative.parts) > 1 else "unclassified"
    return ImageRecord(
        path=relative.as_posix(),
        category=category,
        extension=path.suffix.lower(),
        width=width,
        height=height,
        mode=mode,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        phash=phash,
        is_small=min(width, height) < min_size,
    )


def audit_dataset(dataset_dir: str | Path, min_size: int = 256) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(dataset_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"No existe el dataset: {root}")

    valid: list[dict] = []
    corrupted: list[dict] = []
    for path in iter_image_paths(root):
        try:
            valid.append(asdict(inspect_image(path, root, min_size=min_size)))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupted.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})

    return pd.DataFrame(valid), pd.DataFrame(corrupted)


def summarize_images(images: pd.DataFrame, corrupted: pd.DataFrame) -> dict:
    if images.empty:
        return {
            "total_valid": 0,
            "total_corrupted": len(corrupted),
            "small_images": 0,
            "exact_duplicate_files": 0,
            "categories": {},
            "extensions": {},
        }

    exact_duplicate_files = int(images.duplicated("sha256", keep=False).sum())
    return {
        "total_valid": int(len(images)),
        "total_corrupted": int(len(corrupted)),
        "small_images": int(images["is_small"].sum()),
        "exact_duplicate_files": exact_duplicate_files,
        "categories": images["category"].value_counts().to_dict(),
        "extensions": images["extension"].value_counts().to_dict(),
        "width": {
            "min": int(images["width"].min()),
            "max": int(images["width"].max()),
            "mean": float(images["width"].mean()),
        },
        "height": {
            "min": int(images["height"].min()),
            "max": int(images["height"].max()),
            "mean": float(images["height"].mean()),
        },
    }
