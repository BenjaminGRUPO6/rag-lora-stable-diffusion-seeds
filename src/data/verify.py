from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EXPECTED_CLASSES = ("intact", "spotted", "immature", "broken", "skin_damaged")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetVerification:
    counts: dict[str, int]
    missing_classes: tuple[str, ...]
    unexpected_directories: tuple[str, ...]
    extensions: tuple[str, ...]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def valid(self) -> bool:
        return not self.missing_classes


def verify_dataset(root: Path) -> DatasetVerification:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    directories = sorted(p.name for p in root.iterdir() if p.is_dir())
    missing = tuple(name for name in EXPECTED_CLASSES if name not in directories)
    unexpected = tuple(name for name in directories if name not in EXPECTED_CLASSES)
    counts: dict[str, int] = {}
    extensions: set[str] = set()

    for class_name in EXPECTED_CLASSES:
        class_dir = root / class_name
        if not class_dir.exists():
            counts[class_name] = 0
            continue
        images = [p for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        counts[class_name] = len(images)
        extensions.update(p.suffix.lower() for p in images)

    return DatasetVerification(
        counts=counts,
        missing_classes=missing,
        unexpected_directories=unexpected,
        extensions=tuple(sorted(extensions)),
    )
