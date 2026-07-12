from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
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
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_images(root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
        relative = path.relative_to(root)
        category = relative.parts[0] if len(relative.parts) > 1 else 'unknown'
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                phash = str(imagehash.phash(image.convert('RGB')))
            records.append(ImageRecord(str(relative), category, path.suffix.lower(), width, height, mode, path.stat().st_size, _sha256(path), phash, True, None))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            records.append(ImageRecord(str(relative), category, path.suffix.lower(), None, None, None, path.stat().st_size, None, None, False, str(exc)))
    return records


def summarize(records: list[ImageRecord]) -> dict:
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
