from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.synthetic_data.captions import (
    DEFAULT_TRIGGER_WORD,
    EXPECTED_CLASSES,
    build_caption,
    validate_caption_templates,
)


SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
REPORT_COLUMNS: tuple[str, ...] = (
    "source_path",
    "output_path",
    "label",
    "caption",
    "sha256",
    "width_original",
    "height_original",
    "selected",
    "rejection_reason",
)


@dataclass(frozen=True)
class LoraDatasetSettings:
    """Configuration required to prepare a Stable Diffusion LoRA image dataset."""

    source_dir: Path
    output_dir: Path
    metadata_file: str
    trigger_word: str
    images_per_class: int
    seed: int
    resolution: int
    classes: tuple[str, ...]


@dataclass(frozen=True)
class LoraPreparationResult:
    """Summary returned by LoRA dataset preparation."""

    output_dir: Path
    metadata_path: Path
    report_path: Path
    selected_counts: dict[str, int]
    candidate_counts: dict[str, int]
    dry_run: bool


def settings_from_config(
    config: dict[str, Any],
    project_root: Path,
    images_per_class: int | None = None,
    seed: int | None = None,
) -> LoraDatasetSettings:
    """Create preparation settings from the LoRA YAML configuration."""

    dataset = config.get("dataset", {})
    model = config.get("model", {})
    classes = tuple(config.get("classes", EXPECTED_CLASSES))
    return LoraDatasetSettings(
        source_dir=_resolve_repo_path(project_root, Path(dataset["source_dir"])),
        output_dir=_resolve_repo_path(project_root, Path(dataset["output_dir"])),
        metadata_file=str(dataset.get("metadata_file", "metadata.jsonl")),
        trigger_word=str(dataset.get("trigger_word", DEFAULT_TRIGGER_WORD)),
        images_per_class=int(images_per_class or dataset.get("images_per_class", 200)),
        seed=int(seed if seed is not None else dataset.get("seed", 42)),
        resolution=int(model.get("resolution", 512)),
        classes=classes,
    )


def prepare_lora_dataset(
    settings: LoraDatasetSettings,
    project_root: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> LoraPreparationResult:
    """Prepare a LoRA dataset from data/processed/train without modifying source images."""

    project_root = project_root.resolve()
    _validate_settings(settings, project_root=project_root)
    validate_caption_templates(settings.classes, trigger_word=settings.trigger_word)

    candidates = _collect_candidates(settings.source_dir, settings.classes)
    selected = _select_candidates(
        candidates,
        classes=settings.classes,
        images_per_class=settings.images_per_class,
        seed=settings.seed,
    )
    selected_paths = {path for paths in selected.values() for path in paths}

    candidate_counts = {label: len(candidates[label]) for label in settings.classes}
    selected_counts = {label: len(selected[label]) for label in settings.classes}
    metadata_path = settings.output_dir / settings.metadata_file
    report_path = settings.output_dir / "selection_report.csv"

    if dry_run:
        return LoraPreparationResult(
            output_dir=settings.output_dir,
            metadata_path=metadata_path,
            report_path=report_path,
            selected_counts=selected_counts,
            candidate_counts=candidate_counts,
            dry_run=True,
        )

    _prepare_output_dir(settings.output_dir, overwrite=overwrite)

    metadata_records: list[dict[str, str]] = []
    report_rows: list[dict[str, object]] = []
    output_index_by_class = {label: 0 for label in settings.classes}

    for label in settings.classes:
        caption = build_caption(label, trigger_word=settings.trigger_word)
        for source_path in candidates[label]:
            is_selected = source_path in selected_paths
            output_path = ""
            sha256 = ""
            width_original = ""
            height_original = ""
            rejection_reason = "" if is_selected else "not_selected_seed_limit"

            if is_selected:
                _ensure_selected_path_is_train_only(source_path, settings.source_dir, project_root)
                output_index_by_class[label] += 1
                output_relative = Path("images") / f"{label}_{output_index_by_class[label]:05d}.jpg"
                destination = settings.output_dir / output_relative
                width_original, height_original = _copy_as_rgb_padded_jpeg(
                    source_path=source_path,
                    destination=destination,
                    resolution=settings.resolution,
                )
                sha256 = compute_sha256(destination)
                output_path = _relative_posix(destination, project_root)
                metadata_records.append(
                    {
                        "file_name": output_relative.as_posix(),
                        "text": caption,
                    }
                )

            report_rows.append(
                {
                    "source_path": _relative_posix(source_path, project_root),
                    "output_path": output_path,
                    "label": label,
                    "caption": caption,
                    "sha256": sha256,
                    "width_original": width_original,
                    "height_original": height_original,
                    "selected": is_selected,
                    "rejection_reason": rejection_reason,
                }
            )

    _write_metadata(metadata_path, metadata_records)
    _write_selection_report(report_path, report_rows)

    return LoraPreparationResult(
        output_dir=settings.output_dir,
        metadata_path=metadata_path,
        report_path=report_path,
        selected_counts=selected_counts,
        candidate_counts=candidate_counts,
        dry_run=False,
    )


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_settings(settings: LoraDatasetSettings, project_root: Path) -> None:
    """Validate paths, class names, and numeric settings before selection."""

    if settings.classes != EXPECTED_CLASSES:
        raise ValueError(f"Expected classes {EXPECTED_CLASSES}, got {settings.classes}")
    if settings.images_per_class < 1:
        raise ValueError("images_per_class must be at least 1.")
    if settings.resolution != 512:
        raise ValueError("This LoRA preparation step expects 512 pixel output resolution.")
    if not settings.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {settings.source_dir}")

    expected_train_dir = (project_root / "data" / "processed" / "train").resolve()
    if settings.source_dir.resolve() != expected_train_dir:
        raise ValueError(f"LoRA preparation must read only from: {expected_train_dir}")

    missing_dirs = [label for label in settings.classes if not (settings.source_dir / label).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing class directories in train split: {missing_dirs}")


def _collect_candidates(source_dir: Path, classes: tuple[str, ...]) -> dict[str, list[Path]]:
    """Collect supported image files from each class directory."""

    candidates: dict[str, list[Path]] = {}
    for label in classes:
        class_dir = source_dir / label
        class_candidates = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not class_candidates:
            raise ValueError(f"No image candidates found for class: {label}")
        candidates[label] = class_candidates
    return candidates


def _select_candidates(
    candidates: dict[str, list[Path]],
    classes: tuple[str, ...],
    images_per_class: int,
    seed: int,
) -> dict[str, list[Path]]:
    """Select a deterministic random subset per class."""

    selected: dict[str, list[Path]] = {}
    for label in classes:
        rng = random.Random(f"{seed}:{label}")
        class_candidates = list(candidates[label])
        rng.shuffle(class_candidates)
        selected[label] = sorted(class_candidates[:images_per_class])
    return selected


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create an output directory and guard existing generated files."""

    images_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.jsonl"
    report_path = output_dir / "selection_report.csv"
    generated_paths = [images_dir, metadata_path, report_path]
    existing = [path for path in generated_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "LoRA output already exists. Use --overwrite to replace generated files: "
            f"{[str(path) for path in existing]}"
        )

    if overwrite:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        for file_path in (metadata_path, report_path):
            if file_path.exists():
                file_path.unlink()

    images_dir.mkdir(parents=True, exist_ok=True)


def _copy_as_rgb_padded_jpeg(source_path: Path, destination: Path, resolution: int) -> tuple[int, int]:
    """Copy an image as RGB JPEG with white padding while preserving aspect ratio."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        width_original, height_original = image.size
        rgb_image = image.convert("RGB")
        rgb_image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (resolution, resolution), color=(255, 255, 255))
        offset = (
            (resolution - rgb_image.width) // 2,
            (resolution - rgb_image.height) // 2,
        )
        canvas.paste(rgb_image, offset)
        canvas.save(destination, format="JPEG", quality=95)
    return width_original, height_original


def _write_metadata(metadata_path: Path, records: list[dict[str, str]]) -> None:
    """Write DreamBooth-compatible metadata JSONL records."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_selection_report(report_path: Path, rows: list[dict[str, object]]) -> None:
    """Write the LoRA selection report CSV."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_selected_path_is_train_only(source_path: Path, train_dir: Path, project_root: Path) -> None:
    """Ensure a selected source path belongs to train and not validation or test."""

    resolved_source = source_path.resolve()
    resolved_train = train_dir.resolve()
    if not _is_relative_to(resolved_source, resolved_train):
        raise ValueError(f"Selected image is outside train: {source_path}")

    for split in ("validation", "test"):
        forbidden_dir = (project_root / "data" / "processed" / split).resolve()
        if forbidden_dir.exists() and _is_relative_to(resolved_source, forbidden_dir):
            raise ValueError(f"Selected image cannot come from {split}: {source_path}")


def _resolve_repo_path(project_root: Path, path: Path) -> Path:
    """Resolve a config path against the repository root."""

    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _relative_posix(path: Path, project_root: Path) -> str:
    """Return a repository-relative POSIX path when possible."""

    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(project_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is contained by parent for Python 3.10 compatibility."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
