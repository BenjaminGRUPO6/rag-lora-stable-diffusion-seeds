from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.synthetic_data.captions import BANNED_DIAGNOSTIC_TERMS, EXPECTED_CLASSES
from src.synthetic_data.prepare_lora_dataset import compute_sha256, settings_from_config


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for validating a prepared LoRA dataset."""

    parser = argparse.ArgumentParser(description="Validate the prepared SD 1.5 LoRA dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs") / "lora_sd15_config.yaml")
    return parser.parse_args(argv)


def validate_lora_dataset(config_path: Path, project_root: Path = Path.cwd()) -> list[str]:
    """Validate LoRA metadata, images, hashes, captions, and train-only provenance."""

    config = _load_config(config_path)
    settings = settings_from_config(config, project_root=project_root)
    metadata_path = settings.output_dir / settings.metadata_file
    report_path = settings.output_dir / "selection_report.csv"
    images_dir = settings.output_dir / "images"
    errors: list[str] = []

    if not metadata_path.exists():
        return [f"Missing metadata file: {metadata_path}"]
    if not report_path.exists():
        return [f"Missing selection report: {report_path}"]
    if not images_dir.is_dir():
        return [f"Missing images directory: {images_dir}"]

    metadata_records = _read_jsonl(metadata_path, errors)
    report_rows = _read_report(report_path, errors)
    if errors:
        return errors

    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
    metadata_names = [str(record.get("file_name", "")) for record in metadata_records]
    metadata_counts = Counter(metadata_names)

    if len(metadata_records) != len(image_paths):
        errors.append(
            f"Metadata record count ({len(metadata_records)}) does not match image count ({len(image_paths)})."
        )
    duplicates = [name for name, count in metadata_counts.items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate metadata file_name values: {duplicates}")

    labels_present: set[str] = set()
    for record in metadata_records:
        file_name = str(record.get("file_name", ""))
        caption = str(record.get("text", ""))
        output_image = settings.output_dir / file_name
        label = _label_from_file_name(file_name)
        if label:
            labels_present.add(label)

        if not file_name:
            errors.append("Metadata record has empty file_name.")
        elif not output_image.exists():
            errors.append(f"Metadata file_name does not exist: {file_name}")
        if not caption.strip():
            errors.append(f"Caption is empty for: {file_name}")
        if settings.trigger_word not in caption:
            errors.append(f"Caption missing trigger word for: {file_name}")
        banned = [term for term in BANNED_DIAGNOSTIC_TERMS if term in caption.lower()]
        if banned:
            errors.append(f"Caption contains prohibited terms for {file_name}: {banned}")

    missing_classes = [label for label in settings.classes if label not in labels_present]
    if missing_classes:
        errors.append(f"Missing classes in metadata: {missing_classes}")

    selected_rows = [row for row in report_rows if _as_bool(row.get("selected", ""))]
    if len(selected_rows) != len(metadata_records):
        errors.append("Selected report rows do not match metadata record count.")

    output_hashes: list[str] = []
    train_dir = settings.source_dir.resolve()
    for row in selected_rows:
        source_path = _resolve_repo_path(project_root, str(row.get("source_path", "")))
        output_path = _resolve_repo_path(project_root, str(row.get("output_path", "")))
        if not _is_relative_to(source_path.resolve(), train_dir):
            errors.append(f"Selected source is outside train: {row.get('source_path', '')}")
        if _path_contains_split(source_path, project_root, split="validation"):
            errors.append(f"Selected source comes from validation: {row.get('source_path', '')}")
        if _path_contains_split(source_path, project_root, split="test"):
            errors.append(f"Selected source comes from test: {row.get('source_path', '')}")
        if not output_path.exists():
            errors.append(f"Selected output path does not exist: {row.get('output_path', '')}")
            continue

        recorded_hash = str(row.get("sha256", ""))
        actual_hash = compute_sha256(output_path)
        output_hashes.append(actual_hash)
        if recorded_hash != actual_hash:
            errors.append(f"SHA-256 mismatch for: {row.get('output_path', '')}")

        with Image.open(output_path) as image:
            if image.mode != "RGB":
                errors.append(f"Image is not RGB: {row.get('output_path', '')}")
            if image.size != (settings.resolution, settings.resolution):
                errors.append(
                    f"Image is not {settings.resolution}x{settings.resolution}: "
                    f"{row.get('output_path', '')}"
                )

    duplicate_hashes = [digest for digest, count in Counter(output_hashes).items() if count > 1]
    if duplicate_hashes:
        errors.append(f"Duplicate output image hashes found: {duplicate_hashes}")

    return errors


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config data."""

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def _read_jsonl(metadata_path: Path, errors: list[str]) -> list[dict[str, Any]]:
    """Read metadata JSONL records."""

    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"Invalid JSONL at line {line_number}: {error}")
    return records


def _read_report(report_path: Path, errors: list[str]) -> list[dict[str, str]]:
    """Read the selection report CSV."""

    with report_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "source_path",
            "output_path",
            "label",
            "caption",
            "sha256",
            "width_original",
            "height_original",
            "selected",
            "rejection_reason",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            errors.append(f"Selection report missing columns: {sorted(missing)}")
            return []
        return list(reader)


def _label_from_file_name(file_name: str) -> str:
    """Infer the class label from generated image names."""

    stem = Path(file_name).stem
    for label in EXPECTED_CLASSES:
        if stem.startswith(f"{label}_"):
            return label
    return ""


def _resolve_repo_path(project_root: Path, path_value: str) -> Path:
    """Resolve a relative report path against the repository root."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _path_contains_split(path: Path, project_root: Path, split: str) -> bool:
    """Check whether a path is inside a forbidden processed split."""

    split_dir = (project_root / "data" / "processed" / split).resolve()
    return split_dir.exists() and _is_relative_to(path.resolve(), split_dir)


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is contained by parent for Python 3.10 compatibility."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _as_bool(value: object) -> bool:
    """Interpret CSV booleans written by DictWriter."""

    return str(value).strip().lower() in {"true", "1", "yes"}


def main(argv: Sequence[str] | None = None) -> int:
    """Run LoRA dataset validation."""

    arguments = parse_arguments(argv)
    errors = validate_lora_dataset(arguments.config, project_root=Path.cwd())
    if errors:
        print("LoRA dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("LoRA dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
