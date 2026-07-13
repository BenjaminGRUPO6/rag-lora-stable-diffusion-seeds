from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image

from scripts.validate_lora_dataset import validate_lora_dataset
from src.synthetic_data.captions import EXPECTED_CLASSES
from src.synthetic_data.prepare_lora_dataset import LoraDatasetSettings, prepare_lora_dataset


def test_validate_lora_dataset_accepts_prepared_dataset(tmp_path: Path) -> None:
    config_path = _prepare_valid_dataset(tmp_path)

    assert validate_lora_dataset(config_path, project_root=tmp_path) == []


def test_validate_lora_dataset_rejects_banned_caption_term(tmp_path: Path) -> None:
    config_path = _prepare_valid_dataset(tmp_path)
    metadata_path = tmp_path / "data" / "lora" / "train" / "metadata.jsonl"
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["text"] = "photo of a soyseed soybean seed with disease"
    metadata_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    errors = validate_lora_dataset(config_path, project_root=tmp_path)

    assert any("prohibited terms" in error for error in errors)


def test_validate_lora_dataset_rejects_validation_source_in_report(tmp_path: Path) -> None:
    config_path = _prepare_valid_dataset(tmp_path)
    report_path = tmp_path / "data" / "lora" / "train" / "selection_report.csv"
    with report_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0].keys())
    rows[0]["source_path"] = "data/processed/validation/intact/0.png"
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = validate_lora_dataset(config_path, project_root=tmp_path)

    assert any("outside train" in error for error in errors)
    assert any("validation" in error for error in errors)


def _prepare_valid_dataset(tmp_path: Path) -> Path:
    _make_project_dataset(tmp_path)
    config_path = tmp_path / "configs" / "lora_sd15_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model": {"resolution": 512},
        "dataset": {
            "source_dir": "data/processed/train",
            "output_dir": "data/lora/train",
            "metadata_file": "metadata.jsonl",
            "trigger_word": "soyseed",
            "images_per_class": 1,
            "seed": 42,
        },
        "classes": list(EXPECTED_CLASSES),
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = LoraDatasetSettings(
        source_dir=tmp_path / "data" / "processed" / "train",
        output_dir=tmp_path / "data" / "lora" / "train",
        metadata_file="metadata.jsonl",
        trigger_word="soyseed",
        images_per_class=1,
        seed=42,
        resolution=512,
        classes=EXPECTED_CLASSES,
    )
    prepare_lora_dataset(settings, project_root=tmp_path, overwrite=True)
    return config_path


def _make_project_dataset(tmp_path: Path) -> None:
    for split in ("train", "validation", "test"):
        for class_index, label in enumerate(EXPECTED_CLASSES):
            class_dir = tmp_path / "data" / "processed" / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            color = (
                (class_index * 45 + len(split)) % 255,
                (class_index * 65 + len(split) * 2) % 255,
                (class_index * 85 + len(split) * 3) % 255,
            )
            Image.new("RGB", (40 + class_index, 30 + class_index), color=color).save(
                class_dir / "0.png"
            )
