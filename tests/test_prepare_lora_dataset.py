from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from src.synthetic_data.captions import EXPECTED_CLASSES
from src.synthetic_data.prepare_lora_dataset import (
    LoraDatasetSettings,
    prepare_lora_dataset,
)


def test_prepare_lora_dataset_dry_run_does_not_create_outputs(tmp_path: Path) -> None:
    _make_project_dataset(tmp_path, images_per_class=2)
    settings = _settings(tmp_path, images_per_class=1)

    result = prepare_lora_dataset(settings, project_root=tmp_path, dry_run=True)

    assert result.dry_run is True
    assert result.selected_counts == {label: 1 for label in EXPECTED_CLASSES}
    assert not (tmp_path / "data" / "lora" / "train").exists()


def test_prepare_lora_dataset_copies_rgb_512_images_and_metadata(tmp_path: Path) -> None:
    _make_project_dataset(tmp_path, images_per_class=3)
    settings = _settings(tmp_path, images_per_class=2)

    result = prepare_lora_dataset(settings, project_root=tmp_path, overwrite=True)

    output_images = sorted((result.output_dir / "images").glob("*.jpg"))
    assert len(output_images) == 10
    assert result.selected_counts == {label: 2 for label in EXPECTED_CLASSES}

    with result.metadata_path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    assert len(records) == 10
    assert records[0]["file_name"].startswith("images/")
    assert "soyseed" in records[0]["text"]

    with output_images[0].open("rb") as file:
        assert file.read(2) == b"\xff\xd8"
    with Image.open(output_images[0]) as image:
        assert image.mode == "RGB"
        assert image.size == (512, 512)

    with result.report_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 15
    selected_rows = [row for row in rows if row["selected"] == "True"]
    assert len(selected_rows) == 10
    assert all(not Path(row["source_path"]).is_absolute() for row in rows)
    assert all(not Path(row["output_path"]).is_absolute() for row in selected_rows)
    assert all(row["source_path"].startswith("data/processed/train/") for row in selected_rows)


def test_prepare_lora_dataset_rejects_non_train_source(tmp_path: Path) -> None:
    _make_project_dataset(tmp_path, images_per_class=1)
    settings = LoraDatasetSettings(
        source_dir=tmp_path / "data" / "processed" / "validation",
        output_dir=tmp_path / "data" / "lora" / "train",
        metadata_file="metadata.jsonl",
        trigger_word="soyseed",
        images_per_class=1,
        seed=42,
        resolution=512,
        classes=EXPECTED_CLASSES,
    )

    try:
        prepare_lora_dataset(settings, project_root=tmp_path)
    except ValueError as error:
        assert "must read only" in str(error)
    else:
        raise AssertionError("Expected non-train source to be rejected.")


def _settings(tmp_path: Path, images_per_class: int) -> LoraDatasetSettings:
    return LoraDatasetSettings(
        source_dir=tmp_path / "data" / "processed" / "train",
        output_dir=tmp_path / "data" / "lora" / "train",
        metadata_file="metadata.jsonl",
        trigger_word="soyseed",
        images_per_class=images_per_class,
        seed=42,
        resolution=512,
        classes=EXPECTED_CLASSES,
    )


def _make_project_dataset(tmp_path: Path, images_per_class: int) -> None:
    for split in ("train", "validation", "test"):
        for class_index, label in enumerate(EXPECTED_CLASSES):
            class_dir = tmp_path / "data" / "processed" / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            count = images_per_class if split == "train" else 1
            for image_index in range(count):
                mode = "RGBA" if image_index == 0 else "RGB"
                color = (
                    (class_index * 40 + image_index * 7) % 255,
                    (class_index * 70 + image_index * 11) % 255,
                    (class_index * 90 + image_index * 13) % 255,
                    255,
                )
                image = Image.new(mode, (32 + image_index, 24 + class_index), color=color)
                image.save(class_dir / f"{image_index}.png")
