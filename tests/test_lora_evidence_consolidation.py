from __future__ import annotations

import json
from pathlib import Path

from scripts.consolidate_lora_evidence import (
    CLASS_NAMES,
    count_distribution,
    extract_notebook_parameters,
    metadata_has_private_paths,
    read_metadata_records,
)


def test_extract_notebook_parameters_from_sources(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model_id = 'stable-diffusion-v1-5/stable-diffusion-v1-5'\n",
                    "rank = 8\n",
                    "learning_rate = 0.0001\n",
                    "--train_batch_size=1 --gradient_accumulation_steps=4\n",
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = tmp_path / "training.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    parameters = extract_notebook_parameters(notebook_path)

    assert parameters["base_model"].value == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert parameters["rank"].value == 8
    assert parameters["learning_rate"].value == 0.0001
    assert parameters["train_batch_size"].value == 1
    assert parameters["gradient_accumulation_steps"].value == 4


def test_metadata_jsonl_count_and_distribution(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    records = [
        {"file_name": "images/intact_00001.jpg", "text": "photo of intact soybean seed"},
        {"file_name": "images/broken_00001.jpg", "text": "photo of broken soybean seed"},
        {
            "file_name": "images/skin_damaged_00001.jpg",
            "text": "photo of skin_damaged soybean seed",
        },
    ]
    metadata_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = read_metadata_records(metadata_path)
    distribution = count_distribution(loaded)

    assert len(loaded) == 3
    assert distribution["intact"] == 1
    assert distribution["broken"] == 1
    assert distribution["skin_damaged"] == 1


def test_metadata_private_absolute_paths_are_detected() -> None:
    records = [
        {
            "file_name": r"C:\Users\ExampleUser\private\seed.jpg",
            "text": "photo of intact soybean seed",
        }
    ]

    assert metadata_has_private_paths(records)


def test_repository_lora_metadata_has_no_private_absolute_paths() -> None:
    records = read_metadata_records(Path("data/lora/train/metadata.jsonl"))

    assert not metadata_has_private_paths(records)


def test_repository_lora_distribution_is_coherent() -> None:
    metadata_path = Path("data/lora/train/metadata.jsonl")
    records = read_metadata_records(metadata_path)
    distribution = count_distribution(records)

    assert len(records) == sum(distribution.values())
    assert set(distribution) == set(CLASS_NAMES)
    assert all(distribution[class_name] == 200 for class_name in CLASS_NAMES)
