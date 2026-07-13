from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.validate_dataset_split import validate_dataset_split
from src.data.split_dataset import stratified_group_split
from src.data.verify import EXPECTED_CLASSES


def test_stratified_group_split_prevents_leakage_and_keeps_synthetic_in_train() -> None:
    frame = _split_frame(records_per_class=10)
    review = pd.DataFrame(
        [
            {
                "path_a": "intact/0.jpg",
                "path_b": "intact/1.jpg",
                "equivalent": True,
            }
        ]
    )
    frame.loc[0, "sha256"] = "same-hash"
    frame.loc[1, "sha256"] = "same-hash"
    frame.loc[len(frame)] = {
        "image_id": "synthetic_intact_0",
        "relative_path": "synthetic/intact/0.jpg",
        "label": "intact",
        "sha256": "synthetic-hash",
        "is_synthetic": True,
        "exclusion_status": "included",
    }

    result = stratified_group_split(frame, review_frame=review, seed=7)

    equivalent_splits = set(
        result[result["relative_path"].isin(["intact/0.jpg", "intact/1.jpg"])]["split"]
    )
    assert len(equivalent_splits) == 1
    assert result.loc[
        result["relative_path"] == "synthetic/intact/0.jpg",
        "split",
    ].item() == "train"

    split_counts = result["split"].value_counts().to_dict()
    assert split_counts["train"] >= split_counts["validation"]
    assert split_counts["train"] >= split_counts["test"]


def test_validate_dataset_split_accepts_valid_manifest(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for split, count in {"train": 40, "validation": 5, "test": 5}.items():
        for index in range(count):
            label = EXPECTED_CLASSES[index % len(EXPECTED_CLASSES)]
            source = tmp_path / "raw" / label / f"{split}_{index}.jpg"
            processed = tmp_path / "processed" / split / label / f"{index}.jpg"
            source.parent.mkdir(parents=True, exist_ok=True)
            processed.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source")
            processed.write_bytes(b"processed")
            rows.append(
                {
                    "image_id": f"{split}_{index}",
                    "source_path": source.as_posix(),
                    "processed_path": processed.as_posix(),
                    "label": label,
                    "split": split,
                    "sha256": f"{split}-{index}",
                    "is_synthetic": False,
                    "exclusion_status": "included",
                }
            )

    manifest_path = tmp_path / "dataset_split.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    assert validate_dataset_split(manifest_path, base_dir=tmp_path) == []


def _split_frame(records_per_class: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in EXPECTED_CLASSES:
        for index in range(records_per_class):
            relative_path = f"{label}/{index}.jpg"
            rows.append(
                {
                    "image_id": f"{label}_{index}",
                    "relative_path": relative_path,
                    "label": label,
                    "sha256": f"{label}-{index}",
                    "is_synthetic": False,
                    "exclusion_status": "included",
                }
            )
    return pd.DataFrame(rows)
