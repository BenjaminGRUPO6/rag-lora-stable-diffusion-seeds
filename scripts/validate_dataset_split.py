from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.cleaning import normalize_bool
from src.data.verify import EXPECTED_CLASSES

EXPECTED_SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for split validation."""

    parser = argparse.ArgumentParser(description="Validate a prepared dataset split.")
    parser.add_argument(
        "--dataset-split",
        type=Path,
        default=Path("data") / "metadata" / "dataset_split.csv",
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--tolerance", type=float, default=0.05)
    return parser.parse_args(argv)


def validate_dataset_split(
    dataset_split: Path,
    base_dir: Path = Path.cwd(),
    tolerance: float = 0.05,
) -> list[str]:
    """Validate split metadata and referenced files."""

    if not dataset_split.exists():
        raise FileNotFoundError(f"Dataset split file not found: {dataset_split}")

    frame = pd.read_csv(dataset_split).fillna("")
    errors: list[str] = []

    required_columns = {
        "image_id",
        "source_path",
        "processed_path",
        "label",
        "split",
        "sha256",
        "is_synthetic",
        "exclusion_status",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        return [f"Missing required columns: {sorted(missing)}"]

    included = frame[frame["exclusion_status"] == "included"].copy()
    split_rows = included[included["split"].isin(EXPECTED_SPLIT_RATIOS)]

    labels = set(included["label"].astype(str).tolist())
    missing_classes = [class_name for class_name in EXPECTED_CLASSES if class_name not in labels]
    if missing_classes:
        errors.append(f"Missing classes in included records: {missing_classes}")

    if frame["source_path"].duplicated().any():
        errors.append("Duplicate source_path values found.")

    processed_paths = split_rows["processed_path"]
    if processed_paths.duplicated().any():
        errors.append("Duplicate processed_path values found.")

    hash_splits = (
        split_rows.groupby("sha256")["split"].nunique().reset_index(name="split_count")
    )
    leaked_hashes = hash_splits[hash_splits["split_count"] > 1]
    if not leaked_hashes.empty:
        errors.append("Exact hashes appear in more than one split.")

    holdout = split_rows[split_rows["split"].isin({"validation", "test"})]
    holdout_synthetic = holdout["is_synthetic"].map(normalize_bool)
    if holdout_synthetic.any():
        errors.append("Synthetic images found in validation or test.")

    errors.extend(_validate_ratios(split_rows, tolerance=tolerance))
    errors.extend(_validate_paths(frame, split_rows, base_dir=base_dir))
    return errors


def _validate_ratios(frame: pd.DataFrame, tolerance: float) -> list[str]:
    """Validate approximate split proportions."""

    if frame.empty:
        return ["No included split records found."]

    errors: list[str] = []
    total = len(frame)
    counts = frame["split"].value_counts().to_dict()

    for split, expected_ratio in EXPECTED_SPLIT_RATIOS.items():
        actual_ratio = counts.get(split, 0) / total
        if abs(actual_ratio - expected_ratio) > tolerance:
            errors.append(
                f"Split ratio out of tolerance for {split}: "
                f"expected {expected_ratio:.2f}, got {actual_ratio:.2f}."
            )

    return errors


def _validate_paths(
    all_rows: pd.DataFrame,
    split_rows: pd.DataFrame,
    base_dir: Path,
) -> list[str]:
    """Validate that all registered source and processed files exist."""

    errors: list[str] = []
    for source_path in all_rows["source_path"].astype(str):
        if source_path and not _resolve_path(source_path, base_dir).exists():
            errors.append(f"Missing source file: {source_path}")

    for processed_path in split_rows["processed_path"].astype(str):
        if not processed_path:
            errors.append("Included record has an empty processed_path.")
        elif not _resolve_path(processed_path, base_dir).exists():
            errors.append(f"Missing processed file: {processed_path}")

    return errors


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve a CSV path value against base_dir when it is relative."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the split validation command."""

    arguments = parse_arguments(argv)
    errors = validate_dataset_split(
        dataset_split=arguments.dataset_split,
        base_dir=arguments.base_dir,
        tolerance=arguments.tolerance,
    )
    if errors:
        print("Dataset split validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Dataset split validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
