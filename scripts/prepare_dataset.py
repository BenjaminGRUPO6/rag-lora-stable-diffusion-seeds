"""Punto de entrada reservado para crear splits reproducibles después de la auditoría.

No ejecutar hasta revisar manualmente `results/dataset_audit`.
"""

if False and __name__ == "__main__":
    print("Pendiente: implementar splits después de validar la auditoría del dataset.")
import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.cleaning import (
    build_cleaning_manifest,
    create_exclusion_list,
    create_near_duplicates_review,
    read_audit_reports,
    write_metadata_reports,
)
from src.data.split_dataset import stratified_group_split
from src.data.verify import EXPECTED_CLASSES

SPLITS = ("train", "validation", "test")
DATASET_SPLIT_COLUMNS = [
    "image_id",
    "source_path",
    "processed_path",
    "label",
    "split",
    "sha256",
    "is_synthetic",
    "exclusion_status",
]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""

    parser = argparse.ArgumentParser(
        description="Prepare a cleaned, leakage-safe dataset split."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def relative_display_path(path: Path, base_dir: Path | None = None) -> str:
    """Return a portable display path, relative to base_dir when possible."""

    resolved_base = (base_dir or Path.cwd()).resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def prepare_dataset(
    dataset: Path,
    audit_dir: Path,
    output: Path,
    metadata_output: Path,
    seed: int = 42,
) -> dict[str, object]:
    """Prepare metadata and copy included images into split directories."""

    reports = read_audit_reports(audit_dir)
    exclusions = create_exclusion_list(reports)

    review_path = metadata_output / "near_duplicates_review.csv"
    if review_path.exists():
        near_duplicates_review = pd.read_csv(review_path)
    else:
        near_duplicates_review = create_near_duplicates_review(reports)

    write_metadata_reports(
        exclusions=exclusions,
        near_duplicates_review=near_duplicates_review,
        metadata_output=metadata_output,
    )

    manifest = build_cleaning_manifest(reports, exclusions)
    included = manifest[manifest["exclusion_status"] == "included"].copy()
    split_included = stratified_group_split(
        included,
        review_frame=near_duplicates_review,
        seed=seed,
    )

    split_lookup = split_included.set_index("relative_path")["split"].to_dict()
    manifest["split"] = manifest["relative_path"].map(split_lookup).fillna("excluded")

    for split in SPLITS:
        for class_name in EXPECTED_CLASSES:
            (output / split / class_name).mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        relative_path = Path(str(row["relative_path"]))
        source_path = dataset / relative_path
        split = str(row["split"])
        processed_path = ""

        if row["exclusion_status"] == "included":
            processed_file = output / split / relative_path
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, processed_file)
            processed_path = relative_display_path(processed_file)

        split_rows.append(
            {
                "image_id": row["image_id"],
                "source_path": relative_display_path(source_path),
                "processed_path": processed_path,
                "label": row["label"],
                "split": split,
                "sha256": row["sha256"],
                "is_synthetic": bool(row["is_synthetic"]),
                "exclusion_status": row["exclusion_status"],
            }
        )

    dataset_split = pd.DataFrame(split_rows, columns=DATASET_SPLIT_COLUMNS)
    metadata_output.mkdir(parents=True, exist_ok=True)
    dataset_split.to_csv(
        metadata_output / "dataset_split.csv",
        index=False,
        encoding="utf-8",
    )

    results_dir = Path("results") / "dataset_preparation"
    results_dir.mkdir(parents=True, exist_ok=True)

    class_distribution = (
        dataset_split[dataset_split["exclusion_status"] == "included"]
        .groupby(["split", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "label"])
    )
    class_distribution.to_csv(
        results_dir / "class_distribution_by_split.csv",
        index=False,
        encoding="utf-8",
    )

    summary = build_summary(dataset_split, exclusions, reports.summary)
    with (results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return summary


def build_summary(
    dataset_split: pd.DataFrame,
    exclusions: pd.DataFrame,
    audit_summary: dict[str, object],
) -> dict[str, object]:
    """Build a JSON-serializable preparation summary."""

    included = dataset_split[dataset_split["exclusion_status"] == "included"]
    split_counts = included["split"].value_counts().sort_index().to_dict()
    exclusion_reasons = (
        exclusions["exclusion_reason"].value_counts().sort_index().to_dict()
        if not exclusions.empty
        else {}
    )
    synthetic_train = included[
        (included["split"] == "train") & (included["is_synthetic"].astype(bool))
    ]

    return {
        "audit_total_files": int(audit_summary.get("total_files", len(dataset_split))),
        "included_images": int(len(included)),
        "excluded_images": int(len(exclusions)),
        "exclusion_reasons": {
            str(key): int(value) for key, value in exclusion_reasons.items()
        },
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "synthetic_train_images": int(len(synthetic_train)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset preparation command."""

    arguments = parse_arguments(argv)
    summary = prepare_dataset(
        dataset=arguments.dataset,
        audit_dir=arguments.audit_dir,
        output=arguments.output,
        metadata_output=arguments.metadata_output,
        seed=arguments.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
