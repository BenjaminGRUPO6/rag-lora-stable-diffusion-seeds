from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "image_id",
    "file_path",
    "seed_type",
    "main_label",
    "verified",
    "source",
    "synthetic",
}


def validate_metadata(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        errors.append(f"Faltan columnas: {sorted(missing)}")
    if "image_id" in frame.columns and frame["image_id"].duplicated().any():
        errors.append("Existen image_id duplicados.")
    if "main_label" in frame.columns and frame["main_label"].isna().any():
        errors.append("Existen imágenes sin etiqueta principal.")
    return errors


AUDIT_FILES = {
    "summary": "summary.json",
    "images": "images.csv",
    "category_distribution": "category_distribution.csv",
    "corrupted": "corrupted_files.csv",
    "exact_duplicates": "exact_duplicates.csv",
    "near_duplicates": "possible_near_duplicates.csv",
}

EXCLUSION_COLUMNS = [
    "image_id",
    "relative_path",
    "label",
    "sha256",
    "exclusion_status",
    "exclusion_reason",
    "duplicate_group_id",
    "kept_relative_path",
]

NEAR_DUPLICATE_REVIEW_COLUMNS = [
    "review_group_id",
    "path_a",
    "path_b",
    "label",
    "phash_a",
    "phash_b",
    "phash_distance",
    "equivalent",
    "review_status",
    "notes",
]


@dataclass(frozen=True)
class AuditReports:
    """Audit outputs loaded from results/dataset_audit."""

    summary: dict[str, Any]
    images: pd.DataFrame
    category_distribution: pd.DataFrame
    corrupted: pd.DataFrame
    exact_duplicates: pd.DataFrame
    near_duplicates: pd.DataFrame


def read_audit_reports(audit_dir: str | Path) -> AuditReports:
    """Load all dataset audit reports from a directory."""

    root = Path(audit_dir)
    missing = [
        file_name
        for file_name in AUDIT_FILES.values()
        if not (root / file_name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing audit report files in {root}: {', '.join(sorted(missing))}"
        )

    with (root / AUDIT_FILES["summary"]).open("r", encoding="utf-8") as file:
        summary = json.load(file)

    return AuditReports(
        summary=summary,
        images=pd.read_csv(root / AUDIT_FILES["images"]),
        category_distribution=pd.read_csv(root / AUDIT_FILES["category_distribution"]),
        corrupted=pd.read_csv(root / AUDIT_FILES["corrupted"]),
        exact_duplicates=pd.read_csv(root / AUDIT_FILES["exact_duplicates"]),
        near_duplicates=pd.read_csv(root / AUDIT_FILES["near_duplicates"]),
    )


def make_image_id(relative_path: str) -> str:
    """Create a stable image identifier from a relative path."""

    without_extension = Path(relative_path).with_suffix("").as_posix()
    image_id = re.sub(r"[^A-Za-z0-9]+", "_", without_extension).strip("_")
    return image_id.lower()


def normalize_bool(value: object) -> bool:
    """Convert common CSV boolean representations to bool."""

    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "s"}


def is_synthetic_path(relative_path: str) -> bool:
    """Infer whether a record is synthetic from its relative path."""

    normalized = Path(relative_path).as_posix().lower()
    parts = set(normalized.split("/"))
    return "synthetic" in parts or normalized.startswith("synthetic/")


def create_exclusion_list(reports: AuditReports) -> pd.DataFrame:
    """
    Build the controlled exclusion list.

    Corrupted files are excluded. Exact duplicate groups keep one canonical file,
    selected by lexicographic relative path, and exclude the remaining files.
    Possible near duplicates are not excluded automatically.
    """

    exclusions: dict[str, dict[str, object]] = {}

    for _, row in reports.corrupted.fillna("").iterrows():
        relative_path = str(row["relative_path"])
        exclusions[relative_path] = {
            "image_id": make_image_id(relative_path),
            "relative_path": relative_path,
            "label": str(row.get("category", "")),
            "sha256": "",
            "exclusion_status": "excluded",
            "exclusion_reason": "corrupted_file",
            "duplicate_group_id": "",
            "kept_relative_path": "",
        }

    if not reports.exact_duplicates.empty:
        grouped = reports.exact_duplicates.fillna("").groupby("group_id", sort=True)
        for group_id, group in grouped:
            ordered = sorted(str(path) for path in group["relative_path"].tolist())
            if len(ordered) <= 1:
                continue

            kept_path = ordered[0]
            for duplicate_path in ordered[1:]:
                duplicate_row = group[group["relative_path"] == duplicate_path].iloc[0]
                exclusions[duplicate_path] = {
                    "image_id": make_image_id(duplicate_path),
                    "relative_path": duplicate_path,
                    "label": str(duplicate_row.get("category", "")),
                    "sha256": str(duplicate_row.get("sha256", "")),
                    "exclusion_status": "excluded",
                    "exclusion_reason": "exact_duplicate",
                    "duplicate_group_id": str(group_id),
                    "kept_relative_path": kept_path,
                }

    rows = [exclusions[path] for path in sorted(exclusions)]
    return pd.DataFrame(rows, columns=EXCLUSION_COLUMNS)


def create_near_duplicates_review(reports: AuditReports) -> pd.DataFrame:
    """Create the manual review table for possible visual duplicates."""

    rows: list[dict[str, object]] = []
    near_duplicates = reports.near_duplicates.fillna("")

    for index, row in near_duplicates.iterrows():
        rows.append(
            {
                "review_group_id": f"near_{index + 1:04d}",
                "path_a": str(row.get("path_a", "")),
                "path_b": str(row.get("path_b", "")),
                "label": str(row.get("category", "")),
                "phash_a": str(row.get("phash_a", "")),
                "phash_b": str(row.get("phash_b", "")),
                "phash_distance": row.get("phash_distance", ""),
                "equivalent": False,
                "review_status": "pending",
                "notes": "",
            }
        )

    return pd.DataFrame(rows, columns=NEAR_DUPLICATE_REVIEW_COLUMNS)


def write_metadata_reports(
    exclusions: pd.DataFrame,
    near_duplicates_review: pd.DataFrame,
    metadata_output: str | Path,
) -> None:
    """Write cleaning metadata CSV files."""

    output_dir = Path(metadata_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclusions.to_csv(output_dir / "exclusions.csv", index=False, encoding="utf-8")
    near_duplicates_review.to_csv(
        output_dir / "near_duplicates_review.csv",
        index=False,
        encoding="utf-8",
    )


def build_cleaning_manifest(
    reports: AuditReports,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    """Return audited images annotated with inclusion or exclusion status."""

    images = reports.images.copy()
    if images.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "relative_path",
                "label",
                "sha256",
                "is_synthetic",
                "exclusion_status",
                "exclusion_reason",
            ]
        )

    images["relative_path"] = images["relative_path"].astype(str)
    images["image_id"] = images["relative_path"].map(make_image_id)
    images["label"] = images["category"].astype(str)
    images["is_synthetic"] = images.apply(_detect_synthetic_row, axis=1)

    exclusion_lookup = (
        exclusions.set_index("relative_path").to_dict(orient="index")
        if not exclusions.empty
        else {}
    )

    statuses: list[str] = []
    reasons: list[str] = []

    for relative_path in images["relative_path"]:
        exclusion = exclusion_lookup.get(relative_path)
        if exclusion:
            statuses.append(str(exclusion["exclusion_status"]))
            reasons.append(str(exclusion["exclusion_reason"]))
        else:
            statuses.append("included")
            reasons.append("")

    images["exclusion_status"] = statuses
    images["exclusion_reason"] = reasons

    return images[
        [
            "image_id",
            "relative_path",
            "label",
            "sha256",
            "is_synthetic",
            "exclusion_status",
            "exclusion_reason",
        ]
    ].copy()


def _detect_synthetic_row(row: pd.Series) -> bool:
    """Detect synthetic records from explicit columns or path convention."""

    for column in ("is_synthetic", "synthetic"):
        if column in row.index and normalize_bool(row[column]):
            return True
    return is_synthetic_path(str(row["relative_path"]))
