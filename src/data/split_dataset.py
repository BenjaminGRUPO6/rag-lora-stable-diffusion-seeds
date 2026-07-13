from __future__ import annotations

import random
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    frame: pd.DataFrame,
    label_column: str = "main_label",
    seed: int = 42,
    train_size: float = 0.8,
    validation_size: float = 0.1,
) -> pd.DataFrame:
    if abs(train_size + validation_size - 0.9) > 1e-9:
        raise ValueError("La configuración esperada reserva 10% para test.")

    train, remaining = train_test_split(
        frame,
        train_size=train_size,
        random_state=seed,
        stratify=frame[label_column],
    )
    validation, test = train_test_split(
        remaining,
        train_size=0.5,
        random_state=seed,
        stratify=remaining[label_column],
    )
    output = frame.copy()
    output["split"] = ""
    output.loc[train.index, "split"] = "train"
    output.loc[validation.index, "split"] = "validation"
    output.loc[test.index, "split"] = "test"
    return output


class UnionFind:
    """Small union-find helper for split leakage groups."""

    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        """Return the canonical parent for a value."""

        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        """Merge two values into the same group."""

        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def reviewed_equivalent_pairs(review_frame: pd.DataFrame | None) -> list[tuple[str, str]]:
    """Return near-duplicate pairs reviewed as equivalent."""

    if review_frame is None or review_frame.empty:
        return []
    if not {"path_a", "path_b", "equivalent"}.issubset(review_frame.columns):
        return []

    pairs: list[tuple[str, str]] = []
    for _, row in review_frame.iterrows():
        equivalent = str(row["equivalent"]).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "si",
            "s",
        }
        if equivalent:
            pairs.append((str(row["path_a"]), str(row["path_b"])))
    return pairs


def assign_leakage_groups(
    frame: pd.DataFrame,
    review_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign leakage-safe groups by exact hash and reviewed visual equivalence."""

    required = {"relative_path", "sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for grouped split: {sorted(missing)}")

    output = frame.copy()
    output["relative_path"] = output["relative_path"].astype(str)
    union_find = UnionFind(output["relative_path"].tolist())

    for _, group in output.dropna(subset=["sha256"]).groupby("sha256"):
        paths = sorted(str(path) for path in group["relative_path"].tolist())
        for path in paths[1:]:
            union_find.union(paths[0], path)

    known_paths = set(output["relative_path"].tolist())
    for first, second in reviewed_equivalent_pairs(review_frame):
        if first in known_paths and second in known_paths:
            union_find.union(first, second)

    output["leakage_group"] = output["relative_path"].map(union_find.find)
    return output


def stratified_group_split(
    frame: pd.DataFrame,
    review_frame: pd.DataFrame | None = None,
    label_column: str = "label",
    seed: int = 42,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
) -> pd.DataFrame:
    """Split clean records by label while keeping leakage groups together."""

    if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
        raise ValueError("Split sizes must sum to 1.0.")
    required = {label_column, "relative_path", "sha256", "is_synthetic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for dataset split: {sorted(missing)}")

    grouped_frame = assign_leakage_groups(frame, review_frame=review_frame)
    grouped_frame["split"] = ""

    group_rows: list[dict[str, object]] = []
    for group_id, group in grouped_frame.groupby("leakage_group", sort=True):
        labels = [str(label) for label in group[label_column].tolist()]
        label = Counter(labels).most_common(1)[0][0]
        group_rows.append(
            {
                "leakage_group": group_id,
                "label": label,
                "is_synthetic": bool(group["is_synthetic"].astype(bool).any()),
            }
        )

    groups = pd.DataFrame(group_rows)
    split_by_group: dict[str, str] = {}

    synthetic_groups = groups[groups["is_synthetic"]]
    for group_id in synthetic_groups["leakage_group"].tolist():
        split_by_group[str(group_id)] = "train"

    real_groups = groups[~groups["is_synthetic"]]
    rng = random.Random(seed)

    for label, label_groups in real_groups.groupby("label", sort=True):
        group_ids = [str(group_id) for group_id in label_groups["leakage_group"].tolist()]
        group_ids.sort()
        rng.shuffle(group_ids)

        validation_count, test_count = _holdout_counts(
            len(group_ids),
            validation_size=validation_size,
            test_size=test_size,
        )

        test_groups = set(group_ids[:test_count])
        validation_groups = set(group_ids[test_count : test_count + validation_count])

        for group_id in group_ids:
            if group_id in test_groups:
                split_by_group[group_id] = "test"
            elif group_id in validation_groups:
                split_by_group[group_id] = "validation"
            else:
                split_by_group[group_id] = "train"

    grouped_frame["split"] = grouped_frame["leakage_group"].map(split_by_group)
    return grouped_frame


def _holdout_counts(
    total: int,
    validation_size: float,
    test_size: float,
) -> tuple[int, int]:
    """Calculate validation and test group counts for one class."""

    if total <= 2:
        return 0, 0

    validation_count = max(1, round(total * validation_size))
    test_count = max(1, round(total * test_size))

    while validation_count + test_count >= total:
        if validation_count >= test_count and validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    return validation_count, test_count


def stratified_split(
    frame: pd.DataFrame,
    label_column: str = "main_label",
    seed: int = 42,
    train_size: float = 0.8,
    validation_size: float = 0.1,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    If the frame includes the cleaning columns, it uses leakage-safe grouped
    splitting. Otherwise it falls back to the original row-level behavior.
    """

    grouped_columns = {"relative_path", "sha256", "is_synthetic"}
    if grouped_columns.issubset(frame.columns):
        resolved_label = label_column if label_column in frame.columns else "label"
        return stratified_group_split(
            frame=frame,
            label_column=resolved_label,
            seed=seed,
            train_size=train_size,
            validation_size=validation_size,
            test_size=1.0 - train_size - validation_size,
        )

    if abs(train_size + validation_size - 0.9) > 1e-9:
        raise ValueError("La configuracion esperada reserva 10% para test.")

    train, remaining = train_test_split(
        frame,
        train_size=train_size,
        random_state=seed,
        stratify=frame[label_column],
    )
    validation, test = train_test_split(
        remaining,
        train_size=0.5,
        random_state=seed,
        stratify=remaining[label_column],
    )
    output = frame.copy()
    output["split"] = ""
    output.loc[train.index, "split"] = "train"
    output.loc[validation.index, "split"] = "validation"
    output.loc[test.index, "split"] = "test"
    return output
