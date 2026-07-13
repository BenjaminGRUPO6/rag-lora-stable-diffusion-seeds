from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.reconcile_vision_results import (
    EXPECTED_TEST_SAMPLES,
    validate_class_contract,
    validate_manifest_test_split,
)
from src.vision.dataset import EXPECTED_CLASSES


RESULTS_DIR = Path("results/vision/resnet18_baseline")
MANIFEST_PATH = Path("data/metadata/dataset_split.csv")


def test_dataset_split_test_contract_matches_baseline_expectations() -> None:
    """The original test split must contain 522 non-synthetic images."""
    validation = validate_manifest_test_split(
        manifest_path=MANIFEST_PATH,
        expected_classes=EXPECTED_CLASSES,
        expected_test_samples=EXPECTED_TEST_SAMPLES,
    )

    assert validation["sample_count"] == EXPECTED_TEST_SAMPLES
    assert validation["synthetic_count"] == 0
    assert validation["support"] == {
        "intact": 91,
        "spotted": 106,
        "immature": 112,
        "broken": 100,
        "skin_damaged": 113,
    }
    assert sum(validation["support"].values()) == EXPECTED_TEST_SAMPLES


def test_class_contract_uses_expected_order() -> None:
    """Config and checkpoint mappings must use the expected visual class order."""
    mapping = {class_name: index for index, class_name in enumerate(EXPECTED_CLASSES)}

    assert validate_class_contract(EXPECTED_CLASSES, mapping) == mapping


def test_reconciled_run_summary_matches_metrics_test() -> None:
    """run_summary must read final test metrics from metrics_test.json."""
    metrics = _read_json(RESULTS_DIR / "metrics_test.json")
    summary = _read_json(RESULTS_DIR / "run_summary.json")

    assert summary["canonical_evaluation"] is True
    assert summary["metrics_source"] == str(RESULTS_DIR / "metrics_test.json")
    assert summary["test_samples"] == EXPECTED_TEST_SAMPLES
    assert summary["test_accuracy"] == metrics["accuracy"]
    assert summary["test_macro_f1"] == metrics["macro_f1"]


def test_classification_report_support_sum_is_522() -> None:
    """Per-class support in the report must sum to the canonical test size."""
    report = pd.read_csv(RESULTS_DIR / "classification_report.csv", index_col=0)
    support_sum = int(report.loc[list(EXPECTED_CLASSES), "support"].sum())

    assert support_sum == EXPECTED_TEST_SAMPLES


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
