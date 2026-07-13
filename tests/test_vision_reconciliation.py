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


RESULTS_DIR = Path("results/vision/resultados_1_baseline")
MANIFEST_PATH = Path("data/metadata/dataset_split.csv")
PNG_FILENAMES = (
    "r1_metricas_resumen.png",
    "r1_f1_por_clase.png",
    "r1_precision_recall_por_clase.png",
    "r1_matriz_confusion.png",
    "r1_matriz_confusion_normalizada.png",
    "r1_distribucion_confianza.png",
)


def test_dataset_split_test_contract_matches_baseline_expectations() -> None:
    """The registered test split must contain 522 non-synthetic images."""
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


def test_reconciled_metrics_json_matches_run_summary() -> None:
    """run_summary must read final test metrics from r1_metricas.json."""
    metrics = _read_json(RESULTS_DIR / "r1_metricas.json")
    summary = _read_json(RESULTS_DIR / "run_summary.json")

    assert summary["canonical_evaluation"] is True
    assert summary["metrics_source"] == (RESULTS_DIR / "r1_metricas.json").as_posix()
    assert summary["test_samples"] == EXPECTED_TEST_SAMPLES
    assert summary["synthetic_in_test"] == 0
    assert summary["test_accuracy"] == metrics["accuracy"]
    assert summary["test_macro_f1"] == metrics["macro_f1"]
    assert summary["test_macro_precision"] == metrics["macro_precision"]
    assert summary["test_macro_recall"] == metrics["macro_recall"]


def test_classification_report_contains_all_classes_and_support_sum_is_522() -> None:
    """Per-class support in the canonical report must sum to the test size."""
    report = pd.read_csv(RESULTS_DIR / "r1_reporte_clasificacion.csv", index_col=0)

    assert list(EXPECTED_CLASSES) == [
        class_name for class_name in EXPECTED_CLASSES if class_name in report.index
    ]
    support_sum = int(report.loc[list(EXPECTED_CLASSES), "support"].sum())
    assert support_sum == EXPECTED_TEST_SAMPLES


def test_reconciliation_report_validates_no_synthetics_and_evaluation_mode() -> None:
    """The reconciliation report records the required execution validations."""
    report = _read_json(RESULTS_DIR / "r1_reconciliation_report.json")

    assert report["validations"]["manifest"]["synthetic_count"] == 0
    assert report["validations"]["manifest"]["sample_count"] == EXPECTED_TEST_SAMPLES
    assert report["validations"]["classes"] == list(EXPECTED_CLASSES)
    assert report["validations"]["evaluation"]["model_eval"] is True
    assert report["validations"]["evaluation"]["torch_inference_mode"] is True
    assert report["validations"]["dataset"]["deterministic_transforms"] is True


def test_manifest_is_reconciled_with_checkpoint_and_final_metrics() -> None:
    """The Resultados 1 manifest must expose final reconciled metadata."""
    manifest = _read_json(RESULTS_DIR / "manifest.json")
    metrics = _read_json(RESULTS_DIR / "r1_metricas.json")

    assert manifest["status"] == "RECONCILED"
    assert manifest["checkpoint_sha256"]
    assert manifest["test_samples"] == EXPECTED_TEST_SAMPLES
    assert manifest["final_metrics"]["accuracy"] == metrics["accuracy"]
    assert manifest["final_metrics"]["macro_f1"] == metrics["macro_f1"]


def test_reconciled_png_artifacts_exist_and_are_non_empty() -> None:
    """All requested PNG charts must exist and have non-zero size."""
    for filename in PNG_FILENAMES:
        path = RESULTS_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 0


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
