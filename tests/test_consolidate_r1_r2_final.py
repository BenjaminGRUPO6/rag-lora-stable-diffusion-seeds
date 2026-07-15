from __future__ import annotations

from scripts.consolidate_r1_r2_final import (
    CLASS_NAMES,
    ModelCandidate,
    build_final_metrics,
    delta,
    select_final_candidate,
    validate_candidates,
)


def test_select_final_candidate_uses_validation_macro_f1_only() -> None:
    """A higher test score must not override lower validation macro-F1."""
    low_validation_high_test = _candidate(
        model_id="a",
        validation_macro_f1=0.80,
        test_macro_f1=0.99,
    )
    high_validation_low_test = _candidate(
        model_id="b",
        validation_macro_f1=0.81,
        test_macro_f1=0.50,
    )

    selected = select_final_candidate([low_validation_high_test, high_validation_low_test])

    assert selected.model_id == "b"


def test_validate_candidates_checks_splits_seed_synthetic_and_test_rows() -> None:
    """Candidate validation should fail when test rows or split support differ."""
    expected_counts = {
        "train": {class_name: 2 for class_name in CLASS_NAMES},
        "validation": {class_name: 1 for class_name in CLASS_NAMES},
        "test": {class_name: 1 for class_name in CLASS_NAMES},
    }
    candidate = _candidate(
        model_id="resnet18_baseline",
        validation_macro_f1=0.7,
        test_macro_f1=0.6,
        class_distribution=expected_counts,
        predictions_path=None,
    )
    dataset_summary = {
        "counts": expected_counts,
        "synthetic_counts": {"train": 0, "validation": 0, "test": 0},
        "source": "data/metadata/dataset_split.csv",
    }

    checks = validate_candidates([candidate], dataset_summary)

    assert checks["all_same_splits"] is True
    assert checks["model_candidates"][0]["seed_registered"] is True
    assert checks["model_candidates"][0]["no_synthetic_used"] is True
    assert checks["all_test_not_modified"] is False


def test_build_final_metrics_reports_positive_improvement_without_using_test_selection() -> None:
    """Final metrics should preserve the validation selection rule and compute deltas."""
    baseline = _candidate(
        result_group="Resultados 1",
        model_id="resnet18_baseline",
        validation_macro_f1=0.5,
        test_macro_f1=0.4,
    )
    r2_base = _candidate(
        result_group="Resultados 2",
        model_id="resnet18_v2",
        validation_macro_f1=0.7,
        test_macro_f1=0.6,
    )
    selected = _candidate(
        result_group="Resultados 2",
        model_id="resnet18_v2_tta_light",
        validation_macro_f1=0.8,
        test_macro_f1=0.7,
    )

    payload = build_final_metrics(
        [baseline, r2_base, selected],
        selected,
        validations={
            "all_same_splits": True,
            "all_seed_registered_or_not_applicable": True,
            "all_no_synthetic_used": True,
            "all_test_not_modified": True,
        },
    )

    assert payload["selection_rule"].startswith("Select by validation macro-F1")
    assert payload["final_model"]["model_id"] == "resnet18_v2_tta_light"
    assert payload["improvement_vs_resultados_1"]["test_macro_f1"] == delta(0.7, 0.4)


def _candidate(
    *,
    model_id: str,
    validation_macro_f1: float,
    test_macro_f1: float,
    result_group: str = "Resultados 2",
    class_distribution: dict[str, dict[str, int]] | None = None,
    predictions_path: str | None = "predictions.csv",
) -> ModelCandidate:
    metrics = {
        "accuracy": test_macro_f1,
        "macro_f1": test_macro_f1,
        "macro_precision": test_macro_f1,
        "macro_recall": test_macro_f1,
        "per_class": {
            class_name: {
                "precision": test_macro_f1,
                "recall": test_macro_f1,
                "f1": test_macro_f1,
                "support": 1,
            }
            for class_name in CLASS_NAMES
        },
    }
    default_distribution = {
        "train": {class_name: 2 for class_name in CLASS_NAMES},
        "validation": {class_name: 1 for class_name in CLASS_NAMES},
        "test": {class_name: 1 for class_name in CLASS_NAMES},
    }
    return ModelCandidate(
        result_group=result_group,
        model_id=model_id,
        architecture="resnet18",
        validation_macro_f1=validation_macro_f1,
        test_metrics=metrics,
        validation_source="validation.json",
        test_source="test.json",
        config_source="config.yaml",
        checkpoint="checkpoint.pt",
        seed=42,
        class_distribution=class_distribution or default_distribution,
        predictions_path=predictions_path,
    )
