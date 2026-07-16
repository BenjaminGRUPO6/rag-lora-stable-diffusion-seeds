from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_vision_errors import (
    REVIEW_COLUMNS,
    build_quality_error_summary,
    build_recommendations,
    error_categories,
    quality_status,
    save_confusion_outputs,
    save_review_csv,
    top_two,
)


def test_top_two_returns_confidence_second_option_and_margin() -> None:
    """Top-2 extraction must preserve the class names and numeric margin."""
    label, confidence, second_label, second_probability, margin = top_two(
        [0.12, 0.51, 0.08, 0.21, 0.08],
        ["intact", "spotted", "immature", "broken", "skin_damaged"],
    )

    assert label == "spotted"
    assert confidence == 0.51
    assert second_label == "broken"
    assert second_probability == 0.21
    assert margin == pytest.approx(0.30)


def test_error_categories_are_non_mutually_exclusive() -> None:
    """A validation case can be both a class confusion and low-certainty case."""
    categories = error_categories(
        true_label="intact",
        predicted_label="broken",
        confidence=0.55,
        margin=0.04,
        confidence_threshold=0.60,
        margin_threshold=0.15,
    )

    assert categories == [
        "true_intact_predicted_other",
        "predicted_broken_true_other",
        "low_confidence",
        "low_top1_top2_margin",
    ]


def test_quality_status_prioritizes_fallback_then_warnings() -> None:
    """Quality status should expose crop fallback before secondary warnings."""
    assert quality_status({"used_fallback": True, "crop_confidence": 0.2}) == "fallback_crop"
    assert (
        quality_status(
            {
                "used_fallback": False,
                "crop_confidence": 0.5,
                "quality_warnings": "possible_blur",
            }
        )
        == "low_crop_confidence"
    )
    assert (
        quality_status(
            {
                "used_fallback": False,
                "crop_confidence": 0.9,
                "quality_warnings": "possible_blur",
            }
        )
        == "warning:possible_blur"
    )
    assert quality_status({"used_fallback": False, "crop_confidence": 0.9}) == "ok"


def test_confusion_and_quality_summaries_are_counted(tmp_path: Path) -> None:
    """Confusion tables and quality error rates must use validation rows only."""
    rows = [
        _row("a", "intact", "intact", True, "ok"),
        _row("b", "intact", "broken", False, "ok"),
        _row("c", "broken", "intact", False, "warning:possible_blur"),
    ]

    matrix, top_confusions = save_confusion_outputs(
        rows,
        ["intact", "broken"],
        tmp_path,
    )
    quality = build_quality_error_summary(rows)

    assert int(matrix.loc["intact", "broken"]) == 1
    assert int(matrix.loc["broken", "intact"]) == 1
    assert set(top_confusions["count"].tolist()) == {1}
    ok_row = quality[quality["quality_status"] == "ok"].iloc[0]
    warning_row = quality[quality["quality_status"] == "warning:possible_blur"].iloc[0]
    assert ok_row["errors"] == 1
    assert ok_row["total"] == 2
    assert warning_row["error_rate"] == 1.0


def test_review_csv_has_required_columns_without_label_decision(tmp_path: Path) -> None:
    """The review CSV must request human review without declaring corrected labels."""
    rows = [
        {
            **_row("case-1", "intact", "broken", False, "ok"),
            "confidence": 0.8,
            "second_label": "intact",
            "second_probability": 0.1,
            "error_categories": "true_intact_predicted_other",
        },
        {
            **_row("case-2", "broken", "broken", True, "ok"),
            "confidence": 0.9,
            "second_label": "intact",
            "second_probability": 0.05,
            "error_categories": "",
        },
    ]

    frame = save_review_csv(rows, tmp_path / "review.csv")
    saved = pd.read_csv(tmp_path / "review.csv")

    assert list(frame.columns) == list(REVIEW_COLUMNS)
    assert list(saved.columns) == list(REVIEW_COLUMNS)
    assert len(saved) == 1
    assert saved.loc[0, "suspected_label_issue"] == "pending_human_review"
    assert saved.loc[0, "reviewed_by"] != "model"


def test_recommendations_are_scoped_to_validation_observations() -> None:
    """Recommendations must state validation scope and human review policy."""
    rows = [
        {
            **_row("case-1", "intact", "broken", False, "ok"),
            "confidence": 0.55,
            "margin_top1_top2": 0.02,
            "error_categories": "true_intact_predicted_other;low_confidence;low_top1_top2_margin",
        }
    ]
    quality = build_quality_error_summary(rows)
    top_confusions = pd.DataFrame(
        [
            {
                "true_label": "intact",
                "predicted_label": "broken",
                "count": 1,
                "rate_within_true_label": 1.0,
            }
        ]
    )

    payload = build_recommendations(rows, top_confusions, quality)

    assert payload["scope"] == "validation_only"
    assert "No label is declared incorrect" in payload["label_change_policy"]
    assert payload["recommendations"]


def _row(
    image_id: str,
    true_label: str,
    predicted_label: str,
    is_correct: bool,
    status: str,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "is_correct": is_correct,
        "quality_status": status,
    }
