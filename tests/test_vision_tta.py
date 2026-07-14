from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.vision.inference import VisionInferenceEngine
from src.vision.tta import (
    TTASplitResult,
    aggregate_logits,
    available_policy_names,
    get_policy,
    predict_with_tta,
    select_policy,
)


class TinyClassifier(torch.nn.Module):
    """Deterministic classifier for TTA inference tests."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = int(inputs.shape[0])
        return torch.tensor([[3.0, 1.0]], dtype=torch.float32).repeat(batch_size, 1)


def _result(policy: str, macro_f1: float, recall_intact: float, recall_broken: float) -> TTASplitResult:
    return TTASplitResult(
        policy_name=policy,
        split="validation",
        temperature=1.0,
        metrics={
            "accuracy": macro_f1,
            "macro_precision": macro_f1,
            "macro_recall": macro_f1,
            "macro_f1": macro_f1,
            "recall_intact": recall_intact,
            "recall_broken": recall_broken,
            "per_class": {},
        },
        y_true=[0],
        y_pred=[0],
        logits=torch.zeros(1, 2),
        probabilities=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        latency_seconds_total=0.1,
        latency_seconds_per_image=0.1,
        view_count=get_policy(policy).view_count,
    )


def test_tta_policies_match_required_views() -> None:
    assert available_policy_names() == ["none", "light", "standard"]
    assert [view.name for view in get_policy("none").views] == ["original"]
    assert [view.name for view in get_policy("light").views] == [
        "original",
        "horizontal_flip",
    ]
    assert [view.name for view in get_policy("standard").views] == [
        "original",
        "horizontal_flip",
        "rotate_plus_5",
        "rotate_minus_5",
    ]


def test_aggregate_logits_averages_views_explicitly() -> None:
    view_logits = torch.tensor(
        [
            [[2.0, 0.0], [4.0, 2.0]],
            [[0.0, 3.0], [2.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    averaged = aggregate_logits(view_logits)

    assert torch.allclose(
        averaged,
        torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32),
    )


def test_select_policy_keeps_none_without_macro_f1_improvement() -> None:
    selected, enabled = select_policy(
        [
            _result("none", macro_f1=0.80, recall_intact=0.70, recall_broken=0.70),
            _result("light", macro_f1=0.80, recall_intact=0.90, recall_broken=0.90),
        ]
    )

    assert selected.policy_name == "none"
    assert enabled is False


def test_select_policy_enables_tta_only_when_validation_macro_f1_improves() -> None:
    selected, enabled = select_policy(
        [
            _result("none", macro_f1=0.80, recall_intact=0.90, recall_broken=0.90),
            _result("light", macro_f1=0.85, recall_intact=0.70, recall_broken=0.70),
        ]
    )

    assert selected.policy_name == "light"
    assert enabled is True


def test_predict_with_tta_returns_tta_metadata() -> None:
    engine = VisionInferenceEngine(
        model=TinyClassifier(),
        labels=["intact", "broken"],
        transform=lambda _: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
    )

    prediction = predict_with_tta(
        engine=engine,
        image=Image.new("RGB", (8, 8), color="white"),
        policy_name="light",
        temperature=1.5,
    )

    assert prediction["label"] == "intact"
    assert prediction["tta_enabled"] is True
    assert prediction["tta_policy"] == "light"
    assert prediction["tta_views"] == 2
    assert prediction["aggregation"] == "mean_logits"
    assert prediction["calibration_temperature"] == pytest.approx(1.5)
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
