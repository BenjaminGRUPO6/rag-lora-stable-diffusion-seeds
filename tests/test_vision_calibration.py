from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.vision.calibration import (
    classes_unchanged_after_temperature,
    compute_ece,
    fit_temperature,
    softmax_with_temperature,
)
from src.vision.inference import VisionInferenceEngine


class TinyClassifier(torch.nn.Module):
    """Small deterministic classifier for calibration fallback tests."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = int(inputs.shape[0])
        logits = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float32)
        return logits.repeat(batch_size, 1)


def test_fit_temperature_returns_positive_value() -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.5, 2.0, 0.0],
            [0.3, 0.2, 1.5],
            [2.0, 1.8, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    temperature = fit_temperature(logits, labels, max_iter=25)

    assert temperature > 0.0


def test_temperature_scaled_probabilities_sum_to_one() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.0, 0.5, 1.0]], dtype=torch.float32)
    probabilities = softmax_with_temperature(logits, temperature=1.7)

    assert probabilities.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_temperature_scaling_does_not_change_classes() -> None:
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.2, 2.0, 0.1]], dtype=torch.float32)

    assert classes_unchanged_after_temperature(logits, temperature=2.5)


def test_ece_matches_manual_two_bin_example() -> None:
    probabilities = torch.tensor(
        [
            [0.40, 0.35, 0.25],
            [0.45, 0.40, 0.15],
            [0.80, 0.10, 0.10],
            [0.60, 0.30, 0.10],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    ece = compute_ece(probabilities, labels, n_bins=2)

    assert ece == pytest.approx(0.1375)


def test_inference_falls_back_without_calibrator(tmp_path: Path) -> None:
    image = Image.new("RGB", (8, 8), color="white")
    engine = VisionInferenceEngine(
        model=TinyClassifier(),
        labels=["intact", "spotted", "broken"],
        transform=lambda _: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
    )

    prediction = engine.predict_dict(image)

    assert prediction["calibration_applied"] is False
    assert prediction["calibration_temperature"] is None
    assert prediction["confidence"] == pytest.approx(prediction["uncalibrated_confidence"])
    assert prediction["probabilities"] == pytest.approx(prediction["uncalibrated_probabilities"])
