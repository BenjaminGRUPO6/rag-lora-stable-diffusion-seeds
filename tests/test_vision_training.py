from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from src.vision.dataset import EXPECTED_CLASSES
from src.vision.train import train_experiment


def test_train_experiment_writes_checkpoint_and_metrics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A tiny run should cover load, forward, backward, checkpoint and outputs."""
    data_root = tmp_path / "processed"
    _make_image_dataset(data_root, images_per_class=1, image_size=40)
    monkeypatch.setattr("src.vision.train.create_model", _tiny_model_factory)

    config = {
        "model": {
            "architecture": "resnet18",
            "pretrained": False,
            "num_classes": 5,
            "dropout": 0.2,
        },
        "data": {
            "root": str(data_root),
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
        },
        "training": {
            "seed": 42,
            "epochs": 1,
            "frozen_epochs": 1,
            "learning_rate_head": 0.001,
            "learning_rate_backbone": 0.0001,
            "weight_decay": 0.0001,
            "patience": 4,
            "mixed_precision": True,
            "monitor_metric": "macro_f1",
        },
        "classes": list(EXPECTED_CLASSES),
        "output": {
            "model_dir": str(tmp_path / "models"),
            "results_dir": str(tmp_path / "results"),
        },
    }

    summary = train_experiment(config=config, device_name="cpu")

    checkpoint_path = tmp_path / "models" / "resnet18_baseline_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["class_to_idx"]["intact"] == 0
    assert (tmp_path / "results" / "training_history.csv").exists()
    assert (tmp_path / "results" / "metrics_validation.json").exists()
    assert (tmp_path / "results" / "metrics_test.json").exists()
    assert (tmp_path / "results" / "test_predictions.csv").exists()
    assert summary["epochs_ran"] == 1


class TinyResNetLike(nn.Module):
    """Small ResNet-shaped model used to avoid downloading pretrained weights."""

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 16),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(16, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer4(inputs))


def _tiny_model_factory(
    architecture: str,
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> TinyResNetLike:
    assert architecture == "resnet18"
    assert pretrained is False
    return TinyResNetLike(num_classes=num_classes, dropout=dropout)


def _make_image_dataset(root: Path, images_per_class: int, image_size: int) -> None:
    for split in ("train", "validation", "test"):
        for class_index, class_name in enumerate(EXPECTED_CLASSES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(images_per_class):
                image = Image.new(
                    "RGB",
                    (image_size, image_size),
                    color=(40 + class_index * 20, 30 + image_index * 10, 120),
                )
                image.save(class_dir / f"{class_name}_{image_index}.jpg")
