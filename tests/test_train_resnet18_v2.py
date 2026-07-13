from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from src.vision.dataset import EXPECTED_CLASSES
from src.vision.train_v2 import (
    AutoCropTransform,
    build_class_weights,
    build_v2_transforms,
    phase_for_epoch,
    train_resnet18_v2,
)


def test_v2_phase_schedule_reaches_all_progressive_phases() -> None:
    """The V2 schedule should train head, then layer4, then layer3 plus layer4."""
    config = {"frozen_head_epochs": 2, "layer4_epochs": 3}

    assert phase_for_epoch(epoch=1, config=config) == "head"
    assert phase_for_epoch(epoch=3, config=config) == "layer4"
    assert phase_for_epoch(epoch=6, config=config) == "layer3_layer4"


def test_v2_eval_transform_is_deterministic_with_auto_crop() -> None:
    """Validation/test preprocessing should be deterministic and random-free."""
    image = Image.new("RGB", (48, 40), color=(240, 240, 240))
    transform = build_v2_transforms(image_size=32, train=False, auto_crop=True)

    first = transform(image)
    second = transform(image)

    assert isinstance(transform.transforms[0], AutoCropTransform)
    assert torch.equal(first, second)
    assert tuple(first.shape) == (3, 32, 32)


def test_class_weights_are_optional_and_train_only() -> None:
    """Class weights are disabled by default and calculable from train counts."""
    distribution = {
        "intact": 2,
        "spotted": 4,
        "immature": 2,
        "broken": 2,
        "skin_damaged": 2,
    }

    assert build_class_weights(
        distribution=distribution,
        class_names=EXPECTED_CLASSES,
        device=torch.device("cpu"),
        enabled=False,
    ) is None
    weights = build_class_weights(
        distribution=distribution,
        class_names=EXPECTED_CLASSES,
        device=torch.device("cpu"),
        enabled=True,
    )

    assert weights is not None
    assert weights.shape == (5,)
    assert float(weights[1]) < float(weights[0])


def test_train_resnet18_v2_writes_required_outputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A tiny V2 run should write required artifacts without downloading weights."""
    data_root = tmp_path / "processed"
    _make_image_dataset(data_root, images_per_class=1, image_size=44)
    baseline_dir = tmp_path / "results" / "vision" / "resultados_1_baseline"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "metrics_test.json").write_text(
        """
{
  "accuracy": 0.5,
  "macro_f1": 0.5,
  "macro_precision": 0.5,
  "macro_recall": 0.5,
  "per_class": {
    "intact": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
    "spotted": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
    "immature": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
    "broken": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
    "skin_damaged": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1}
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.vision.train_v2.create_model", _tiny_model_factory)

    config = {
        "seed": 42,
        "image_size": 32,
        "batch_size": 2,
        "fallback_batch_size": 1,
        "max_epochs": 1,
        "patience": 1,
        "frozen_head_epochs": 1,
        "layer4_epochs": 1,
        "learning_rate_head": 0.001,
        "learning_rate_backbone": 0.00003,
        "weight_decay": 0.0001,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "mixed_precision": True,
        "monitor": "validation_macro_f1",
        "use_class_weights": False,
        "model": {
            "architecture": "resnet18",
            "pretrained": False,
            "num_classes": 5,
            "dropout": 0.2,
        },
        "data": {"root": str(data_root), "num_workers": 0},
        "preprocessing": {"auto_crop": True},
        "classes": list(EXPECTED_CLASSES),
        "output": {
            "checkpoint_path": str(tmp_path / "models" / "vision" / "resnet18_v2_best.pt"),
            "results_dir": str(
                tmp_path / "results" / "vision" / "resultados_2_mejoras" / "05_resnet18_v2"
            ),
        },
    }

    summary = train_resnet18_v2(config=config, device_name="cpu")
    output_dir = Path(config["output"]["results_dir"])

    assert summary["best_epoch"] == 1
    assert Path(config["output"]["checkpoint_path"]).exists()
    assert (output_dir / "training_history.csv").exists()
    assert (output_dir / "metrics_validation.json").exists()
    assert (output_dir / "metrics_test.json").exists()
    assert (output_dir / "classification_report.csv").exists()
    assert (output_dir / "predictions_test.csv").exists()
    assert (output_dir / "run_config.yaml").exists()
    assert (output_dir / "run_summary.json").exists()
    assert (output_dir / "r2_curvas_entrenamiento_resnet18_v2.png").exists()
    assert (output_dir / "r1_vs_r2_f1_resnet18.png").exists()


class TinyResNetLike(nn.Module):
    """Small ResNet-shaped model for tests."""

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 32),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(16, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer4(self.layer3(inputs)))


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
                    color=(50 + class_index * 25, 40 + image_index * 5, 170),
                )
                image.save(class_dir / f"{class_name}_{image_index}.jpg")
