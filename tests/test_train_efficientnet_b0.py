from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from src.vision.dataset import EXPECTED_CLASSES
from src.vision.model import (
    create_model,
    freeze_backbone,
    unfreeze_layer3_layer4_and_head,
    unfreeze_layer4_and_head,
)
from src.vision.train_v2 import train_efficientnet_b0


def test_create_efficientnet_b0_replaces_classifier_for_five_classes() -> None:
    """EfficientNet-B0 should expose the requested five-class classifier."""
    model = create_model(
        architecture="efficientnet_b0",
        num_classes=5,
        pretrained=False,
        dropout=0.2,
    )

    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == 5


def test_efficientnet_progressive_unfreeze_targets_last_feature_blocks() -> None:
    """Progressive phases should train head, last block, then last two blocks."""
    model = TinyEfficientNetLike(num_classes=5, dropout=0.2)

    freeze_backbone(model)
    assert _trainable_names(model) == {
        "classifier.1.weight",
        "classifier.1.bias",
    }

    unfreeze_layer4_and_head(model)
    assert _trainable_names(model) == {
        "features.2.weight",
        "features.2.bias",
        "classifier.1.weight",
        "classifier.1.bias",
    }

    unfreeze_layer3_layer4_and_head(model)
    assert _trainable_names(model) == {
        "features.1.weight",
        "features.1.bias",
        "features.2.weight",
        "features.2.bias",
        "classifier.1.weight",
        "classifier.1.bias",
    }


def test_train_efficientnet_b0_writes_comparison_outputs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A tiny EfficientNet run should write prefixed metrics and comparison files."""
    data_root = tmp_path / "processed"
    _make_image_dataset(data_root, images_per_class=1, image_size=44)
    _write_reference_metrics(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.vision.train_v2.create_model", _tiny_model_factory)

    output_dir = tmp_path / "results" / "vision" / "resultados_2_mejoras" / "08_comparacion_modelos"
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
            "architecture": "efficientnet_b0",
            "pretrained": False,
            "num_classes": 5,
            "dropout": 0.2,
        },
        "data": {"root": str(data_root), "num_workers": 0},
        "preprocessing": {"auto_crop": True},
        "classes": list(EXPECTED_CLASSES),
        "output": {
            "checkpoint_path": str(tmp_path / "models" / "vision" / "efficientnet.pt"),
            "results_dir": str(output_dir),
            "metrics_validation_file": "efficientnet_metrics_validation.json",
            "metrics_test_file": "efficientnet_metrics_test.json",
            "classification_report_file": "efficientnet_classification_report.csv",
            "predictions_file": "efficientnet_predictions_test.csv",
        },
    }

    summary = train_efficientnet_b0(config=config, device_name="cpu")

    assert summary["model_selection"]["selected_model"] in {
        "resnet18_v2",
        "efficientnet_b0_v2",
    }
    assert (output_dir / "efficientnet_metrics_validation.json").exists()
    assert (output_dir / "efficientnet_metrics_test.json").exists()
    assert (output_dir / "efficientnet_classification_report.csv").exists()
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "model_selection.json").exists()
    assert (output_dir / "latency_comparison.csv").exists()
    assert (output_dir / "parameter_comparison.csv").exists()
    assert (output_dir / "r2_resnet18_vs_efficientnet_f1.png").exists()
    assert (output_dir / "r2_matrices_modelos.png").exists()
    assert (tmp_path / "configs" / "production_vision_model.yaml").exists()


class TinyEfficientNetLike(nn.Module):
    """Small EfficientNet-shaped model for tests."""

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3 * 32 * 32, 24),
            nn.Linear(24, 16),
            nn.Linear(16, 8),
        )
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(8, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = torch.flatten(inputs, start_dim=1)
        features = self.features(flattened)
        return self.classifier(features)


class TinyResNetLike(nn.Module):
    """Small ResNet-shaped model for comparison tests."""

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.layer3 = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16))
        self.layer4 = nn.Sequential(nn.Linear(16, 8))
        self.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(8, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.layer4(self.layer3(inputs)))


def _tiny_model_factory(
    architecture: str,
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    assert pretrained is False
    if architecture == "efficientnet_b0":
        return TinyEfficientNetLike(num_classes=num_classes, dropout=dropout)
    if architecture == "resnet18":
        return TinyResNetLike(num_classes=num_classes, dropout=dropout)
    raise AssertionError(f"Unexpected architecture: {architecture}")


def _trainable_names(model: nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


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


def _write_reference_metrics(root: Path) -> None:
    baseline_dir = root / "results" / "vision" / "resultados_1_baseline"
    resnet_dir = root / "results" / "vision" / "resultados_2_mejoras" / "05_resnet18_v2"
    baseline_dir.mkdir(parents=True)
    resnet_dir.mkdir(parents=True)
    metrics = _metrics_payload(value=0.5)
    (baseline_dir / "metrics_test.json").write_text(metrics, encoding="utf-8")
    (resnet_dir / "metrics_validation.json").write_text(metrics, encoding="utf-8")
    (resnet_dir / "metrics_test.json").write_text(metrics, encoding="utf-8")
    rows = ["image_path,true_label,predicted_label,predicted_probability"]
    rows.extend(f"{class_name}.jpg,{class_name},{class_name},1.0" for class_name in EXPECTED_CLASSES)
    (resnet_dir / "predictions_test.csv").write_text("\n".join(rows), encoding="utf-8")


def _metrics_payload(value: float) -> str:
    per_class = ",\n".join(
        (
            f'    "{class_name}": '
            f'{{"precision": {value}, "recall": {value}, "f1": {value}, "support": 1}}'
        )
        for class_name in EXPECTED_CLASSES
    )
    return f"""
{{
  "accuracy": {value},
  "macro_f1": {value},
  "macro_precision": {value},
  "macro_recall": {value},
  "per_class": {{
{per_class}
  }}
}}
""".strip()
