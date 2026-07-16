from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from src.vision.gradcam import find_last_convolutional_layer, generate_gradcam_with_fallback


class TinyConvClassifier(torch.nn.Module):
    """Small CNN used to exercise Grad-CAM hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


class TinyLinearClassifier(torch.nn.Module):
    """Model without Conv2d layers, used to validate fallback behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(3 * 16 * 16, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.flatten(1))


def test_find_last_convolutional_layer_selects_final_conv() -> None:
    model = TinyConvClassifier()

    name, layer = find_last_convolutional_layer(model)

    assert name == "features.2"
    assert isinstance(layer, torch.nn.Conv2d)


def test_generate_gradcam_with_fallback_returns_heatmap_for_conv_model() -> None:
    model = TinyConvClassifier()
    image = Image.new("RGB", (16, 16), color=(180, 170, 120))
    transform = transforms.ToTensor()

    result = generate_gradcam_with_fallback(
        model=model,
        image=image,
        transform=transform,
        device=torch.device("cpu"),
        target_class_index=1,
    )

    assert result.available
    assert result.heatmap.shape == (16, 16)
    assert 0.0 <= result.intensity <= 1.0
    assert result.target_layer_name == "features.2"


def test_generate_gradcam_with_fallback_does_not_raise_without_conv_layer() -> None:
    model = TinyLinearClassifier()
    image = Image.new("RGB", (16, 16), color="white")
    transform = transforms.ToTensor()

    result = generate_gradcam_with_fallback(
        model=model,
        image=image,
        transform=transform,
        device=torch.device("cpu"),
        target_class_index=0,
    )

    assert not result.available
    assert result.status == "fallback"
    assert result.target_layer_name == "fallback"
    assert result.heatmap.shape == (16, 16)
