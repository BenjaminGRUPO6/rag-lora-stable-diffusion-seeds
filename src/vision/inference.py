from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.vision.evaluation import load_checkpoint
from src.vision.model import create_model


ImageInput = str | Path | Image.Image


def labels_from_class_to_idx(class_to_idx: dict[str, int]) -> list[str]:
    """Return class labels ordered by their numeric index."""
    return [
        label
        for label, _ in sorted(class_to_idx.items(), key=lambda item: int(item[1]))
    ]


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Build the deterministic transform used by ResNet18 inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_resnet18_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    """Load a trained ResNet18 checkpoint and return model, labels and checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    checkpoint_config = checkpoint.get("config", {})
    model_config = {}
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        model_config.update(checkpoint_config["model"])
    if config and isinstance(config.get("model"), dict):
        model_config.update(config["model"])

    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("El checkpoint no contiene class_to_idx valido.")
    labels = labels_from_class_to_idx({str(key): int(value) for key, value in class_to_idx.items()})

    model = create_model(
        architecture=str(model_config.get("architecture", "resnet18")),
        num_classes=len(labels),
        pretrained=False,
        dropout=float(model_config.get("dropout", 0.2)),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("El checkpoint no contiene model_state_dict valido.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, labels, checkpoint


def _load_image(image: ImageInput) -> Image.Image:
    """Return an RGB PIL image from a path or PIL image input."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(Path(image)) as opened:
        return opened.convert("RGB")


def predict_image(
    model: nn.Module,
    image_path: ImageInput,
    transform,
    labels: list[str],
    device: torch.device,
) -> dict:
    """Run single-image inference and return label, confidence and probabilities."""
    model.eval()
    image = _load_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax().item())
    return {
        "label": labels[index],
        "confidence": float(probabilities[index].item()),
        "probabilities": {label: float(probabilities[i].item()) for i, label in enumerate(labels)},
    }
