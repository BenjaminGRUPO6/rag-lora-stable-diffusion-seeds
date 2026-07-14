from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.vision.calibration import load_temperature, softmax_with_temperature
from src.vision.evaluation import load_checkpoint
from src.vision.model import create_model


ImageInput = str | Path | Image.Image
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class InferenceResult:
    """Serializable output from a single deterministic vision inference."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    logits: dict[str, float]
    top_3: list[dict[str, float | str]]
    uncalibrated_confidence: float
    uncalibrated_probabilities: dict[str, float]
    calibration_temperature: float | None
    calibration_applied: bool
    second_class: str | None
    second_confidence: float | None
    top1_top2_margin: float

    def to_prediction_dict(self) -> dict[str, Any]:
        """Return the dictionary shape consumed by the RAG pipeline and CLI."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "logits": self.logits,
            "top_3": self.top_3,
            "uncalibrated_confidence": self.uncalibrated_confidence,
            "uncalibrated_probabilities": self.uncalibrated_probabilities,
            "calibration_temperature": self.calibration_temperature,
            "calibration_applied": self.calibration_applied,
            "second_class": self.second_class,
            "second_confidence": self.second_confidence,
            "top1_top2_margin": self.top1_top2_margin,
        }


class VisionInferenceEngine:
    """Single inference path for model loading, preprocessing and prediction."""

    def __init__(
        self,
        *,
        model: nn.Module,
        labels: list[str],
        transform: Callable[[Image.Image], torch.Tensor],
        device: torch.device,
        temperature: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.labels = list(labels)
        self.transform = transform
        self.device = device
        if temperature is not None and temperature <= 0.0:
            raise ValueError("temperature must be greater than zero.")
        self.temperature = temperature

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        device: torch.device,
        config: dict[str, Any] | None = None,
        temperature_path: str | Path | None = None,
    ) -> "VisionInferenceEngine":
        """Create an inference engine from a trained ResNet18 checkpoint."""
        checkpoint = Path(checkpoint_path)
        model, labels, _ = load_resnet18_checkpoint(
            checkpoint_path=checkpoint,
            device=device,
            config=config,
        )
        image_size = int(get_nested(config or {}, ("data", "image_size"), 224))
        resolved_temperature_path = (
            Path(temperature_path)
            if temperature_path is not None
            else default_temperature_path(checkpoint)
        )
        temperature = load_temperature(resolved_temperature_path)
        return cls(
            model=model,
            labels=labels,
            transform=build_inference_transform(image_size=image_size),
            device=device,
            temperature=temperature,
        )

    def predict(self, image: ImageInput) -> InferenceResult:
        """Run deterministic single-image inference with logits and top-3 output."""
        pil_image = load_rgb_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_tensor = self.model(tensor)[0].detach().cpu()
            uncalibrated_probabilities_tensor = torch.softmax(logits_tensor, dim=0)
            probabilities_tensor = (
                softmax_with_temperature(logits_tensor, self.temperature)
                if self.temperature is not None
                else uncalibrated_probabilities_tensor
            )

        index = int(logits_tensor.argmax().item())
        probabilities = {
            label: float(probabilities_tensor[label_index].item())
            for label_index, label in enumerate(self.labels)
        }
        uncalibrated_probabilities = {
            label: float(uncalibrated_probabilities_tensor[label_index].item())
            for label_index, label in enumerate(self.labels)
        }
        logits = {
            label: float(logits_tensor[label_index].item())
            for label_index, label in enumerate(self.labels)
        }
        sorted_indices = torch.argsort(probabilities_tensor, descending=True).tolist()
        top_indices = sorted_indices[: min(3, len(self.labels))]
        top_3 = [
            {
                "label": self.labels[int(label_index)],
                "probability": float(probabilities_tensor[int(label_index)].item()),
            }
            for label_index in top_indices
        ]
        second_index = int(sorted_indices[1]) if len(sorted_indices) >= 2 else None
        second_confidence = (
            float(probabilities_tensor[second_index].item()) if second_index is not None else None
        )
        top1_top2_margin = (
            float(probabilities_tensor[index].item()) - second_confidence
            if second_confidence is not None
            else float(probabilities_tensor[index].item())
        )
        return InferenceResult(
            label=self.labels[index],
            confidence=float(probabilities_tensor[index].item()),
            probabilities=probabilities,
            logits=logits,
            top_3=top_3,
            uncalibrated_confidence=float(uncalibrated_probabilities_tensor[index].item()),
            uncalibrated_probabilities=uncalibrated_probabilities,
            calibration_temperature=self.temperature,
            calibration_applied=self.temperature is not None,
            second_class=self.labels[second_index] if second_index is not None else None,
            second_confidence=second_confidence,
            top1_top2_margin=top1_top2_margin,
        )

    def predict_dict(self, image: ImageInput) -> dict[str, Any]:
        """Run inference and return the pipeline-compatible dictionary."""
        return self.predict(image).to_prediction_dict()


def labels_from_class_to_idx(class_to_idx: dict[str, int]) -> list[str]:
    """Return class labels ordered by their numeric index."""
    return [
        label
        for label, _ in sorted(class_to_idx.items(), key=lambda item: int(item[1]))
    ]


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Build the deterministic production transform used by app and CLI inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def default_temperature_path(checkpoint_path: str | Path) -> Path:
    """Return the default temperature JSON path for a checkpoint."""
    checkpoint = Path(checkpoint_path)
    stem = checkpoint.stem
    if stem.endswith("_best"):
        stem = stem[: -len("_best")]
    return checkpoint.with_name(f"{stem}_temperature.json")


def load_resnet18_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    """Load a trained ResNet18 checkpoint and return model, labels and checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    checkpoint_config = checkpoint.get("config", {})
    model_config: dict[str, Any] = {}
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        model_config.update(checkpoint_config["model"])
    if config and isinstance(config.get("model"), dict):
        model_config.update(config["model"])

    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("El checkpoint no contiene class_to_idx valido.")
    labels = labels_from_class_to_idx(
        {str(key): int(value) for key, value in class_to_idx.items()}
    )

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


def load_rgb_image(image: ImageInput) -> Image.Image:
    """Return an RGB PIL image from a path or PIL image input."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(Path(image)) as opened:
        return opened.convert("RGB")


def predict_image(
    model: nn.Module,
    image_path: ImageInput,
    transform: Callable[[Image.Image], torch.Tensor],
    labels: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Run single-image inference through the shared engine."""
    engine = VisionInferenceEngine(
        model=model,
        labels=labels,
        transform=transform,
        device=device,
    )
    return engine.predict_dict(image_path)


def get_nested(config: dict[str, Any], keys: tuple[str, ...], default: object) -> object:
    """Read a nested config value with a default."""
    value: object = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
