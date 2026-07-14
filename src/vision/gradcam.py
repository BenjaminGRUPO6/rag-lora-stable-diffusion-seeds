from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.vision.inference import ImageInput, load_rgb_image
from src.vision.visualization import fallback_heatmap, normalize_intensity, overlay_heatmap


@dataclass(frozen=True)
class GradCamResult:
    """Serializable Grad-CAM output and display artifacts."""

    heatmap: np.ndarray
    overlay: Image.Image
    intensity: float
    target_class_index: int
    target_layer_name: str
    status: str
    message: str

    @property
    def available(self) -> bool:
        """Return true when the heatmap came from gradients instead of fallback."""
        return self.status == "ok"


def find_last_convolutional_layer(model: nn.Module) -> tuple[str, nn.Conv2d]:
    """Return the final Conv2d layer in a supported CNN model."""
    selected: tuple[str, nn.Conv2d] | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            selected = (name, module)
    if selected is None:
        raise ValueError("No se encontro una capa convolucional compatible para Grad-CAM.")
    return selected


def generate_gradcam(
    *,
    model: nn.Module,
    image: ImageInput,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    target_class_index: int | None = None,
    target_layer: nn.Module | None = None,
    target_layer_name: str | None = None,
) -> GradCamResult:
    """Generate Grad-CAM for one image and class index."""
    pil_image = load_rgb_image(image)
    layer_name, layer = (
        (target_layer_name or "custom_layer", target_layer)
        if target_layer is not None
        else find_last_convolutional_layer(model)
    )
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def save_gradient(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        gradients.append(grad_output[0])

    forward_hook = layer.register_forward_hook(save_activation)
    backward_hook = layer.register_full_backward_hook(save_gradient)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        tensor = transform(pil_image).unsqueeze(0).to(device)
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("La salida del modelo no tiene forma de clasificacion esperada.")
        class_index = int(target_class_index) if target_class_index is not None else int(logits.argmax(dim=1).item())
        if class_index < 0 or class_index >= int(logits.shape[1]):
            raise ValueError("target_class_index fuera de rango.")
        score = logits[0, class_index]
        score.backward()
        if not activations or not gradients:
            raise RuntimeError("No se capturaron activaciones o gradientes para Grad-CAM.")
        heatmap = build_gradcam_map(activations[-1], gradients[-1])
        overlay = overlay_heatmap(pil_image, heatmap)
        return GradCamResult(
            heatmap=heatmap,
            overlay=overlay,
            intensity=float(np.mean(heatmap)),
            target_class_index=class_index,
            target_layer_name=layer_name,
            status="ok",
            message="Grad-CAM generado como explicacion aproximada del modelo.",
        )
    finally:
        forward_hook.remove()
        backward_hook.remove()
        model.zero_grad(set_to_none=True)


def build_gradcam_map(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
    """Convert captured activations and gradients into a normalized heatmap."""
    if activation.ndim != 4 or gradient.ndim != 4:
        raise ValueError("Grad-CAM requiere tensores 4D de activacion y gradiente.")
    activation_map = activation.detach()[0]
    gradient_map = gradient.detach()[0]
    weights = gradient_map.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activation_map).sum(dim=0))
    return normalize_intensity(cam.detach().cpu().numpy())


def generate_gradcam_with_fallback(
    *,
    model: nn.Module,
    image: ImageInput,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    target_class_index: int | None = None,
) -> GradCamResult:
    """Generate Grad-CAM and return a neutral fallback if any compatible step fails."""
    pil_image = load_rgb_image(image)
    try:
        return generate_gradcam(
            model=model,
            image=pil_image,
            transform=transform,
            device=device,
            target_class_index=target_class_index,
        )
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        heatmap = fallback_heatmap(pil_image.size)
        return GradCamResult(
            heatmap=heatmap,
            overlay=overlay_heatmap(pil_image, heatmap, alpha=0.20),
            intensity=0.0,
            target_class_index=int(target_class_index or 0),
            target_layer_name="fallback",
            status="fallback",
            message=f"Grad-CAM no disponible: {exc.__class__.__name__}.",
        )
