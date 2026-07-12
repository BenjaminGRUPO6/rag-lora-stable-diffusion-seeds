from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def predict_image(model, image_path: str | Path, transform, labels: list[str], device: torch.device) -> dict:
    model.eval()
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax().item())
    return {
        "label": labels[index],
        "confidence": float(probabilities[index].item()),
        "probabilities": {label: float(probabilities[i].item()) for i, label in enumerate(labels)},
    }
