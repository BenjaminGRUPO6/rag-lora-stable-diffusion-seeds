from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Arquitectura no soportada: {architecture}")
