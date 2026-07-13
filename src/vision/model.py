from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models


def create_model(
    architecture: str = "resnet18",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Create the supported vision classifier for Experiment A."""
    if architecture.lower() != "resnet18":
        raise ValueError(f"Unsupported architecture for Experiment A: {architecture}")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all ResNet18 layers except the classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_layer4_and_head(model: nn.Module) -> None:
    """Train only the final residual block and classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_layer3_layer4_and_head(model: nn.Module) -> None:
    """Train the final two residual blocks and classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer3.parameters():
        parameter.requires_grad = True
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def get_parameter_groups(
    model: nn.Module,
    learning_rate_head: float,
    learning_rate_backbone: float,
) -> list[dict[str, Any]]:
    """Return optimizer parameter groups with separate head and backbone rates."""
    groups: list[dict[str, Any]] = []
    layer3_parameters = [
        parameter for parameter in model.layer3.parameters() if parameter.requires_grad
    ]
    backbone_parameters = [
        parameter for parameter in model.layer4.parameters() if parameter.requires_grad
    ]
    head_parameters = [parameter for parameter in model.fc.parameters() if parameter.requires_grad]
    if layer3_parameters:
        groups.append(
            {
                "params": layer3_parameters,
                "lr": learning_rate_backbone,
                "name": "layer3",
            }
        )
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": learning_rate_backbone,
                "name": "layer4",
            }
        )
    if head_parameters:
        groups.append(
            {
                "params": head_parameters,
                "lr": learning_rate_head,
                "name": "head",
            }
        )
    if not groups:
        raise ValueError("No trainable parameters found for optimizer.")
    return groups
