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
    """Create a supported ImageNet-pretrained vision classifier."""
    architecture_name = architecture.lower()
    if architecture_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        return model
    if architecture_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    raise ValueError(f"Unsupported vision architecture: {architecture}")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone layers except the classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in _classifier_head(model).parameters():
        parameter.requires_grad = True


def unfreeze_layer4_and_head(model: nn.Module) -> None:
    """Train only the final backbone block and classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in _last_backbone_blocks(model, count=1):
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in _classifier_head(model).parameters():
        parameter.requires_grad = True


def unfreeze_layer3_layer4_and_head(model: nn.Module) -> None:
    """Train the final two backbone blocks and classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in _last_backbone_blocks(model, count=2):
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in _classifier_head(model).parameters():
        parameter.requires_grad = True


def get_parameter_groups(
    model: nn.Module,
    learning_rate_head: float,
    learning_rate_backbone: float,
) -> list[dict[str, Any]]:
    """Return optimizer parameter groups with separate head and backbone rates."""
    groups: list[dict[str, Any]] = []
    head_ids = {id(parameter) for parameter in _classifier_head(model).parameters()}
    named_backbone_groups: list[tuple[str, list[nn.Parameter]]] = []
    if hasattr(model, "layer3") and hasattr(model, "layer4"):
        for group_name in ("layer3", "layer4"):
            layer = getattr(model, group_name)
            parameters = [
                parameter
                for parameter in layer.parameters()
                if parameter.requires_grad and id(parameter) not in head_ids
            ]
            if parameters:
                named_backbone_groups.append((group_name, parameters))
    else:
        backbone_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        if backbone_parameters:
            named_backbone_groups.append(("backbone", backbone_parameters))
    head_parameters = [
        parameter
        for parameter in _classifier_head(model).parameters()
        if parameter.requires_grad
    ]
    for backbone_group_name, backbone_parameters in named_backbone_groups:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": learning_rate_backbone,
                "name": backbone_group_name,
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


def _classifier_head(model: nn.Module) -> nn.Module:
    """Return the classifier head for supported torchvision architectures."""
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise ValueError("Unsupported model head layout.")


def _last_backbone_blocks(model: nn.Module, count: int) -> list[nn.Module]:
    """Return the last trainable backbone blocks for progressive unfreezing."""
    if hasattr(model, "layer3") and hasattr(model, "layer4"):
        blocks = [model.layer3, model.layer4]
        return blocks[-count:]
    features = getattr(model, "features", None)
    if isinstance(features, nn.Sequential):
        blocks = list(features.children())
        return blocks[-count:]
    raise ValueError("Unsupported model backbone layout.")
