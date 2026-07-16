from __future__ import annotations

import torch.nn as nn

from src.vision.model import (
    create_model,
    freeze_backbone,
    get_parameter_groups,
    unfreeze_layer3_layer4_and_head,
    unfreeze_layer4_and_head,
)


def test_create_model_replaces_resnet18_head_without_downloading_weights() -> None:
    """The classifier head must be Dropout plus Linear for five classes."""
    model = create_model(
        architecture="resnet18",
        num_classes=5,
        pretrained=False,
        dropout=0.2,
    )

    assert isinstance(model.fc, nn.Sequential)
    assert isinstance(model.fc[0], nn.Dropout)
    assert model.fc[0].p == 0.2
    assert isinstance(model.fc[1], nn.Linear)
    assert model.fc[1].out_features == 5


def test_freeze_and_parameter_groups_train_head_then_layer4() -> None:
    """Training phases should expose only the requested parameters."""
    model = create_model(
        architecture="resnet18",
        num_classes=5,
        pretrained=False,
        dropout=0.2,
    )

    freeze_backbone(model)
    assert all(not parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    head_groups = get_parameter_groups(
        model,
        learning_rate_head=0.001,
        learning_rate_backbone=0.0001,
    )
    assert [group["name"] for group in head_groups] == ["head"]

    unfreeze_layer4_and_head(model)
    assert all(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    groups = get_parameter_groups(
        model,
        learning_rate_head=0.001,
        learning_rate_backbone=0.0001,
    )
    assert [group["name"] for group in groups] == ["layer4", "head"]
    assert [group["lr"] for group in groups] == [0.0001, 0.001]

    unfreeze_layer3_layer4_and_head(model)
    assert all(parameter.requires_grad for parameter in model.layer3.parameters())
    assert all(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    wider_groups = get_parameter_groups(
        model,
        learning_rate_head=0.001,
        learning_rate_backbone=0.0001,
    )
    assert [group["name"] for group in wider_groups] == ["layer3", "layer4", "head"]
