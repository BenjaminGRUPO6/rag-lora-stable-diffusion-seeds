from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.vision.dataset import EXPECTED_CLASSES, create_dataloaders
from src.vision.evaluation import evaluate_model, load_checkpoint, save_evaluation_outputs
from src.vision.model import (
    create_model,
    freeze_backbone,
    get_parameter_groups,
    unfreeze_layer4_and_head,
)


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected for one epoch and split."""

    loss: float
    accuracy: float
    macro_f1: float


def train_experiment(
    config: dict[str, Any],
    device_name: str | None = None,
    smoke_test: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Train Experiment A and evaluate the best validation checkpoint once on test."""
    config = _prepare_config(config=config, smoke_test=smoke_test)
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = resolve_device(device_name)
    data_config = config["data"]
    training_config = config["training"]
    output_config = config["output"]
    class_names = list(config.get("classes", EXPECTED_CLASSES))

    model_dir = Path(output_config["model_dir"])
    results_dir = Path(output_config["results_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(config=config, smoke_test=smoke_test)
    _write_yaml(results_dir / "run_config.yaml", config)

    dataloaders = create_dataloaders(
        data_root=data_config["root"],
        classes=class_names,
        image_size=int(data_config["image_size"]),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        seed=seed,
        smoke_test=smoke_test,
    )
    model = create_model(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    class_weights, class_weights_used = _class_weights(
        distribution=dataloaders.distributions["train"],
        class_names=class_names,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_epochs = int(training_config["epochs"])
    frozen_epochs = int(training_config["frozen_epochs"])
    best_validation_macro_f1 = -1.0
    start_epoch = 1

    current_phase = _configure_phase(
        model=model,
        epoch=start_epoch,
        frozen_epochs=frozen_epochs,
    )
    optimizer = _build_optimizer(model=model, config=config)
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_macro_f1 = float(checkpoint.get("best_validation_macro_f1", -1.0))
        current_phase = _configure_phase(
            model=model,
            epoch=start_epoch,
            frozen_epochs=frozen_epochs,
        )
        optimizer = _build_optimizer(model=model, config=config)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            optimizer = _build_optimizer(model=model, config=config)

    use_amp = bool(training_config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0

    for epoch in range(start_epoch, total_epochs + 1):
        desired_phase = _phase_for_epoch(epoch=epoch, frozen_epochs=frozen_epochs)
        if desired_phase != current_phase:
            current_phase = _configure_phase(
                model=model,
                epoch=epoch,
                frozen_epochs=frozen_epochs,
            )
            optimizer = _build_optimizer(model=model, config=config)

        started_at = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=dataloaders.train,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=dataloaders.validation,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
        )
        epoch_time = time.perf_counter() - started_at
        learning_rate = max(group["lr"] for group in optimizer.param_groups)
        history.append(
            {
                "epoch": epoch,
                "phase": current_phase,
                "train_loss": train_metrics.loss,
                "validation_loss": validation_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "validation_accuracy": validation_metrics.accuracy,
                "validation_macro_f1": validation_metrics.macro_f1,
                "learning_rate": learning_rate,
                "epoch_time": epoch_time,
            }
        )

        if validation_metrics.macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_metrics.macro_f1
            epochs_without_improvement = 0
            save_training_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_macro_f1=best_validation_macro_f1,
                class_to_idx=dataloaders.class_to_idx,
                config=config,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(training_config["patience"]):
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(results_dir / "training_history.csv", index=False)
    _plot_training_curves(history_frame, results_dir / "training_curves.png")

    best_checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    validation_metrics, _, _, _ = evaluate_model(
        model=model,
        loader=dataloaders.validation,
        device=device,
        class_names=class_names,
    )
    _write_json(results_dir / "metrics_validation.json", validation_metrics)

    test_metrics, y_true, y_pred, probabilities = evaluate_model(
        model=model,
        loader=dataloaders.test,
        device=device,
        class_names=class_names,
    )
    save_evaluation_outputs(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
        dataset=dataloaders.test.dataset,
        output_dir=results_dir,
        metrics_filename="metrics_test.json",
        save_predictions=True,
    )
    canonical_test_metrics = _read_json(results_dir / "metrics_test.json")

    summary = {
        "checkpoint": str(checkpoint_path),
        "best_validation_macro_f1": best_validation_macro_f1,
        "test_macro_f1": canonical_test_metrics["macro_f1"],
        "test_accuracy": canonical_test_metrics["accuracy"],
        "metrics_source": str(results_dir / "metrics_test.json"),
        "epochs_ran": len(history),
        "device": str(device),
        "class_weights_used": class_weights_used,
        "class_distribution": dataloaders.distributions,
        "smoke_test": smoke_test,
    }
    _write_json(results_dir / "run_summary.json", summary)
    return summary


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> EpochMetrics:
    """Run one train or evaluation epoch."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        predictions = logits.argmax(dim=1)
        batch_size = int(inputs.size(0))
        total_loss += float(loss.item()) * batch_size
        correct += int((predictions == labels).sum().item())
        total += batch_size
        labels_all.extend(int(label) for label in labels.detach().cpu().tolist())
        predictions_all.extend(
            int(prediction) for prediction in predictions.detach().cpu().tolist()
        )

    macro_f1 = f1_score(labels_all, predictions_all, average="macro", zero_division=0)
    return EpochMetrics(
        loss=total_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        macro_f1=float(macro_f1),
    )


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_macro_f1: float,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
) -> None:
    """Save a state_dict checkpoint with training metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_macro_f1": best_validation_macro_f1,
            "class_to_idx": class_to_idx,
            "architecture": config["model"]["architecture"],
            "config": config,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        },
        output_path,
    )


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducible CPU and CUDA training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve a requested device while keeping CPU/CUDA compatibility."""
    if device_name:
        requested = torch.device(device_name)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_config(config: dict[str, Any], smoke_test: bool) -> dict[str, Any]:
    prepared = json.loads(json.dumps(config))
    if smoke_test:
        prepared["training"]["epochs"] = min(int(prepared["training"]["epochs"]), 1)
        prepared["training"]["frozen_epochs"] = 1
        prepared["data"]["num_workers"] = 0
        prepared["output"]["model_dir"] = "results/vision/smoke_test"
        prepared["output"]["results_dir"] = "results/vision/smoke_test"
    return prepared


def _checkpoint_path(config: dict[str, Any], smoke_test: bool) -> Path:
    if smoke_test:
        return Path(config["output"]["results_dir"]) / "smoke_test_best.pt"
    return Path(config["output"]["model_dir"]) / "resnet18_baseline_best.pt"


def _phase_for_epoch(epoch: int, frozen_epochs: int) -> str:
    return "head" if epoch <= frozen_epochs else "layer4"


def _configure_phase(model: nn.Module, epoch: int, frozen_epochs: int) -> str:
    phase = _phase_for_epoch(epoch=epoch, frozen_epochs=frozen_epochs)
    if phase == "head":
        freeze_backbone(model)
    else:
        unfreeze_layer4_and_head(model)
    return phase


def _build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_config = config["training"]
    return torch.optim.AdamW(
        get_parameter_groups(
            model=model,
            learning_rate_head=float(training_config["learning_rate_head"]),
            learning_rate_backbone=float(training_config["learning_rate_backbone"]),
        ),
        weight_decay=float(training_config["weight_decay"]),
    )


def _class_weights(
    distribution: dict[str, int],
    class_names: list[str],
    device: torch.device,
    imbalance_threshold: float = 1.5,
) -> tuple[torch.Tensor | None, bool]:
    counts = [int(distribution[class_name]) for class_name in class_names]
    positive_counts = [count for count in counts if count > 0]
    if not positive_counts:
        return None, False
    imbalance_ratio = max(positive_counts) / max(min(positive_counts), 1)
    if imbalance_ratio < imbalance_threshold:
        return None, False
    total = sum(counts)
    weights = [
        total / (len(class_names) * count) if count > 0 else 0.0
        for count in counts
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device), True


def _plot_training_curves(history: pd.DataFrame, output_path: str | Path) -> None:
    if history.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].plot(history["epoch"], history["validation_macro_f1"], label="validation")
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
