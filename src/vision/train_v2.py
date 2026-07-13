from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2

from src.vision.dataset import (
    EXPECTED_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OrderedImageFolder,
    class_distribution,
    subset_per_class,
)
from src.vision.evaluation import image_paths, load_checkpoint
from src.vision.model import (
    create_model,
    freeze_backbone,
    get_parameter_groups,
    unfreeze_layer3_layer4_and_head,
    unfreeze_layer4_and_head,
)
from src.vision.preprocessing import PreprocessingConfig, preprocess_image


RESULTS_1_DIR = Path("results/vision/resultados_1_baseline")
RESULTS_2_DIR = Path("results/vision/resultados_2_mejoras/05_resnet18_v2")
CHECKPOINT_PATH = Path("models/vision/resnet18_v2_best.pt")
SMOKE_RESULTS_DIR = Path("results/vision/smoke_test_resnet18_v2")


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected for one epoch and split."""

    loss: float
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class VisionV2Loaders:
    """Dataloaders and class metadata for the V2 experiment."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    class_to_idx: dict[str, int]
    distributions: dict[str, dict[str, int]]


class AutoCropTransform:
    """Apply deterministic automatic seed cropping before augmentation."""

    def __init__(self, image_size: int, enabled: bool = True) -> None:
        self.image_size = image_size
        self.enabled = enabled
        self.config = PreprocessingConfig(output_size=image_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return the automatic crop or a controlled full-image fallback crop."""
        if not self.enabled:
            return image.convert("RGB")
        return preprocess_image(image, config=self.config).crop


def train_resnet18_v2(
    config: dict[str, Any],
    *,
    device_name: str | None = None,
    smoke_test: bool = False,
    resume: bool = False,
    use_class_weights: bool | None = None,
) -> dict[str, Any]:
    """Train ResNet18 V2, select by validation macro-F1 and evaluate test once."""
    prepared = prepare_v2_config(config=config, smoke_test=smoke_test)
    if use_class_weights is not None:
        prepared["use_class_weights"] = bool(use_class_weights)
    set_seed(int(prepared["seed"]))
    device = resolve_device(device_name)
    class_names = list(prepared.get("classes", EXPECTED_CLASSES))
    results_dir = Path(prepared["output"]["results_dir"])
    checkpoint_path = Path(prepared["output"]["checkpoint_path"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(results_dir / "run_config.yaml", prepared)

    loaders = create_v2_dataloaders(
        data_root=prepared["data"]["root"],
        classes=class_names,
        image_size=int(prepared["image_size"]),
        batch_size=int(prepared["batch_size"]),
        num_workers=int(prepared["data"].get("num_workers", 0)),
        seed=int(prepared["seed"]),
        auto_crop=bool(prepared["preprocessing"].get("auto_crop", True)),
        smoke_test=smoke_test,
    )
    model = create_model(
        architecture=str(prepared["model"]["architecture"]),
        num_classes=int(prepared["model"]["num_classes"]),
        pretrained=bool(prepared["model"]["pretrained"]),
        dropout=float(prepared["model"]["dropout"]),
    ).to(device)

    class_weights = build_class_weights(
        distribution=loaders.distributions["train"],
        class_names=class_names,
        device=device,
        enabled=bool(prepared.get("use_class_weights", False)),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    max_epochs = int(prepared["max_epochs"])
    best_validation_macro_f1 = -1.0
    best_epoch = 0
    start_epoch = 1
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0

    current_phase = configure_phase(model=model, epoch=start_epoch, config=prepared)
    optimizer = build_optimizer(model=model, config=prepared)
    scheduler = build_scheduler(optimizer=optimizer, config=prepared, max_epochs=max_epochs)
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
        best_validation_macro_f1 = float(checkpoint["best_validation_macro_f1"])
        history = list(checkpoint.get("history", []))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        current_phase = configure_phase(model=model, epoch=start_epoch, config=prepared)
        optimizer = build_optimizer(model=model, config=prepared)
        scheduler = build_scheduler(optimizer=optimizer, config=prepared, max_epochs=max_epochs)
        if checkpoint.get("phase") == current_phase:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (KeyError, ValueError):
                optimizer = build_optimizer(model=model, config=prepared)
                scheduler = build_scheduler(
                    optimizer=optimizer,
                    config=prepared,
                    max_epochs=max_epochs,
                )

    use_amp = bool(prepared["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    started_at = time.perf_counter()

    for epoch in range(start_epoch, max_epochs + 1):
        desired_phase = phase_for_epoch(epoch=epoch, config=prepared)
        if desired_phase != current_phase:
            current_phase = configure_phase(model=model, epoch=epoch, config=prepared)
            optimizer = build_optimizer(model=model, config=prepared)
            scheduler = build_scheduler(optimizer=optimizer, config=prepared, max_epochs=max_epochs)

        epoch_started_at = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=loaders.train,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=loaders.validation,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
        )
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_started_at
        learning_rate = max(float(group["lr"]) for group in optimizer.param_groups)
        improved = validation_metrics.macro_f1 > best_validation_macro_f1
        history.append(
            {
                "epoch": epoch,
                "phase": current_phase,
                "train_loss": train_metrics.loss,
                "validation_loss": validation_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "validation_accuracy": validation_metrics.accuracy,
                "train_macro_f1": train_metrics.macro_f1,
                "validation_macro_f1": validation_metrics.macro_f1,
                "learning_rate": learning_rate,
                "epoch_time_seconds": epoch_time,
                "is_best": improved,
            }
        )

        if improved:
            best_validation_macro_f1 = validation_metrics.macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_macro_f1=best_validation_macro_f1,
                class_to_idx=loaders.class_to_idx,
                config=prepared,
                phase=current_phase,
                history=history,
                epochs_without_improvement=epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(prepared["patience"]):
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(results_dir / "training_history.csv", index=False)
    plot_training_curves(
        history=history_frame,
        output_path=results_dir / "r2_curvas_entrenamiento_resnet18_v2.png",
    )

    best_checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    validation_metrics, _, _, _ = evaluate_split(
        model=model,
        loader=loaders.validation,
        device=device,
        class_names=class_names,
    )
    write_json(results_dir / "metrics_validation.json", validation_metrics)

    test_metrics, y_true, y_pred, probabilities = evaluate_split(
        model=model,
        loader=loaders.test,
        device=device,
        class_names=class_names,
    )
    save_test_outputs(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
        dataset=loaders.test.dataset,
        output_dir=results_dir,
        metrics=test_metrics,
    )
    baseline_metrics = read_json(RESULTS_1_DIR / "metrics_test.json")
    save_v2_plots(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        output_dir=results_dir,
    )

    elapsed_seconds = time.perf_counter() - started_at
    validation_imbalance = imbalance_summary(loaders.distributions["validation"])
    train_imbalance = imbalance_summary(loaders.distributions["train"])
    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(best_checkpoint.get("best_epoch", best_epoch)),
        "best_validation_macro_f1": float(best_checkpoint["best_validation_macro_f1"]),
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "device": str(device),
        "epochs_ran": len(history_frame),
        "elapsed_seconds": elapsed_seconds,
        "class_weights_used": class_weights is not None,
        "class_distribution": loaders.distributions,
        "train_imbalance": train_imbalance,
        "validation_imbalance": validation_imbalance,
        "smoke_test": smoke_test,
        "results_dir": str(results_dir),
        "metrics_source": str(results_dir / "metrics_test.json"),
    }
    write_json(results_dir / "run_summary.json", summary)
    return summary


def build_v2_transforms(
    *,
    image_size: int,
    train: bool,
    auto_crop: bool = True,
) -> v2.Compose:
    """Build V2 transforms with auto-crop first and deterministic eval preprocessing."""
    transforms: list[Any] = [AutoCropTransform(image_size=image_size, enabled=auto_crop)]
    if train:
        transforms.extend(
            [
                v2.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=10),
                v2.ColorJitter(brightness=0.08, contrast=0.08),
                v2.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.12,
                ),
                v2.RandomPerspective(distortion_scale=0.08, p=0.12),
            ]
        )
    transforms.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return v2.Compose(transforms)


def create_v2_dataloaders(
    *,
    data_root: str | Path,
    classes: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    auto_crop: bool,
    smoke_test: bool = False,
) -> VisionV2Loaders:
    """Create V2 dataloaders without changing train, validation or test splits."""
    root = Path(data_root)
    expected_mapping = {class_name: index for index, class_name in enumerate(classes)}
    datasets: dict[str, Dataset] = {
        "train": OrderedImageFolder(
            root / "train",
            expected_classes=classes,
            transform=build_v2_transforms(
                image_size=image_size,
                train=True,
                auto_crop=auto_crop,
            ),
        ),
        "validation": OrderedImageFolder(
            root / "validation",
            expected_classes=classes,
            transform=build_v2_transforms(
                image_size=image_size,
                train=False,
                auto_crop=auto_crop,
            ),
        ),
        "test": OrderedImageFolder(
            root / "test",
            expected_classes=classes,
            transform=build_v2_transforms(
                image_size=image_size,
                train=False,
                auto_crop=auto_crop,
            ),
        ),
    }
    for split_name, dataset in datasets.items():
        class_to_idx = getattr(dataset, "class_to_idx", {})
        if class_to_idx != expected_mapping:
            raise ValueError(
                f"{split_name} class_to_idx mismatch. Expected {expected_mapping}, "
                f"got {class_to_idx}."
            )
    if smoke_test:
        datasets = {
            split_name: subset_per_class(dataset, max_per_class=1)
            for split_name, dataset in datasets.items()
        }
    generator = torch.Generator().manual_seed(seed)
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return VisionV2Loaders(
        train=DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            drop_last=False,
            **common_kwargs,
        ),
        validation=DataLoader(datasets["validation"], shuffle=False, **common_kwargs),
        test=DataLoader(datasets["test"], shuffle=False, **common_kwargs),
        class_to_idx=expected_mapping,
        distributions={
            "train": class_distribution(datasets["train"], expected_mapping),
            "validation": class_distribution(datasets["validation"], expected_mapping),
            "test": class_distribution(datasets["test"], expected_mapping),
        },
    )


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> EpochMetrics:
    """Run one train or validation epoch."""
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
    macro_f1 = precision_recall_fscore_support(
        labels_all,
        predictions_all,
        average="macro",
        zero_division=0,
    )[2]
    return EpochMetrics(
        loss=total_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        macro_f1=float(macro_f1),
    )


def evaluate_split(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], list[int], list[int], list[list[float]]]:
    """Evaluate one split and return metrics plus raw predictions."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            batch_probabilities = torch.softmax(logits, dim=1).detach().cpu()
            predictions = batch_probabilities.argmax(dim=1)
            y_true.extend(int(label) for label in labels.cpu().tolist())
            y_pred.extend(int(prediction) for prediction in predictions.tolist())
            probabilities.extend(
                [float(value) for value in row] for row in batch_probabilities.tolist()
            )
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    return metrics, y_true, y_pred, probabilities


def compute_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute aggregate and per-class metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def save_test_outputs(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
    dataset: Dataset,
    output_dir: Path,
    metrics: dict[str, Any],
) -> None:
    """Save the required V2 metric, report and test-prediction files."""
    write_json(output_dir / "metrics_test.json", metrics)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    paths = image_paths(dataset)
    rows: list[dict[str, Any]] = []
    for index, (true_index, pred_index, row_probabilities) in enumerate(
        zip(y_true, y_pred, probabilities, strict=True)
    ):
        row: dict[str, Any] = {
            "image_path": paths[index] if index < len(paths) else "",
            "true_label": class_names[int(true_index)],
            "predicted_label": class_names[int(pred_index)],
            "predicted_probability": float(row_probabilities[int(pred_index)]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"probability_{class_name}"] = float(row_probabilities[class_index])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "predictions_test.csv", index=False)


def save_v2_plots(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save all required Resultados 2 PNG artifacts."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize="true",
    )
    plot_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        output_path=output_dir / "r2_matriz_confusion_resnet18_v2.png",
        title="Resultados 2 - Matriz de confusion",
        value_format="d",
    )
    plot_confusion_matrix(
        matrix=normalized,
        class_names=class_names,
        output_path=output_dir / "r2_matriz_confusion_normalizada_resnet18_v2.png",
        title="Resultados 2 - Matriz de confusion normalizada",
        value_format=".2f",
    )
    plot_per_class_metric(
        metrics=metrics,
        class_names=class_names,
        metric_name="f1",
        output_path=output_dir / "r2_f1_por_clase_resnet18_v2.png",
        title="Resultados 2 - F1 por clase",
    )
    plot_precision_recall(
        metrics=metrics,
        class_names=class_names,
        output_path=output_dir / "r2_precision_recall_resnet18_v2.png",
    )
    plot_r1_vs_r2_metric(
        baseline_metrics=baseline_metrics,
        current_metrics=metrics,
        class_names=class_names,
        metric_name="f1",
        output_path=output_dir / "r1_vs_r2_f1_resnet18.png",
        title="F1 por clase: Resultados 1 vs Resultados 2",
    )
    plot_r1_vs_r2_metric(
        baseline_metrics=baseline_metrics,
        current_metrics=metrics,
        class_names=["intact", "broken"],
        metric_name="recall",
        output_path=output_dir / "r1_vs_r2_recall_intact_broken.png",
        title="Recall intact/broken: Resultados 1 vs Resultados 2",
    )


def configure_phase(model: nn.Module, *, epoch: int, config: dict[str, Any]) -> str:
    """Freeze or unfreeze trainable layers for the active progressive phase."""
    phase = phase_for_epoch(epoch=epoch, config=config)
    if phase == "head":
        freeze_backbone(model)
    elif phase == "layer4":
        unfreeze_layer4_and_head(model)
    else:
        unfreeze_layer3_layer4_and_head(model)
    return phase


def phase_for_epoch(epoch: int, config: dict[str, Any]) -> str:
    """Return head, layer4 or layer3_layer4 for the configured epoch."""
    head_epochs = int(config["frozen_head_epochs"])
    layer4_epochs = int(config.get("layer4_epochs", 5))
    if epoch <= head_epochs:
        return "head"
    if epoch <= head_epochs + layer4_epochs:
        return "layer4"
    return "layer3_layer4"


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the configured optimizer."""
    optimizer_name = str(config["optimizer"]).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer for ResNet18 V2: {config['optimizer']}")
    return torch.optim.AdamW(
        get_parameter_groups(
            model=model,
            learning_rate_head=float(config["learning_rate_head"]),
            learning_rate_backbone=float(config["learning_rate_backbone"]),
        ),
        weight_decay=float(config["weight_decay"]),
    )


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    max_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the configured learning-rate scheduler."""
    scheduler_name = str(config["scheduler"]).lower()
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler for ResNet18 V2: {config['scheduler']}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(max_epochs, 1),
    )


def build_class_weights(
    *,
    distribution: dict[str, int],
    class_names: Sequence[str],
    device: torch.device,
    enabled: bool,
) -> torch.Tensor | None:
    """Optionally calculate class weights using only the train distribution."""
    if not enabled:
        return None
    counts = [int(distribution[class_name]) for class_name in class_names]
    total = sum(counts)
    if total <= 0:
        return None
    weights = [
        total / (len(class_names) * count) if count > 0 else 0.0
        for count in counts
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_epoch: int,
    best_validation_macro_f1: float,
    class_to_idx: dict[str, int],
    config: dict[str, Any],
    phase: str,
    history: list[dict[str, Any]],
    epochs_without_improvement: int,
) -> None:
    """Save the best validation checkpoint outside Git-tracked source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_validation_macro_f1,
            "class_to_idx": class_to_idx,
            "architecture": config["model"]["architecture"],
            "config": config,
            "phase": phase,
            "history": history,
            "epochs_without_improvement": epochs_without_improvement,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        },
        path,
    )


def prepare_v2_config(config: dict[str, Any], *, smoke_test: bool) -> dict[str, Any]:
    """Normalize the V2 YAML config and apply smoke-test overrides."""
    prepared = json.loads(json.dumps(config))
    prepared.setdefault("model", {})
    prepared["model"].setdefault("architecture", "resnet18")
    prepared["model"].setdefault("pretrained", True)
    prepared["model"].setdefault("num_classes", 5)
    prepared["model"].setdefault("dropout", 0.2)
    prepared.setdefault("data", {})
    prepared["data"].setdefault("root", "data/processed")
    prepared["data"].setdefault("num_workers", 0)
    prepared.setdefault("classes", list(EXPECTED_CLASSES))
    prepared.setdefault("preprocessing", {})
    prepared["preprocessing"].setdefault("auto_crop", True)
    prepared.setdefault("layer4_epochs", 5)
    prepared.setdefault("use_class_weights", False)
    prepared.setdefault("output", {})
    if smoke_test:
        prepared["max_epochs"] = 1
        prepared["batch_size"] = min(int(prepared["batch_size"]), 2)
        prepared["data"]["num_workers"] = 0
        prepared["model"]["pretrained"] = False
        prepared["output"]["results_dir"] = str(SMOKE_RESULTS_DIR)
        prepared["output"]["checkpoint_path"] = str(SMOKE_RESULTS_DIR / "resnet18_v2_smoke.pt")
    else:
        prepared["output"].setdefault("results_dir", str(RESULTS_2_DIR))
        prepared["output"].setdefault("checkpoint_path", str(CHECKPOINT_PATH))
    return prepared


def plot_training_curves(*, history: pd.DataFrame, output_path: Path) -> None:
    """Plot train/validation loss, accuracy and macro-F1."""
    if history.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    axes[2].plot(history["epoch"], history["train_macro_f1"], label="train")
    axes[2].plot(history["epoch"], history["validation_macro_f1"], label="validation")
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("Epoca")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_per_class_metric(
    *,
    metrics: dict[str, Any],
    class_names: Sequence[str],
    metric_name: str,
    output_path: Path,
    title: str,
) -> None:
    """Plot one per-class metric."""
    values = [float(metrics["per_class"][class_name][metric_name]) for class_name in class_names]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(class_names, values, color="#2f6f9f")
    axis.set_ylim(0, 1)
    axis.set_ylabel(metric_name.upper())
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_precision_recall(
    *,
    metrics: dict[str, Any],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """Plot precision and recall per class."""
    precision = [float(metrics["per_class"][class_name]["precision"]) for class_name in class_names]
    recall = [float(metrics["per_class"][class_name]["recall"]) for class_name in class_names]
    x = np.arange(len(class_names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(x - width / 2, precision, width, label="precision", color="#3f7d20")
    axis.bar(x + width / 2, recall, width, label="recall", color="#b5651d")
    axis.set_xticks(x, class_names, rotation=30)
    axis.set_ylim(0, 1)
    axis.set_title("Resultados 2 - Precision y recall por clase")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_r1_vs_r2_metric(
    *,
    baseline_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    class_names: Sequence[str],
    metric_name: str,
    output_path: Path,
    title: str,
) -> None:
    """Plot Resultados 1, Resultados 2 and absolute difference."""
    r1_values = [
        float(baseline_metrics["per_class"][class_name][metric_name])
        for class_name in class_names
    ]
    r2_values = [
        float(current_metrics["per_class"][class_name][metric_name])
        for class_name in class_names
    ]
    diff_values = [abs(r2 - r1) for r1, r2 in zip(r1_values, r2_values, strict=True)]
    x = np.arange(len(class_names))
    width = 0.26
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - width, r1_values, width, label="Resultados 1", color="#667085")
    axis.bar(x, r2_values, width, label="Resultados 2", color="#2563eb")
    axis.bar(x + width, diff_values, width, label="Diferencia absoluta", color="#d97706")
    axis.set_xticks(x, class_names, rotation=30)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(
    *,
    matrix: Any,
    class_names: Sequence[str],
    output_path: Path,
    title: str,
    value_format: str,
) -> None:
    """Plot a confusion matrix."""
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Etiqueta real",
        xlabel="Prediccion",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if getattr(matrix, "size", 0) else 0
    for row_index in range(len(class_names)):
        for column_index in range(len(class_names)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def imbalance_summary(distribution: dict[str, int]) -> dict[str, Any]:
    """Summarize class imbalance for a split distribution."""
    counts = [int(value) for value in distribution.values() if int(value) > 0]
    if not counts:
        return {"ratio": 0.0, "relevant": False}
    ratio = max(counts) / max(min(counts), 1)
    return {"ratio": float(ratio), "relevant": bool(ratio >= 1.5)}


def set_seed(seed: int) -> None:
    """Set CPU and CUDA seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve the requested device."""
    if device_name:
        requested = torch.device(device_name)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML config used for the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
