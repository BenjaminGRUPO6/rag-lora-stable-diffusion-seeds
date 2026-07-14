from __future__ import annotations

import json
import hashlib
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
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
from src.vision.preprocessing import PreprocessingConfig, fallback_crop, preprocess_image


RESULTS_1_DIR = Path("results/vision/resultados_1_baseline")
RESULTS_2_DIR = Path("results/vision/resultados_2_mejoras/05_resnet18_v2")
MODEL_COMPARISON_DIR = Path("results/vision/resultados_2_mejoras/08_comparacion_modelos")
CROP_CACHE_DIR = Path("data/cache/vision_crops")
CHECKPOINT_PATH = Path("models/vision/resnet18_v2_best.pt")
EFFICIENTNET_CHECKPOINT_PATH = Path("models/vision/efficientnet_b0_v2_best.pt")
SMOKE_RESULTS_DIR = Path("results/vision/smoke_test_resnet18_v2")
EFFICIENTNET_SMOKE_RESULTS_DIR = Path("results/vision/smoke_test_efficientnet_b0")
PRODUCTION_CONFIG_PATH = Path("configs/production_vision_model.yaml")


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

    def __init__(
        self,
        image_size: int,
        enabled: bool = True,
        compute_quality: bool = True,
    ) -> None:
        self.image_size = image_size
        self.enabled = enabled
        self.compute_quality = compute_quality
        self.config = PreprocessingConfig(output_size=image_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return the automatic crop or a controlled full-image fallback crop."""
        if not self.enabled:
            return image.convert("RGB")
        return preprocess_image(
            image,
            config=self.config,
            compute_quality=self.compute_quality,
        ).crop


class V2CropImageFolder(OrderedImageFolder):
    """ImageFolder with optional cached automatic crops before tensor transforms."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_classes: Sequence[str],
        split: str,
        image_size: int,
        transform: v2.Transform | None = None,
        auto_crop: bool = True,
        cache_preprocessing: bool = False,
        compute_quality: bool = False,
        fallback_to_original: bool = True,
        cache_dir: str | Path = CROP_CACHE_DIR,
    ) -> None:
        super().__init__(root=root, expected_classes=expected_classes, transform=transform)
        self.split = split
        self.image_size = int(image_size)
        self.auto_crop = bool(auto_crop)
        self.cache_preprocessing = bool(cache_preprocessing)
        self.compute_quality = bool(compute_quality)
        self.fallback_to_original = bool(fallback_to_original)
        self.cache_dir = Path(cache_dir)
        self.preprocessing_config = PreprocessingConfig(output_size=self.image_size)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Load one sample, using a crop cache without writing from DataLoader workers."""
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.auto_crop:
            sample = self._crop_sample(path=Path(path), sample=sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, int(target)

    def _crop_sample(self, *, path: Path, sample: Image.Image) -> Image.Image:
        if self.cache_preprocessing:
            cache_path = crop_cache_path(
                image_path=path,
                data_root=Path(self.root).parent,
                cache_dir=self.cache_dir,
                image_size=self.image_size,
                split=self.split,
            )
            if cache_path.exists():
                with Image.open(cache_path) as cached:
                    return cached.convert("RGB")
            if self.fallback_to_original:
                return fallback_crop(sample.convert("RGB"), self.preprocessing_config)
        try:
            return preprocess_image(
                sample,
                config=self.preprocessing_config,
                compute_quality=self.compute_quality,
            ).crop
        except Exception:
            if self.fallback_to_original:
                return fallback_crop(sample.convert("RGB"), self.preprocessing_config)
            raise


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
    plot_suffix = str(prepared["output"].get("plot_suffix", "resnet18_v2"))
    baseline_suffix = str(prepared["output"].get("baseline_comparison_suffix", "resnet18"))
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(results_dir / "run_config.yaml", prepared)

    if bool(prepared["preprocessing"].get("cache_preprocessing", False)):
        cache_summary = build_crop_cache(
            data_root=prepared["data"]["root"],
            classes=class_names,
            image_size=int(prepared["image_size"]),
            cache_dir=prepared["preprocessing"].get("cache_dir", str(CROP_CACHE_DIR)),
            max_samples=prepared["data"].get("max_samples"),
            compute_quality=bool(prepared["preprocessing"].get("compute_quality", False)),
            fallback_to_original=bool(
                prepared["preprocessing"].get("fallback_to_original", True)
            ),
        )
        write_json(results_dir / "crop_cache_summary.json", cache_summary)

    loaders = create_v2_dataloaders(
        data_root=prepared["data"]["root"],
        classes=class_names,
        image_size=int(prepared["image_size"]),
        batch_size=int(prepared["batch_size"]),
        num_workers=int(prepared["data"].get("num_workers", 0)),
        seed=int(prepared["seed"]),
        auto_crop=bool(prepared["preprocessing"].get("auto_crop", True)),
        cache_preprocessing=bool(prepared["preprocessing"].get("cache_preprocessing", False)),
        compute_quality=bool(prepared["preprocessing"].get("compute_quality", False)),
        fallback_to_original=bool(prepared["preprocessing"].get("fallback_to_original", True)),
        cache_dir=prepared["preprocessing"].get("cache_dir", str(CROP_CACHE_DIR)),
        max_samples=prepared["data"].get("max_samples"),
        smoke_test=smoke_test,
    )
    if bool(prepared.get("profile_dataloader", False)):
        profile = profile_loader(loaders.train, device=device, max_batches=5)
        write_json(results_dir / "dataloader_profile.json", profile)
        print(f"dataloader_profile: {profile}", flush=True)
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
            epoch=epoch,
            total_epochs=max_epochs,
            phase_name="train",
            log_every_n_batches=int(prepared.get("log_every_n_batches", 0)),
        )
        validation_metrics = run_epoch(
            model=model,
            loader=loaders.validation,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=max_epochs,
            phase_name="validation",
            log_every_n_batches=int(prepared.get("log_every_n_batches", 0)),
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

        if bool(prepared.get("checkpoint_every_epoch", False)):
            recovery_dir = checkpoint_path.parent / "recovery"
            save_checkpoint(
                path=recovery_dir / f"{checkpoint_path.stem}_epoch_{epoch:03d}.pt",
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

        print(
            (
                f"epoch={epoch}/{max_epochs} phase={current_phase} "
                f"train_loss={train_metrics.loss:.6f} "
                f"validation_macro_f1={validation_metrics.macro_f1:.6f} "
                f"epoch_time_seconds={epoch_time:.2f} best_epoch={best_epoch}"
            ),
            flush=True,
        )

        if epochs_without_improvement >= int(prepared["patience"]):
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(results_dir / "training_history.csv", index=False)
    plot_training_curves(
        history=history_frame,
        output_path=results_dir / f"r2_curvas_entrenamiento_{plot_suffix}.png",
    )

    best_checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    validation_metrics, _, _, _ = evaluate_split(
        model=model,
        loader=loaders.validation,
        device=device,
        class_names=class_names,
    )
    write_json(
        results_dir
        / str(prepared["output"].get("metrics_validation_file", "metrics_validation.json")),
        validation_metrics,
    )

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
        metrics_filename=str(prepared["output"].get("metrics_test_file", "metrics_test.json")),
        report_filename=str(
            prepared["output"].get("classification_report_file", "classification_report.csv")
        ),
        predictions_filename=str(prepared["output"].get("predictions_file", "predictions_test.csv")),
    )
    baseline_metrics = read_json(RESULTS_1_DIR / "metrics_test.json")
    save_v2_plots(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        output_dir=results_dir,
        plot_suffix=plot_suffix,
        baseline_suffix=baseline_suffix,
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
        "metrics_source": str(
            results_dir / str(prepared["output"].get("metrics_test_file", "metrics_test.json"))
        ),
    }
    write_json(results_dir / "run_summary.json", summary)
    if str(prepared["model"]["architecture"]).lower() == "efficientnet_b0":
        write_efficientnet_required_outputs(
            output_dir=results_dir,
            history_path=results_dir / "training_history.csv",
            run_config_path=results_dir / "run_config.yaml",
            run_summary=summary,
            predictions_path=results_dir
            / str(prepared["output"].get("predictions_file", "efficientnet_predictions_test.csv")),
        )
    return summary


def train_efficientnet_b0(
    config: dict[str, Any],
    *,
    device_name: str | None = None,
    smoke_test: bool = False,
    resume: bool = False,
    use_class_weights: bool | None = None,
) -> dict[str, Any]:
    """Train EfficientNet-B0 V2 and compare it with the ResNet18 V2 evidence."""
    efficientnet_config = json.loads(json.dumps(config))
    efficientnet_config.setdefault("model", {})
    efficientnet_config["model"]["architecture"] = "efficientnet_b0"
    summary = train_resnet18_v2(
        config=efficientnet_config,
        device_name=device_name,
        smoke_test=smoke_test,
        resume=resume,
        use_class_weights=use_class_weights,
    )
    if smoke_test:
        return summary
    comparison = save_model_comparison(
        efficientnet_summary=summary,
        device_name=device_name,
    )
    summary["model_selection"] = comparison
    write_json(Path(summary["results_dir"]) / "run_summary.json", summary)
    return summary


def save_model_comparison(
    *,
    efficientnet_summary: dict[str, Any],
    device_name: str | None,
) -> dict[str, Any]:
    """Save controlled ResNet18 V2 versus EfficientNet-B0 comparison artifacts."""
    output_dir = Path(efficientnet_summary["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = yaml.safe_load((output_dir / "run_config.yaml").read_text(encoding="utf-8"))
    class_names = list(run_config.get("classes", EXPECTED_CLASSES))
    image_size = int(run_config.get("image_size", 224))
    _ = resolve_device(device_name)
    latency_devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        latency_devices.append(torch.device("cuda"))

    resnet_validation = read_json(RESULTS_2_DIR / "metrics_validation.json")
    resnet_test = read_json(RESULTS_2_DIR / "metrics_test.json")
    efficientnet_validation = read_json(output_dir / "efficientnet_metrics_validation.json")
    efficientnet_test = read_json(output_dir / "efficientnet_metrics_test.json")

    parameter_rows = [
        parameter_summary(
            model_name="resnet18_v2",
            architecture="resnet18",
            checkpoint_path=CHECKPOINT_PATH,
        ),
        parameter_summary(
            model_name="efficientnet_b0_v2",
            architecture="efficientnet_b0",
            checkpoint_path=EFFICIENTNET_CHECKPOINT_PATH,
        ),
    ]
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameter_comparison.csv", index=False)

    latency_rows = []
    for latency_device in latency_devices:
        latency_rows.extend(
            [
                latency_summary(
                    model_name="resnet18_v2",
                    architecture="resnet18",
                    checkpoint_path=CHECKPOINT_PATH,
                    device=latency_device,
                    image_size=image_size,
                ),
                latency_summary(
                    model_name="efficientnet_b0_v2",
                    architecture="efficientnet_b0",
                    checkpoint_path=EFFICIENTNET_CHECKPOINT_PATH,
                    device=latency_device,
                    image_size=image_size,
                ),
            ]
        )
    pd.DataFrame(latency_rows).to_csv(output_dir / "latency_comparison.csv", index=False)
    latency_lookup = {
        (str(row["model"]), str(row["device"])): row
        for row in latency_rows
    }

    comparison_rows = [
        model_comparison_row(
            model_name="resnet18_v2",
            validation_metrics=resnet_validation,
            test_metrics=resnet_test,
            latency_rows=latency_lookup,
            parameter_row=parameter_rows[0],
            training_summary=read_json_if_exists(RESULTS_2_DIR / "run_summary.json"),
        ),
        model_comparison_row(
            model_name="efficientnet_b0_v2",
            validation_metrics=efficientnet_validation,
            test_metrics=efficientnet_test,
            latency_rows=latency_lookup,
            parameter_row=parameter_rows[1],
            training_summary=efficientnet_summary,
        ),
    ]
    pd.DataFrame(comparison_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    selection = select_model(comparison_rows)
    write_json(output_dir / "model_selection.json", selection)
    write_production_config(selection=selection, class_names=class_names, image_size=image_size)
    save_model_comparison_plots(
        output_dir=output_dir,
        class_names=class_names,
        resnet_validation=resnet_validation,
        resnet_test=resnet_test,
        efficientnet_validation=efficientnet_validation,
        efficientnet_test=efficientnet_test,
        comparison_rows=comparison_rows,
    )
    return selection


def parameter_summary(
    *,
    model_name: str,
    architecture: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Calculate parameter count and parameter-memory footprint."""
    model = create_model(
        architecture=architecture,
        num_classes=len(EXPECTED_CLASSES),
        pretrained=False,
        dropout=0.2,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "model": model_name,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "parameters": int(parameter_count),
        "trainable_parameters_initial": int(trainable_count),
        "parameter_memory_mb": float(parameter_count * 4 / (1024**2)),
    }


def latency_summary(
    *,
    model_name: str,
    architecture: str,
    checkpoint_path: Path,
    device: torch.device,
    image_size: int,
    warmup_steps: int = 5,
    measured_steps: int = 20,
) -> dict[str, Any]:
    """Measure single-image inference latency on the requested device."""
    model = create_model(
        architecture=architecture,
        num_classes=len(EXPECTED_CLASSES),
        pretrained=False,
        dropout=0.2,
    )
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    inputs = torch.randn(1, 3, image_size, image_size, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed: list[float] = []
        for _ in range(measured_steps):
            started_at = time.perf_counter()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - started_at) * 1000)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else None
    )
    return {
        "model": model_name,
        "architecture": architecture,
        "device": str(device),
        "latency_mean_ms": float(np.mean(elapsed)),
        "latency_p50_ms": float(np.median(elapsed)),
        "latency_min_ms": float(np.min(elapsed)),
        "latency_max_ms": float(np.max(elapsed)),
        "peak_inference_memory_mb": peak_memory_mb,
        "batch_size": 1,
        "image_size": int(image_size),
    }


def model_comparison_row(
    *,
    model_name: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    latency_rows: dict[tuple[str, str], dict[str, Any]],
    parameter_row: dict[str, Any],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build one row for model selection and reporting."""
    cpu_latency = latency_rows.get((model_name, "cpu"), {})
    cuda_latency = latency_rows.get((model_name, "cuda"), {})
    selected_latency = cuda_latency or cpu_latency
    return {
        "model": model_name,
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "validation_macro_precision": float(validation_metrics["macro_precision"]),
        "validation_macro_recall": float(validation_metrics["macro_recall"]),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_recall_intact": per_class_metric(
            validation_metrics,
            class_name="intact",
            metric_name="recall",
        ),
        "validation_f1_intact": per_class_metric(
            validation_metrics,
            class_name="intact",
            metric_name="f1",
        ),
        "validation_recall_broken": per_class_metric(
            validation_metrics,
            class_name="broken",
            metric_name="recall",
        ),
        "validation_f1_broken": per_class_metric(
            validation_metrics,
            class_name="broken",
            metric_name="f1",
        ),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_macro_precision": float(test_metrics["macro_precision"]),
        "test_macro_recall": float(test_metrics["macro_recall"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_recall_intact": per_class_metric(
            test_metrics,
            class_name="intact",
            metric_name="recall",
        ),
        "test_f1_intact": per_class_metric(
            test_metrics,
            class_name="intact",
            metric_name="f1",
        ),
        "test_recall_broken": per_class_metric(
            test_metrics,
            class_name="broken",
            metric_name="recall",
        ),
        "test_f1_broken": per_class_metric(
            test_metrics,
            class_name="broken",
            metric_name="f1",
        ),
        "latency_cpu_mean_ms": float(cpu_latency.get("latency_mean_ms", 0.0)),
        "latency_cuda_mean_ms": float(cuda_latency.get("latency_mean_ms", 0.0)),
        "latency_mean_ms": float(selected_latency.get("latency_mean_ms", 0.0)),
        "max_memory_mb": float(selected_latency.get("peak_inference_memory_mb") or 0.0),
        "parameter_memory_mb": float(parameter_row["parameter_memory_mb"]),
        "parameters": int(parameter_row["parameters"]),
        "checkpoint_size_mb": checkpoint_size_mb(Path(str(parameter_row["checkpoint_path"]))),
        "training_time_seconds": float(training_summary.get("elapsed_seconds", 0.0)),
    }


def per_class_metric(
    metrics: dict[str, Any],
    *,
    class_name: str,
    metric_name: str,
) -> float:
    """Read one per-class metric with explicit float conversion."""
    return float(metrics["per_class"][class_name][metric_name])


def checkpoint_size_mb(path: Path) -> float:
    """Return checkpoint size in MB when present."""
    return float(path.stat().st_size / (1024**2)) if path.exists() else 0.0


def select_model(comparison_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select a model using the configured ordered validation-first criteria."""
    selected = max(
        comparison_rows,
        key=lambda row: (
            row["validation_macro_f1"],
            row["validation_recall_intact"],
            row["validation_recall_broken"],
            -row["latency_mean_ms"],
            -row["parameter_memory_mb"],
        ),
    )
    selected_name = str(selected["model"])
    checkpoint_path = (
        EFFICIENTNET_CHECKPOINT_PATH if selected_name == "efficientnet_b0_v2" else CHECKPOINT_PATH
    )
    architecture = "efficientnet_b0" if selected_name == "efficientnet_b0_v2" else "resnet18"
    return {
        "selected_model": selected_name,
        "selected_architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "reason": (
            "Selected by validation macro-F1, then intact recall, broken recall, "
            "latency and parameter memory."
        ),
        "selection_criteria_order": [
            "validation_macro_f1",
            "validation_recall_intact",
            "validation_recall_broken",
            "latency_mean_ms",
            "parameter_memory_mb",
        ],
        "selected_metrics": selected,
        "test_evaluation_used_once": True,
        "production_replacement": selected_name == "efficientnet_b0_v2",
    }


def write_production_config(
    *,
    selection: dict[str, Any],
    class_names: Sequence[str],
    image_size: int,
) -> None:
    """Write the production vision model config after the comparison is complete."""
    if PRODUCTION_CONFIG_PATH.exists():
        backup_path = PRODUCTION_CONFIG_PATH.with_suffix(".previous.yaml")
        if not backup_path.exists():
            shutil.copyfile(PRODUCTION_CONFIG_PATH, backup_path)
    selected_metrics = selection["selected_metrics"]
    payload = {
        "model_name": selection["selected_model"],
        "architecture": selection["selected_architecture"],
        "checkpoint_path": selection["checkpoint_path"],
        "image_size": int(image_size),
        "class_names": list(class_names),
        "calibration_path": "models/vision/resnet18_v2_temperature.json"
        if selection["selected_model"] == "resnet18_v2"
        else None,
        "tta_policy": "results/vision/resultados_2_mejoras/07_tta/selected_tta_policy.json"
        if selection["selected_model"] == "resnet18_v2"
        else "none",
        "auto_crop": True,
        "selection_reason": selection["reason"],
        "validation_macro_f1": float(selected_metrics["validation_macro_f1"]),
        "test_macro_f1": float(selected_metrics["test_macro_f1"]),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "result_version": "Resultados 2",
        "selection_source": str(MODEL_COMPARISON_DIR / "model_selection.json"),
        "selection_criteria_order": selection["selection_criteria_order"],
    }
    write_yaml(PRODUCTION_CONFIG_PATH, payload)


def save_model_comparison_plots(
    *,
    output_dir: Path,
    class_names: Sequence[str],
    resnet_validation: dict[str, Any],
    resnet_test: dict[str, Any],
    efficientnet_validation: dict[str, Any],
    efficientnet_test: dict[str, Any],
    comparison_rows: Sequence[dict[str, Any]],
) -> None:
    """Save the requested model-comparison PNG artifacts."""
    plot_model_f1(
        output_path=output_dir / "r2_resnet18_vs_efficientnet_f1.png",
        resnet_validation=resnet_validation,
        resnet_test=resnet_test,
        efficientnet_validation=efficientnet_validation,
        efficientnet_test=efficientnet_test,
    )
    plot_model_class_f1(
        output_path=output_dir / "r2_resnet18_vs_efficientnet_clases.png",
        class_names=class_names,
        resnet_metrics=resnet_test,
        efficientnet_metrics=efficientnet_test,
    )
    plot_scatter_metric(
        output_path=output_dir / "r2_macro_f1_vs_latencia.png",
        rows=comparison_rows,
        x_key="latency_mean_ms",
        y_key="validation_macro_f1",
        x_label="Latencia media (ms)",
        y_label="Validation macro-F1",
        title="Macro-F1 vs latencia",
    )
    plot_scatter_metric(
        output_path=output_dir / "r2_parametros_vs_desempeno.png",
        rows=comparison_rows,
        x_key="parameters",
        y_key="validation_macro_f1",
        x_label="Parametros",
        y_label="Validation macro-F1",
        title="Parametros vs desempeno",
    )
    plot_model_confusion_matrices(
        output_path=output_dir / "r2_matrices_modelos.png",
        class_names=class_names,
        resnet_predictions=RESULTS_2_DIR / "predictions_test.csv",
        efficientnet_predictions=output_dir / "efficientnet_predictions_test.csv",
    )
    plot_results_dashboard(
        output_path=output_dir / "r1_vs_r2_modelos_dashboard.png",
        rows=comparison_rows,
    )

def build_v2_transforms(
    *,
    image_size: int,
    train: bool,
    auto_crop: bool = True,
    compute_quality: bool = True,
    resize_when_no_auto_crop: bool = True,
) -> v2.Compose:
    """Build V2 transforms with auto-crop first and deterministic eval preprocessing."""
    transforms: list[Any] = [
        AutoCropTransform(
            image_size=image_size,
            enabled=auto_crop,
            compute_quality=compute_quality,
        )
    ]
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
    elif not auto_crop and resize_when_no_auto_crop:
        resize_size = int(round(image_size * 1.14))
        transforms.extend([v2.Resize(resize_size), v2.CenterCrop(image_size)])
    transforms.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return v2.Compose(transforms)


def crop_cache_path(
    *,
    image_path: str | Path,
    data_root: str | Path,
    cache_dir: str | Path = CROP_CACHE_DIR,
    image_size: int = 224,
    split: str | None = None,
) -> Path:
    """Return the deterministic regenerated crop-cache path for one source image."""
    source = Path(image_path)
    root = Path(data_root)
    try:
        relative = source.resolve().relative_to(root.resolve())
    except ValueError:
        relative = source
    parts = relative.parts
    resolved_split = split or (parts[0] if parts else "unknown")
    class_name = parts[1] if len(parts) >= 2 else source.parent.name
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:16]
    filename = f"{source.stem}_{digest}.jpg"
    return Path(cache_dir) / str(image_size) / resolved_split / class_name / filename


def build_crop_cache(
    *,
    data_root: str | Path,
    classes: Sequence[str],
    image_size: int,
    cache_dir: str | Path = CROP_CACHE_DIR,
    splits: Sequence[str] = ("train", "validation", "test"),
    max_samples: int | None = None,
    compute_quality: bool = False,
    fallback_to_original: bool = True,
) -> dict[str, Any]:
    """Precompute automatic crops outside the training DataLoader loop."""
    root = Path(data_root)
    started_at = time.perf_counter()
    summary: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "image_size": int(image_size),
        "created": 0,
        "existing": 0,
        "fallback": 0,
        "failed": 0,
        "splits": {},
    }
    config = PreprocessingConfig(output_size=image_size)
    for split in splits:
        dataset = OrderedImageFolder(root / split, expected_classes=classes, transform=None)
        indices = list(range(len(dataset.samples)))
        if max_samples is not None:
            indices = indices[: max(0, int(max_samples))]
        split_summary = {"total": len(indices), "created": 0, "existing": 0, "fallback": 0, "failed": 0}
        for index in indices:
            image_path = Path(dataset.samples[index][0])
            output_path = crop_cache_path(
                image_path=image_path,
                data_root=root,
                cache_dir=cache_dir,
                image_size=image_size,
                split=split,
            )
            if output_path.exists():
                split_summary["existing"] += 1
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = preprocess_image(
                    image_path,
                    config=config,
                    compute_quality=compute_quality,
                )
                crop = result.crop
                if result.used_fallback:
                    split_summary["fallback"] += 1
            except Exception:
                split_summary["failed"] += 1
                if not fallback_to_original:
                    raise
                with Image.open(image_path) as opened:
                    crop = opened.convert("RGB")
                split_summary["fallback"] += 1
            crop.save(output_path, format="JPEG", quality=95)
            split_summary["created"] += 1
        summary["splits"][split] = split_summary
        for key in ("created", "existing", "fallback", "failed"):
            summary[key] = int(summary[key]) + int(split_summary[key])
    summary["elapsed_seconds"] = time.perf_counter() - started_at
    return summary


def create_v2_dataloaders(
    *,
    data_root: str | Path,
    classes: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    auto_crop: bool,
    cache_preprocessing: bool = False,
    compute_quality: bool = False,
    fallback_to_original: bool = True,
    cache_dir: str | Path = CROP_CACHE_DIR,
    max_samples: int | None = None,
    smoke_test: bool = False,
) -> VisionV2Loaders:
    """Create V2 dataloaders without changing train, validation or test splits."""
    root = Path(data_root)
    expected_mapping = {class_name: index for index, class_name in enumerate(classes)}
    datasets: dict[str, Dataset] = {
        "train": V2CropImageFolder(
            root / "train",
            expected_classes=classes,
            split="train",
            image_size=image_size,
            transform=build_v2_transforms(
                image_size=image_size,
                train=True,
                auto_crop=False,
                resize_when_no_auto_crop=not auto_crop,
            ),
            auto_crop=auto_crop,
            cache_preprocessing=cache_preprocessing,
            compute_quality=compute_quality,
            fallback_to_original=fallback_to_original,
            cache_dir=cache_dir,
        ),
        "validation": V2CropImageFolder(
            root / "validation",
            expected_classes=classes,
            split="validation",
            image_size=image_size,
            transform=build_v2_transforms(
                image_size=image_size,
                train=False,
                auto_crop=False,
                resize_when_no_auto_crop=not auto_crop,
            ),
            auto_crop=auto_crop,
            cache_preprocessing=cache_preprocessing,
            compute_quality=compute_quality,
            fallback_to_original=fallback_to_original,
            cache_dir=cache_dir,
        ),
        "test": V2CropImageFolder(
            root / "test",
            expected_classes=classes,
            split="test",
            image_size=image_size,
            transform=build_v2_transforms(
                image_size=image_size,
                train=False,
                auto_crop=False,
                resize_when_no_auto_crop=not auto_crop,
            ),
            auto_crop=auto_crop,
            cache_preprocessing=cache_preprocessing,
            compute_quality=compute_quality,
            fallback_to_original=fallback_to_original,
            cache_dir=cache_dir,
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
    if max_samples is not None:
        max_count = max(0, int(max_samples))
        datasets = {
            split_name: Subset(dataset, list(range(min(len(dataset), max_count))))
            for split_name, dataset in datasets.items()
        }
    generator = torch.Generator().manual_seed(seed)
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
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
    epoch: int = 0,
    total_epochs: int = 0,
    phase_name: str = "train",
    log_every_n_batches: int = 0,
) -> EpochMetrics:
    """Run one train or validation epoch."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    epoch_started_at = time.perf_counter()
    last_batch_finished_at = epoch_started_at
    total_batches = len(loader)
    transfer_non_blocking = device.type == "cuda"
    for batch_index, (inputs, labels) in enumerate(loader, start=1):
        data_ready_at = time.perf_counter()
        data_time = data_ready_at - last_batch_finished_at
        inputs = inputs.to(device, non_blocking=transfer_non_blocking)
        labels = labels.to(device, non_blocking=transfer_non_blocking)
        if training:
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_started_at = time.perf_counter()
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_time = time.perf_counter() - compute_started_at
        predictions = logits.argmax(dim=1)
        batch_size = int(inputs.size(0))
        total_loss += float(loss.item()) * batch_size
        correct += int((predictions == labels).sum().item())
        total += batch_size
        labels_all.extend(int(label) for label in labels.detach().cpu().tolist())
        predictions_all.extend(
            int(prediction) for prediction in predictions.detach().cpu().tolist()
        )
        if log_every_n_batches > 0 and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % log_every_n_batches == 0
        ):
            elapsed = time.perf_counter() - epoch_started_at
            images_per_second = total / max(elapsed, 1e-9)
            gpu_memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            metric_name = "train_loss" if training else "validation_loss"
            print(
                (
                    f"epoch={epoch}/{total_epochs} phase={phase_name} "
                    f"batch={batch_index}/{total_batches} "
                    f"{metric_name}={total_loss / max(total, 1):.6f} "
                    f"images_per_second={images_per_second:.2f} "
                    f"data_time={data_time:.4f}s compute_time={compute_time:.4f}s "
                    f"gpu_memory_mb={gpu_memory_mb:.1f} elapsed_time={elapsed:.1f}s"
                ),
                flush=True,
            )
        last_batch_finished_at = time.perf_counter()
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


def profile_loader(
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int = 10,
) -> dict[str, Any]:
    """Measure DataLoader fetch time without model compute."""
    started_at = time.perf_counter()
    previous_at = started_at
    batch_times: list[float] = []
    images = 0
    for batch_index, (inputs, _labels) in enumerate(loader, start=1):
        ready_at = time.perf_counter()
        batch_times.append(ready_at - previous_at)
        images += int(inputs.size(0))
        if device.type == "cuda":
            _ = inputs.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
        previous_at = time.perf_counter()
        if batch_index >= max_batches:
            break
    elapsed = time.perf_counter() - started_at
    return {
        "batches": len(batch_times),
        "images": images,
        "elapsed_seconds": elapsed,
        "mean_data_time_seconds": float(np.mean(batch_times)) if batch_times else 0.0,
        "images_per_second": images / max(elapsed, 1e-9),
        "num_workers": int(loader.num_workers),
        "pin_memory": bool(loader.pin_memory),
        "persistent_workers": bool(loader.persistent_workers),
    }


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
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
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
    metrics_filename: str = "metrics_test.json",
    report_filename: str = "classification_report.csv",
    predictions_filename: str = "predictions_test.csv",
) -> None:
    """Save the required V2 metric, report and test-prediction files."""
    write_json(output_dir / metrics_filename, metrics)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / report_filename)
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
    pd.DataFrame(rows).to_csv(output_dir / predictions_filename, index=False)


def save_v2_plots(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    output_dir: Path,
    plot_suffix: str = "resnet18_v2",
    baseline_suffix: str = "resnet18",
) -> None:
    """Save all required Resultados 2 PNG artifacts."""
    title_prefix = (
        "Resultados 2 - EfficientNet-B0"
        if "efficientnet" in plot_suffix
        else "Resultados 2"
    )
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
        output_path=output_dir / f"r2_matriz_confusion_{plot_suffix}.png",
        title=f"{title_prefix} - Matriz de confusion",
        value_format="d",
    )
    plot_confusion_matrix(
        matrix=normalized,
        class_names=class_names,
        output_path=output_dir / f"r2_matriz_confusion_normalizada_{plot_suffix}.png",
        title=f"{title_prefix} - Matriz de confusion normalizada",
        value_format=".2f",
    )
    plot_per_class_metric(
        metrics=metrics,
        class_names=class_names,
        metric_name="f1",
        output_path=output_dir / f"r2_f1_por_clase_{plot_suffix}.png",
        title=f"{title_prefix} - F1 por clase",
    )
    plot_precision_recall(
        metrics=metrics,
        class_names=class_names,
        output_path=output_dir / f"r2_precision_recall_{plot_suffix}.png",
    )
    plot_r1_vs_r2_metric(
        baseline_metrics=baseline_metrics,
        current_metrics=metrics,
        class_names=class_names,
        metric_name="f1",
        output_path=output_dir / f"r1_vs_r2_f1_{baseline_suffix}.png",
        title="F1 por clase: Resultados 1 vs Resultados 2",
    )
    plot_r1_vs_r2_metric(
        baseline_metrics=baseline_metrics,
        current_metrics=metrics,
        class_names=["intact", "broken"],
        metric_name="recall",
        output_path=output_dir / (
            "r1_vs_r2_recall_intact_broken.png"
            if baseline_suffix == "resnet18"
            else f"r1_vs_r2_recall_intact_broken_{baseline_suffix}.png"
        ),
        title="Recall intact/broken: Resultados 1 vs Resultados 2",
    )


def write_efficientnet_required_outputs(
    *,
    output_dir: Path,
    history_path: Path,
    run_config_path: Path,
    run_summary: dict[str, Any],
    predictions_path: Path,
) -> None:
    """Write the exact EfficientNet artifact names requested for Resultados 2."""
    aliases = {
        history_path: output_dir / "efficientnet_training_history.csv",
        run_config_path: output_dir / "efficientnet_run_config.yaml",
        output_dir / "r2_curvas_entrenamiento_efficientnet_b0.png": output_dir
        / "r2_efficientnet_training_curves.png",
        output_dir / "r2_f1_por_clase_efficientnet_b0.png": output_dir
        / "r2_efficientnet_f1_por_clase.png",
        output_dir / "r2_precision_recall_efficientnet_b0.png": output_dir
        / "r2_efficientnet_precision_recall.png",
        output_dir / "r2_matriz_confusion_efficientnet_b0.png": output_dir
        / "r2_efficientnet_confusion_matrix.png",
        output_dir / "r2_matriz_confusion_normalizada_efficientnet_b0.png": output_dir
        / "r2_efficientnet_confusion_matrix_normalized.png",
    }
    for source, target in aliases.items():
        if source.exists():
            shutil.copyfile(source, target)
    write_json(output_dir / "efficientnet_run_summary.json", run_summary)
    if predictions_path.exists():
        plot_confidence_distribution(
            predictions_path=predictions_path,
            output_path=output_dir / "r2_efficientnet_confidence_distribution.png",
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
    prepared["data"].setdefault("max_samples", None)
    prepared.setdefault("classes", list(EXPECTED_CLASSES))
    prepared.setdefault("preprocessing", {})
    prepared["preprocessing"].setdefault("auto_crop", True)
    prepared["preprocessing"].setdefault("cache_preprocessing", False)
    prepared["preprocessing"].setdefault("compute_quality", False)
    prepared["preprocessing"].setdefault("fallback_to_original", True)
    prepared["preprocessing"].setdefault("cache_dir", str(CROP_CACHE_DIR))
    prepared.setdefault("log_every_n_batches", 0)
    prepared.setdefault("checkpoint_every_epoch", False)
    prepared.setdefault("profile_dataloader", False)
    prepared.setdefault("layer4_epochs", 5)
    prepared.setdefault("use_class_weights", False)
    prepared.setdefault("output", {})
    architecture = str(prepared["model"]["architecture"]).lower()
    if architecture == "efficientnet_b0":
        default_results_dir = MODEL_COMPARISON_DIR
        default_checkpoint_path = EFFICIENTNET_CHECKPOINT_PATH
        default_plot_suffix = "efficientnet_b0"
        default_baseline_suffix = "efficientnet_b0"
        default_validation_file = "efficientnet_metrics_validation.json"
        default_test_file = "efficientnet_metrics_test.json"
        default_report_file = "efficientnet_classification_report.csv"
        default_predictions_file = "efficientnet_predictions_test.csv"
        smoke_results_dir = MODEL_COMPARISON_DIR / "smoke_test"
        smoke_checkpoint = smoke_results_dir / "efficientnet_b0_v2_smoke.pt"
    else:
        default_results_dir = RESULTS_2_DIR
        default_checkpoint_path = CHECKPOINT_PATH
        default_plot_suffix = "resnet18_v2"
        default_baseline_suffix = "resnet18"
        default_validation_file = "metrics_validation.json"
        default_test_file = "metrics_test.json"
        default_report_file = "classification_report.csv"
        default_predictions_file = "predictions_test.csv"
        smoke_results_dir = SMOKE_RESULTS_DIR
        smoke_checkpoint = smoke_results_dir / "resnet18_v2_smoke.pt"
    prepared["output"].setdefault("plot_suffix", default_plot_suffix)
    prepared["output"].setdefault("baseline_comparison_suffix", default_baseline_suffix)
    prepared["output"].setdefault("metrics_validation_file", default_validation_file)
    prepared["output"].setdefault("metrics_test_file", default_test_file)
    prepared["output"].setdefault("classification_report_file", default_report_file)
    prepared["output"].setdefault("predictions_file", default_predictions_file)
    if smoke_test:
        prepared["max_epochs"] = 1
        prepared["data"]["num_workers"] = 0
        prepared["output"]["results_dir"] = str(smoke_results_dir)
        prepared["output"]["checkpoint_path"] = str(smoke_checkpoint)
    else:
        prepared["output"].setdefault("results_dir", str(default_results_dir))
        prepared["output"].setdefault("checkpoint_path", str(default_checkpoint_path))
    return prepared


def plot_training_curves(*, history: pd.DataFrame, output_path: Path) -> None:
    """Plot train/validation loss, accuracy and macro-F1."""
    if history.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if "efficientnet" in output_path.name:
        figure.suptitle("Resultados 2 - EfficientNet-B0")
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
    figure.savefig(output_path, dpi=180, facecolor="white")
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
    figure.savefig(output_path, dpi=180, facecolor="white")
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
    title = (
        "Resultados 2 - EfficientNet-B0 - Precision y recall por clase"
        if "efficientnet" in output_path.name
        else "Resultados 2 - Precision y recall por clase"
    )
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
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
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_model_f1(
    *,
    output_path: Path,
    resnet_validation: dict[str, Any],
    resnet_test: dict[str, Any],
    efficientnet_validation: dict[str, Any],
    efficientnet_test: dict[str, Any],
) -> None:
    """Plot validation and test macro-F1 for the two controlled models."""
    labels = ["validation", "test"]
    resnet_values = [float(resnet_validation["macro_f1"]), float(resnet_test["macro_f1"])]
    efficientnet_values = [
        float(efficientnet_validation["macro_f1"]),
        float(efficientnet_test["macro_f1"]),
    ]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(x - width / 2, resnet_values, width, label="ResNet18 V2", color="#2563eb")
    axis.bar(
        x + width / 2,
        efficientnet_values,
        width,
        label="EfficientNet-B0",
        color="#16a34a",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro-F1")
    axis.set_title("ResNet18 V2 vs EfficientNet-B0")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_model_class_f1(
    *,
    output_path: Path,
    class_names: Sequence[str],
    resnet_metrics: dict[str, Any],
    efficientnet_metrics: dict[str, Any],
) -> None:
    """Plot test F1 per class for ResNet18 V2 and EfficientNet-B0."""
    resnet_values = [
        float(resnet_metrics["per_class"][class_name]["f1"]) for class_name in class_names
    ]
    efficientnet_values = [
        float(efficientnet_metrics["per_class"][class_name]["f1"]) for class_name in class_names
    ]
    x = np.arange(len(class_names))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - width / 2, resnet_values, width, label="ResNet18 V2", color="#2563eb")
    axis.bar(
        x + width / 2,
        efficientnet_values,
        width,
        label="EfficientNet-B0",
        color="#16a34a",
    )
    axis.set_xticks(x, class_names, rotation=30)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test F1")
    axis.set_title("F1 por clase")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_scatter_metric(
    *,
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Plot a two-model scatter chart with labels."""
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    colors = {"resnet18_v2": "#2563eb", "efficientnet_b0_v2": "#16a34a"}
    for row in rows:
        model_name = str(row["model"])
        axis.scatter(
            float(row[x_key]),
            float(row[y_key]),
            s=90,
            label=model_name,
            color=colors.get(model_name, "#475569"),
        )
        axis.annotate(
            model_name,
            (float(row[x_key]), float(row[y_key])),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_model_confusion_matrices(
    *,
    output_path: Path,
    class_names: Sequence[str],
    resnet_predictions: Path,
    efficientnet_predictions: Path,
) -> None:
    """Plot normalized test confusion matrices for both models side by side."""
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    matrices = [
        (
            "ResNet18 V2",
            confusion_from_predictions(
                predictions_path=resnet_predictions,
                class_to_idx=class_to_idx,
            ),
        ),
        (
            "EfficientNet-B0",
            confusion_from_predictions(
                predictions_path=efficientnet_predictions,
                class_to_idx=class_to_idx,
            ),
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for axis, (title, matrix) in zip(axes, matrices, strict=True):
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
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
        for row_index in range(len(class_names)):
            for column_index in range(len(class_names)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    format(value, ".2f"),
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                )
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_results_dashboard(*, output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Plot a compact Resultados 1/2 dashboard for model-selection evidence."""
    labels = [str(row["model"]) for row in rows]
    validation = [float(row["validation_macro_f1"]) for row in rows]
    test = [float(row["test_macro_f1"]) for row in rows]
    latency = [float(row["latency_mean_ms"]) for row in rows]
    params = [float(row["parameters"]) / 1_000_000.0 for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white")
    figure.suptitle("Resultados 2 - Dashboard comparacion de modelos")
    panels = [
        (axes[0, 0], validation, "Validation macro-F1", (0, 1)),
        (axes[0, 1], test, "Test macro-F1", (0, 1)),
        (axes[1, 0], latency, "Latencia CUDA/seleccionada (ms)", None),
        (axes[1, 1], params, "Parametros (millones)", None),
    ]
    for axis, values, title, ylim in panels:
        axis.set_facecolor("white")
        bars = axis.bar(labels, values, color=["#2563eb", "#16a34a"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        if ylim is not None:
            axis.set_ylim(*ylim)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
    axes[0, 0].text(
        0.01,
        0.05,
        "Seleccion por validation macro-F1; test solo para reporte final.",
        transform=axes[0, 0].transAxes,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def confusion_from_predictions(
    *,
    predictions_path: Path,
    class_to_idx: dict[str, int],
) -> np.ndarray:
    """Create a normalized confusion matrix from saved prediction CSV files."""
    frame = pd.read_csv(predictions_path)
    y_true = [class_to_idx[str(label)] for label in frame["true_label"]]
    y_pred = [class_to_idx[str(label)] for label in frame["predicted_label"]]
    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_to_idx))),
        normalize="true",
    )


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
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_confidence_distribution(*, predictions_path: Path, output_path: Path) -> None:
    """Plot predicted-confidence distribution for EfficientNet test predictions."""
    frame = pd.read_csv(predictions_path)
    values = frame["predicted_probability"].astype(float) if "predicted_probability" in frame else []
    figure, axis = plt.subplots(figsize=(7, 4.5), facecolor="white")
    axis.set_facecolor("white")
    axis.hist(values, bins=12, range=(0, 1), color="#2563eb", edgecolor="white")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Confianza predicha")
    axis.set_ylabel("Imagenes")
    axis.set_title("Resultados 2 - EfficientNet-B0 - Distribucion de confianza")
    axis.text(
        0.02,
        0.96,
        f"Resultados 2 | n={len(frame)} imagenes | test",
        transform=axis.transAxes,
        va="top",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
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


def read_json_if_exists(path: Path) -> dict[str, Any]:
    """Read a JSON object or return an empty object when optional evidence is absent."""
    return read_json(path) if path.exists() else {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML config used for the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
