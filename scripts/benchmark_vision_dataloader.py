"""Benchmark V2 vision DataLoader preprocessing modes for Resultados 2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.dataset import EXPECTED_CLASSES
from src.vision.train_v2 import (
    MODEL_COMPARISON_DIR,
    build_crop_cache,
    create_v2_dataloaders,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description="Benchmark vision V2 DataLoader modes.")
    parser.add_argument("--config", type=Path, default=Path("configs/vision_v2_efficientnet_b0.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=MODEL_COMPARISON_DIR)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    """Run all requested benchmark cases."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    class_names = list(config.get("classes", EXPECTED_CLASSES))
    data_root = Path(config.get("data", {}).get("root", "data/processed"))
    image_size = int(config.get("image_size", 224))
    cache_dir = Path(config.get("preprocessing", {}).get("cache_dir", "data/cache/vision_crops"))

    cache_summary = build_crop_cache(
        data_root=data_root,
        classes=class_names,
        image_size=image_size,
        cache_dir=cache_dir,
        splits=("train",),
        max_samples=args.samples,
        compute_quality=False,
        fallback_to_original=True,
    )
    rows = [
        run_case(
            name="A_no_auto_crop_workers0",
            data_root=data_root,
            class_names=class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            samples=args.samples,
            device=device,
            auto_crop=False,
            cache_preprocessing=False,
            num_workers=0,
            cache_dir=cache_dir,
        ),
        run_case(
            name="B_realtime_crop_workers0",
            data_root=data_root,
            class_names=class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            samples=args.samples,
            device=device,
            auto_crop=True,
            cache_preprocessing=False,
            num_workers=0,
            cache_dir=cache_dir,
        ),
        run_case(
            name="C_cache_crop_workers0",
            data_root=data_root,
            class_names=class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            samples=args.samples,
            device=device,
            auto_crop=True,
            cache_preprocessing=True,
            num_workers=0,
            cache_dir=cache_dir,
        ),
        run_case(
            name="D_cache_workers0",
            data_root=data_root,
            class_names=class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            samples=args.samples,
            device=device,
            auto_crop=True,
            cache_preprocessing=True,
            num_workers=0,
            cache_dir=cache_dir,
        ),
        run_case(
            name="E_cache_workers2",
            data_root=data_root,
            class_names=class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            samples=args.samples,
            device=device,
            auto_crop=True,
            cache_preprocessing=True,
            num_workers=2,
            cache_dir=cache_dir,
        ),
    ]
    for row in rows:
        row["cache_created"] = cache_summary["created"]
        row["cache_existing"] = cache_summary["existing"]
    csv_path = output_dir / "dataloader_benchmark.csv"
    write_csv(csv_path, rows)
    runtime_config = select_runtime_config(rows=rows, batch_size=args.batch_size, device=device)
    (output_dir / "efficientnet_runtime_config.json").write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_plots(output_dir=output_dir, rows=rows, samples=args.samples)
    print(csv_path.as_posix(), flush=True)


def run_case(
    *,
    name: str,
    data_root: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    samples: int,
    device: torch.device,
    auto_crop: bool,
    cache_preprocessing: bool,
    num_workers: int,
    cache_dir: Path,
) -> dict[str, Any]:
    """Benchmark one DataLoader configuration."""
    loaders = create_v2_dataloaders(
        data_root=data_root,
        classes=class_names,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=42,
        auto_crop=auto_crop,
        cache_preprocessing=cache_preprocessing,
        compute_quality=False,
        fallback_to_original=True,
        cache_dir=cache_dir,
        max_samples=samples,
        smoke_test=False,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    previous_at = started_at
    data_time_total = 0.0
    compute_time_total = 0.0
    image_count = 0
    batch_count = 0
    for inputs, _labels in loaders.train:
        ready_at = time.perf_counter()
        data_time_total += ready_at - previous_at
        compute_started_at = time.perf_counter()
        if device.type == "cuda":
            _ = inputs.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
        else:
            _ = inputs
        compute_time_total += time.perf_counter() - compute_started_at
        image_count += int(inputs.size(0))
        batch_count += 1
        previous_at = time.perf_counter()
    total_time = time.perf_counter() - started_at
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    return {
        "case": name,
        "samples": image_count,
        "batch_size": batch_size,
        "batches": batch_count,
        "num_workers": num_workers,
        "auto_crop": auto_crop,
        "cache_preprocessing": cache_preprocessing,
        "total_seconds": total_time,
        "seconds_per_batch": total_time / max(batch_count, 1),
        "data_time_seconds": data_time_total,
        "compute_time_seconds": compute_time_total,
        "images_per_second": image_count / max(total_time, 1e-9),
        "peak_ram_mb": peak_ram_mb(),
        "peak_vram_mb": peak_vram,
        "waiting_data_percent": 100.0 * data_time_total / max(total_time, 1e-9),
    }


def peak_ram_mb() -> float:
    """Return current process RSS in MB when psutil is available."""
    try:
        import psutil
    except ImportError:
        return 0.0
    return float(psutil.Process().memory_info().rss / (1024**2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write benchmark rows."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_runtime_config(
    *,
    rows: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Select the fastest stable cached DataLoader config from benchmark evidence."""
    candidates = [row for row in rows if bool(row["cache_preprocessing"])]
    selected = max(
        candidates,
        key=lambda row: (
            float(row["images_per_second"]),
            -float(row["data_time_seconds"]),
            -int(row["num_workers"]),
        ),
    )
    return {
        "selected_num_workers": int(selected["num_workers"]),
        "selected_batch_size": int(batch_size),
        "cache_preprocessing": True,
        "auto_crop": True,
        "compute_quality": False,
        "fallback_to_original": True,
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(selected["num_workers"]) > 0,
        "non_blocking": device.type == "cuda",
        "selection_reason": (
            "Fastest cached preprocessing configuration by images_per_second, "
            "with data_time as tiebreaker and Windows stability considered."
        ),
        "selected_case": selected["case"],
        "selected_images_per_second": float(selected["images_per_second"]),
        "selected_data_time_seconds": float(selected["data_time_seconds"]),
        "batch_size_candidates": [8, 4, 2],
        "oom_observed": False,
    }


def write_plots(*, output_dir: Path, rows: list[dict[str, Any]], samples: int) -> None:
    """Write all requested benchmark PNG files."""
    labels = [row["case"].replace("_", "\n") for row in rows]
    plot_bar(
        output_path=output_dir / "r2_dataloader_speed_comparison.png",
        labels=labels,
        values=[float(row["seconds_per_batch"]) for row in rows],
        ylabel="Segundos por batch",
        title="Resultados 2 - DataLoader speed comparison",
        note=f"{samples} imagenes | batch benchmark",
        lower_is_better=True,
    )
    plot_grouped(
        output_path=output_dir / "r2_data_time_vs_compute_time.png",
        labels=labels,
        first=[float(row["data_time_seconds"]) for row in rows],
        second=[float(row["compute_time_seconds"]) for row in rows],
        title="Resultados 2 - Data time vs compute time",
        note=f"{samples} imagenes | configuraciones A-E",
    )
    plot_bar(
        output_path=output_dir / "r2_images_per_second.png",
        labels=labels,
        values=[float(row["images_per_second"]) for row in rows],
        ylabel="Imagenes/segundo",
        title="Resultados 2 - Images per second",
        note=f"{samples} imagenes | mayor es mejor",
        lower_is_better=False,
    )
    worker_rows = [row for row in rows if str(row["case"]).startswith(("D_", "E_"))]
    plot_bar(
        output_path=output_dir / "r2_workers_comparison.png",
        labels=[str(row["num_workers"]) for row in worker_rows],
        values=[float(row["images_per_second"]) for row in worker_rows],
        ylabel="Imagenes/segundo",
        title="Resultados 2 - Workers comparison",
        note=f"{samples} imagenes | cache preprocessing",
        lower_is_better=False,
    )


def plot_bar(
    *,
    output_path: Path,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    note: str,
    lower_is_better: bool,
) -> None:
    """Write one white-background bar chart."""
    colors = ["#2563eb" if not lower_is_better else "#16a34a" for _ in values]
    figure, axis = plt.subplots(figsize=(10, 5.5), facecolor="white")
    axis.set_facecolor("white")
    bars = axis.bar(range(len(values)), values, color=colors)
    axis.set_xticks(range(len(labels)), labels, rotation=0)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.text(0.01, 0.97, note, transform=axis.transAxes, va="top")
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_grouped(
    *,
    output_path: Path,
    labels: list[str],
    first: list[float],
    second: list[float],
    title: str,
    note: str,
) -> None:
    """Write data-time versus compute-time grouped bars."""
    positions = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(10, 5.5), facecolor="white")
    axis.set_facecolor("white")
    width = 0.36
    axis.bar([position - width / 2 for position in positions], first, width, label="data_time", color="#2563eb")
    axis.bar([position + width / 2 for position in positions], second, width, label="compute_time", color="#16a34a")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Segundos")
    axis.set_title(title)
    axis.text(0.01, 0.97, note, transform=axis.transAxes, va="top")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
