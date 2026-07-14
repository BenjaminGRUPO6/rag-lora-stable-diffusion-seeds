"""CLI for the EfficientNet-B0 V2 visual comparison experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.train_v2 import train_efficientnet_b0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for Resultados 2.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vision_v2_efficientnet_b0.yaml"),
        help="Path to the EfficientNet-B0 YAML configuration.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device, for example cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument(
        "--disable-auto-crop",
        action="store_true",
        help="Disable automatic crop preprocessing.",
    )
    parser.add_argument(
        "--cache-preprocessing",
        action="store_true",
        help="Precompute and read automatic crops from data/cache/vision_crops.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples per split for smoke tests and benchmarks.",
    )
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=None,
        help="Print batch progress every N batches.",
    )
    parser.add_argument(
        "--checkpoint-every-epoch",
        action="store_true",
        help="Write recovery checkpoints after every epoch.",
    )
    parser.add_argument(
        "--profile-dataloader",
        action="store_true",
        help="Measure a few DataLoader batches before training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the EfficientNet-B0 checkpoint when present.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a one-epoch CPU-friendly smoke test on a tiny subset.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights calculated only from train distribution.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """Run EfficientNet-B0 training with OOM fallback to batch size 4."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config.setdefault("data", {})
    config.setdefault("preprocessing", {})
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.disable_auto_crop:
        config["preprocessing"]["auto_crop"] = False
    if args.cache_preprocessing:
        config["preprocessing"]["cache_preprocessing"] = True
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.log_every_n_batches is not None:
        config["log_every_n_batches"] = args.log_every_n_batches
    if args.checkpoint_every_epoch:
        config["checkpoint_every_epoch"] = True
    if args.profile_dataloader:
        config["profile_dataloader"] = True
    attempted_batch_sizes = [int(config.get("batch_size", 8))]
    for candidate in (4, 2):
        if candidate < attempted_batch_sizes[0]:
            attempted_batch_sizes.append(candidate)
    oom_batch_sizes: list[int] = []
    summary: dict[str, Any] | None = None
    for batch_size in attempted_batch_sizes:
        config["batch_size"] = batch_size
        try:
            summary = train_efficientnet_b0(
                config=config,
                device_name=args.device,
                smoke_test=args.smoke_test,
                resume=args.resume or bool(oom_batch_sizes),
                use_class_weights=True if args.use_class_weights else None,
            )
            break
        except RuntimeError as error:
            is_oom = "out of memory" in str(error).lower()
            if args.smoke_test or not is_oom or batch_size == attempted_batch_sizes[-1]:
                raise
            oom_batch_sizes.append(batch_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"CUDA OOM with batch_size={batch_size}; retrying smaller batch.", flush=True)
    if summary is None:
        raise RuntimeError("EfficientNet training did not produce a summary.")
    if oom_batch_sizes:
        summary["oom_batch_sizes"] = oom_batch_sizes
        summary["final_batch_size"] = int(config["batch_size"])
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())
