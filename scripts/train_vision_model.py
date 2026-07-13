"""CLI for Experiment A visual training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.vision.train import train_experiment


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Experiment A training."""
    parser = argparse.ArgumentParser(description="Train the ResNet18 soybean baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vision_config.yaml"),
        help="Path to the vision YAML configuration.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example cpu or cuda.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a one-epoch CPU-friendly smoke test on a tiny subset.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the expected checkpoint path when present.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """Run Experiment A training from the command line."""
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    device_name = args.device or ("cpu" if args.smoke_test else None)
    summary = train_experiment(
        config=config,
        device_name=device_name,
        smoke_test=args.smoke_test,
        resume=args.resume,
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())
