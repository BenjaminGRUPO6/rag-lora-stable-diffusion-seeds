"""CLI for the ResNet18 V2 visual experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.vision.train_v2 import train_resnet18_v2


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ResNet18 V2 for Resultados 2.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vision_v2_resnet18.yaml"),
        help="Path to the ResNet18 V2 YAML configuration.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device, for example cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the V2 checkpoint when present.",
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
    """Run ResNet18 V2 training."""
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    summary = train_resnet18_v2(
        config=config,
        device_name=args.device,
        smoke_test=args.smoke_test,
        resume=args.resume,
        use_class_weights=True if args.use_class_weights else None,
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())
