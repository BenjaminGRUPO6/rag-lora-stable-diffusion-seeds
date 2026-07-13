from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from src.vision.dataset import EXPECTED_CLASSES, create_dataloaders
from src.vision.evaluation import evaluate_model, load_checkpoint, save_evaluation_outputs
from src.vision.model import create_model
from src.vision.train import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for final test evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the ResNet18 soybean baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vision_config.yaml"),
        help="Path to the vision YAML configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the best training checkpoint.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device, for example cpu or cuda.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    """Evaluate the best checkpoint once on the test split."""
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    class_names = list(config.get("classes", EXPECTED_CLASSES))
    dataloaders = create_dataloaders(
        data_root=config["data"]["root"],
        classes=class_names,
        image_size=int(config["data"]["image_size"]),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        seed=int(config["training"]["seed"]),
        smoke_test=False,
    )
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model = create_model(
        architecture=str(config["model"]["architecture"]),
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, y_true, y_pred, probabilities = evaluate_model(
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
        output_dir=config["output"]["results_dir"],
        metrics_filename="metrics_test.json",
        save_predictions=True,
    )
    print(yaml.safe_dump(metrics, sort_keys=False))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
