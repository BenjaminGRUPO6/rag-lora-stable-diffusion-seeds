from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from src.synthetic_data.prepare_lora_dataset import prepare_lora_dataset, settings_from_config


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for LoRA dataset preparation."""

    parser = argparse.ArgumentParser(description="Prepare the SD 1.5 LoRA dataset from train only.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def main(argv: Sequence[str] | None = None) -> int:
    """Run LoRA dataset preparation."""

    arguments = parse_arguments(argv)
    project_root = Path.cwd()
    config = load_config(arguments.config)
    settings = settings_from_config(
        config,
        project_root=project_root,
        images_per_class=arguments.images_per_class,
        seed=arguments.seed,
    )
    result = prepare_lora_dataset(
        settings,
        project_root=project_root,
        dry_run=arguments.dry_run,
        overwrite=arguments.overwrite,
    )

    mode = "Dry run" if result.dry_run else "Preparation"
    print(f"{mode} completed for LoRA dataset.")
    print(f"Source: {settings.source_dir}")
    print(f"Output: {result.output_dir}")
    print("Candidates and expected selection by class:")
    for label in settings.classes:
        print(
            f"- {label}: {result.selected_counts[label]} selected "
            f"from {result.candidate_counts[label]} candidates"
        )
    if not result.dry_run:
        print(f"Metadata: {result.metadata_path}")
        print(f"Selection report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
