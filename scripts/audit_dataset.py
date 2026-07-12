from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.audit import audit_dataset, summarize_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita un dataset de imágenes sin modificarlo.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="results/dataset_audit")
    parser.add_argument("--min-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images, corrupted = audit_dataset(args.dataset, min_size=args.min_size)
    images.to_csv(output / "images.csv", index=False)
    corrupted.to_csv(output / "corrupted_files.csv", index=False)
    if not images.empty:
        images[images.duplicated("sha256", keep=False)].sort_values("sha256").to_csv(
            output / "exact_duplicates.csv", index=False
        )
        images.groupby("category").size().rename("count").reset_index().to_csv(
            output / "category_distribution.csv", index=False
        )
    else:
        (output / "exact_duplicates.csv").write_text("", encoding="utf-8")
        (output / "category_distribution.csv").write_text("", encoding="utf-8")
    summary = summarize_images(images, corrupted)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
