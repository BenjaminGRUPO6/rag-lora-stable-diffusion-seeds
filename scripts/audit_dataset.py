from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from src.data.audit import audit_images, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita el dataset sin modificar imágenes.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = audit_images(args.dataset)
    summary = summarize(records)

    with (args.output / "images.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(records[0]).keys()) if records else ["relative_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
