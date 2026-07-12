from __future__ import annotations

import argparse
from pathlib import Path

from src.data.verify import verify_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica la estructura del dataset Soybean Seeds sin modificar archivos.")
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_dataset(args.dataset)
    for class_name, count in result.counts.items():
        print(f"{class_name}: {count}")
    print(f"Total: {result.total}")
    print(f"Extensiones: {', '.join(result.extensions) or 'ninguna'}")
    if result.missing_classes:
        print("Clases faltantes:", ", ".join(result.missing_classes))
    if result.unexpected_directories:
        print("Carpetas inesperadas:", ", ".join(result.unexpected_directories))
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
