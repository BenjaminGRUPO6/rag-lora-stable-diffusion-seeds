from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from src.data.verify import verify_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for dataset structure verification."""
    parser = argparse.ArgumentParser(description="Verifica la estructura del dataset Soybean Seeds sin modificar archivos.")
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args(argv)


def _format_names(names: tuple[str, ...]) -> str:
    """Return a readable comma-separated list for report sections."""
    return ", ".join(names) if names else "ninguna"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset structure verification command."""
    args = parse_args(argv)
    try:
        result = verify_dataset(args.dataset)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2

    print("Categorías encontradas:", _format_names(result.found_categories))
    print("Categorías faltantes:", _format_names(result.missing_classes))
    print("Carpetas inesperadas:", _format_names(result.unexpected_directories))
    print("Cantidad por categoría:")
    for class_name, count in result.counts.items():
        print(f"  {class_name}: {count}")
    print(f"Total general: {result.total}")
    print("Extensiones encontradas:", _format_names(result.extensions))
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
