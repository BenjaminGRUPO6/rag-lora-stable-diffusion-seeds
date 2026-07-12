from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from src.data.audit import (
    audit_images,
    find_exact_duplicates,
    find_possible_near_duplicates,
    records_to_dicts,
    summarize,
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the dataset audit command."""

    parser = argparse.ArgumentParser(
        description=(
            "Audita un dataset de imágenes sin modificar "
            "los archivos originales."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Ruta de la carpeta raíz del dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Carpeta donde se guardarán los resultados.",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=200,
        help=(
            "Tamaño mínimo permitido para ancho y alto. "
            "Valor predeterminado: 200."
        ),
    )

    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=5,
        help=(
            "Distancia perceptual máxima para considerar dos "
            "imágenes como posibles duplicados visuales."
        ),
    )

    return parser.parse_args(argv)


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """
    Escribe un CSV incluso cuando no existen registros.

    De esta forma siempre se crea el archivo con sus encabezados.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset audit command."""

    arguments = parse_arguments(argv)

    dataset_path = arguments.dataset.resolve()
    output_path = arguments.output.resolve()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Iniciando auditoría...")
    print(f"Dataset: {dataset_path}")
    print(f"Resultados: {output_path}")
    print(f"Tamaño mínimo: {arguments.min_size}px")

    records = audit_images(
        dataset_root=dataset_path,
        min_size=arguments.min_size,
    )

    exact_duplicates = find_exact_duplicates(
        records
    )

    print(
        "Buscando posibles duplicados visuales. "
        "Este paso puede tardar unos minutos..."
    )

    near_duplicates = find_possible_near_duplicates(
        records,
        max_distance=arguments.near_duplicate_distance,
    )

    summary = summarize(
        records,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
    )

    image_rows = records_to_dicts(records)

    corrupted_rows = [
        {
            "relative_path": record.relative_path,
            "category": record.category,
            "extension": record.extension,
            "size_bytes": record.size_bytes,
            "error": record.error,
        }
        for record in records
        if not record.valid
    ]

    category_rows = [
        {
            "category": category,
            "count": count,
        }
        for category, count in sorted(
            summary["category_distribution"].items()
        )
    ]

    write_csv(
        output_path / "images.csv",
        image_rows,
        fieldnames=[
            "relative_path",
            "category",
            "extension",
            "width",
            "height",
            "mode",
            "size_bytes",
            "sha256",
            "perceptual_hash",
            "valid",
            "is_small",
            "error",
        ],
    )

    write_csv(
        output_path / "category_distribution.csv",
        category_rows,
        fieldnames=[
            "category",
            "count",
        ],
    )

    write_csv(
        output_path / "corrupted_files.csv",
        corrupted_rows,
        fieldnames=[
            "relative_path",
            "category",
            "extension",
            "size_bytes",
            "error",
        ],
    )

    write_csv(
        output_path / "exact_duplicates.csv",
        exact_duplicates,
        fieldnames=[
            "group_id",
            "sha256",
            "relative_path",
            "category",
            "size_bytes",
        ],
    )

    write_csv(
        output_path / "possible_near_duplicates.csv",
        near_duplicates,
        fieldnames=[
            "category",
            "path_a",
            "path_b",
            "phash_a",
            "phash_b",
            "phash_distance",
        ],
    )

    with (
        output_path / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Auditoría terminada.")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("Archivos generados:")

    for file_name in [
        "summary.json",
        "images.csv",
        "category_distribution.csv",
        "corrupted_files.csv",
        "exact_duplicates.csv",
        "possible_near_duplicates.csv",
    ]:
        print(f"- {output_path / file_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
