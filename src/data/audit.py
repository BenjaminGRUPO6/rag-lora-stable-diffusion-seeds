from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageRecord:
    relative_path: str
    category: str
    extension: str
    width: int | None
    height: int | None
    mode: str | None
    size_bytes: int
    sha256: str | None
    perceptual_hash: str | None
    valid: bool
    is_small: bool | None
    error: str | None


def calculate_sha256(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def audit_images(
    dataset_root: Path,
    min_size: int = 200,
) -> list[ImageRecord]:
    """
    Audita las imágenes sin modificar los archivos originales.

    Parameters
    ----------
    dataset_root:
        Carpeta raíz del dataset.
    min_size:
        Tamaño mínimo permitido para ancho y alto.
    """

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"No existe el directorio del dataset: {dataset_root}"
        )

    records: list[ImageRecord] = []

    image_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for path in image_paths:
        relative_path = path.relative_to(dataset_root)
        category = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "unknown"
        )

        try:
            # Primera apertura para comprobar integridad.
            with Image.open(path) as image:
                image.verify()

            # Segunda apertura para leer propiedades.
            with Image.open(path) as image:
                image = image.convert("RGB")

                width, height = image.size
                mode = image.mode
                perceptual_hash = str(imagehash.phash(image))

            records.append(
                ImageRecord(
                    relative_path=relative_path.as_posix(),
                    category=category,
                    extension=path.suffix.lower(),
                    width=width,
                    height=height,
                    mode=mode,
                    size_bytes=path.stat().st_size,
                    sha256=calculate_sha256(path),
                    perceptual_hash=perceptual_hash,
                    valid=True,
                    is_small=width < min_size or height < min_size,
                    error=None,
                )
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:
            records.append(
                ImageRecord(
                    relative_path=relative_path.as_posix(),
                    category=category,
                    extension=path.suffix.lower(),
                    width=None,
                    height=None,
                    mode=None,
                    size_bytes=path.stat().st_size,
                    sha256=None,
                    perceptual_hash=None,
                    valid=False,
                    is_small=None,
                    error=str(error),
                )
            )

    return records


def find_exact_duplicates(
    records: list[ImageRecord],
) -> list[dict]:
    """Agrupa archivos que tienen exactamente el mismo SHA-256."""

    hash_groups: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records:
        if record.valid and record.sha256:
            hash_groups[record.sha256].append(record)

    duplicate_rows: list[dict] = []
    group_number = 1

    for sha256, group in hash_groups.items():
        if len(group) <= 1:
            continue

        group_id = f"exact_{group_number:04d}"

        for record in group:
            duplicate_rows.append(
                {
                    "group_id": group_id,
                    "sha256": sha256,
                    "relative_path": record.relative_path,
                    "category": record.category,
                    "size_bytes": record.size_bytes,
                }
            )

        group_number += 1

    return duplicate_rows


def find_possible_near_duplicates(
    records: list[ImageRecord],
    max_distance: int = 5,
) -> list[dict]:
    """
    Detecta imágenes visualmente similares mediante distancia perceptual.

    Solo compara imágenes de la misma categoría para reducir falsos positivos.
    """

    category_groups: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records:
        if record.valid and record.perceptual_hash:
            category_groups[record.category].append(record)

    possible_duplicates: list[dict] = []

    for category, category_records in category_groups.items():
        prepared = [
            (
                record,
                int(record.perceptual_hash, 16),
            )
            for record in category_records
        ]

        for index_a in range(len(prepared)):
            record_a, hash_a = prepared[index_a]

            for index_b in range(index_a + 1, len(prepared)):
                record_b, hash_b = prepared[index_b]

                # Los duplicados exactos ya se registran en otro archivo.
                if (
                    record_a.sha256
                    and record_a.sha256 == record_b.sha256
                ):
                    continue

                distance = (hash_a ^ hash_b).bit_count()

                if distance <= max_distance:
                    possible_duplicates.append(
                        {
                            "category": category,
                            "path_a": record_a.relative_path,
                            "path_b": record_b.relative_path,
                            "phash_a": record_a.perceptual_hash,
                            "phash_b": record_b.perceptual_hash,
                            "phash_distance": distance,
                        }
                    )

    return possible_duplicates


def summarize(
    records: list[ImageRecord],
    exact_duplicates: list[dict] | None = None,
    near_duplicates: list[dict] | None = None,
) -> dict:
    """Genera el resumen general de la auditoría."""

    valid_records = [
        record for record in records if record.valid
    ]

    corrupted_records = [
        record for record in records if not record.valid
    ]

    exact_duplicates = exact_duplicates or []
    near_duplicates = near_duplicates or []

    exact_group_count = len(
        {
            row["group_id"]
            for row in exact_duplicates
        }
    )

    return {
        "total_files": len(records),
        "valid_images": len(valid_records),
        "corrupted_images": len(corrupted_records),
        "small_images": sum(
            1
            for record in valid_records
            if record.is_small
        ),
        "exact_duplicate_groups": exact_group_count,
        "exact_duplicate_files": len(exact_duplicates),
        "possible_near_duplicate_pairs": len(near_duplicates),
        "category_distribution": dict(
            Counter(
                record.category
                for record in valid_records
            )
        ),
        "extension_distribution": dict(
            Counter(
                record.extension
                for record in valid_records
            )
        ),
        "width": {
            "minimum": min(
                (
                    record.width
                    for record in valid_records
                    if record.width is not None
                ),
                default=None,
            ),
            "maximum": max(
                (
                    record.width
                    for record in valid_records
                    if record.width is not None
                ),
                default=None,
            ),
            "average": (
                sum(
                    record.width
                    for record in valid_records
                    if record.width is not None
                )
                / len(valid_records)
                if valid_records
                else None
            ),
        },
        "height": {
            "minimum": min(
                (
                    record.height
                    for record in valid_records
                    if record.height is not None
                ),
                default=None,
            ),
            "maximum": max(
                (
                    record.height
                    for record in valid_records
                    if record.height is not None
                ),
                default=None,
            ),
            "average": (
                sum(
                    record.height
                    for record in valid_records
                    if record.height is not None
                )
                / len(valid_records)
                if valid_records
                else None
            ),
        },
    }


def records_to_dicts(
    records: list[ImageRecord],
) -> list[dict]:
    """Convierte los registros en diccionarios para exportarlos."""

    return [
        asdict(record)
        for record in records

    ]

def audit_dataset(
    dataset_dir: str | Path,
    min_size: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Función de compatibilidad con las pruebas anteriores.

    Devuelve dos DataFrames:
    - imágenes válidas;
    - imágenes corruptas.
    """

    records = audit_images(
        dataset_root=Path(dataset_dir),
        min_size=min_size,
    )

    valid_rows = [
        asdict(record)
        for record in records
        if record.valid
    ]

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

    valid_columns = [
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
    ]

    corrupted_columns = [
        "relative_path",
        "category",
        "extension",
        "size_bytes",
        "error",
    ]

    return (
        pd.DataFrame(
            valid_rows,
            columns=valid_columns,
        ),
        pd.DataFrame(
            corrupted_rows,
            columns=corrupted_columns,
        ),
    )


def summarize_images(
    images: pd.DataFrame,
    corrupted: pd.DataFrame,
) -> dict:
    """
    Función de compatibilidad con las pruebas anteriores.

    Resume los DataFrames creados por audit_dataset().
    """

    if images.empty:
        return {
            "total_valid": 0,
            "total_corrupted": int(len(corrupted)),
            "small_images": 0,
            "exact_duplicate_files": 0,
            "categories": {},
            "extensions": {},
        }

    exact_duplicate_files = 0

    if "sha256" in images.columns:
        exact_duplicate_files = int(
            images.duplicated(
                subset=["sha256"],
                keep=False,
            ).sum()
        )

    small_images = 0

    if "is_small" in images.columns:
        small_images = int(
            images["is_small"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    categories = {}

    if "category" in images.columns:
        categories = (
            images["category"]
            .value_counts()
            .to_dict()
        )

    extensions = {}

    if "extension" in images.columns:
        extensions = (
            images["extension"]
            .value_counts()
            .to_dict()
        )

    return {
        "total_valid": int(len(images)),
        "total_corrupted": int(len(corrupted)),
        "small_images": small_images,
        "exact_duplicate_files": exact_duplicate_files,
        "categories": categories,
        "extensions": extensions,
    }