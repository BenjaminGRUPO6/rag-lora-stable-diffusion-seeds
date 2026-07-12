from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "image_id",
    "file_path",
    "seed_type",
    "main_label",
    "verified",
    "source",
    "synthetic",
}


def validate_metadata(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        errors.append(f"Faltan columnas: {sorted(missing)}")
    if "image_id" in frame.columns and frame["image_id"].duplicated().any():
        errors.append("Existen image_id duplicados.")
    if "main_label" in frame.columns and frame["main_label"].isna().any():
        errors.append("Existen imágenes sin etiqueta principal.")
    return errors
