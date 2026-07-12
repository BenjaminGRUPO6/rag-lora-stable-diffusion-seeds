from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    frame: pd.DataFrame,
    label_column: str = "main_label",
    seed: int = 42,
    train_size: float = 0.8,
    validation_size: float = 0.1,
) -> pd.DataFrame:
    if abs(train_size + validation_size - 0.9) > 1e-9:
        raise ValueError("La configuración esperada reserva 10% para test.")

    train, remaining = train_test_split(
        frame,
        train_size=train_size,
        random_state=seed,
        stratify=frame[label_column],
    )
    validation, test = train_test_split(
        remaining,
        train_size=0.5,
        random_state=seed,
        stratify=remaining[label_column],
    )
    output = frame.copy()
    output["split"] = ""
    output.loc[train.index, "split"] = "train"
    output.loc[validation.index, "split"] = "validation"
    output.loc[test.index, "split"] = "test"
    return output
