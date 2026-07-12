from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def classification_metrics(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict:
    return classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)


def save_confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[str], path: str | Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output)
