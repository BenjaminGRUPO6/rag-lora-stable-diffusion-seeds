from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_image_metadata(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
