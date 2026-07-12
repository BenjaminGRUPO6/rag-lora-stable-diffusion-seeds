from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_caption(row: pd.Series, trigger_word: str = "soybeanseed") -> str:
    seed_type = row.get("seed_type", "soybean")
    label = row.get("main_label", "unknown")
    sub_label = row.get("sub_label", "")
    detail = f", {sub_label}" if isinstance(sub_label, str) and sub_label else ""
    return (
        f"documentary photo of {trigger_word} {seed_type}, {label}{detail}, "
        "neutral background, detailed seed surface, agricultural inspection photography"
    )


def create_metadata_jsonl(frame: pd.DataFrame, output_dir: str | Path, trigger_word: str = "soybeanseed") -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as file:
        for _, row in frame.iterrows():
            record = {"file_name": Path(str(row["file_path"])).name, "text": build_caption(row, trigger_word)}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metadata_path
