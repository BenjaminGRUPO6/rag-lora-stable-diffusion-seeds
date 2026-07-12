from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.cleaning import validate_metadata
from src.data.split_dataset import stratified_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="data/metadata/images_with_split.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.metadata)
    errors = validate_metadata(frame)
    if errors:
        raise ValueError(" | ".join(errors))
    verified = frame[frame["verified"].astype(bool) & ~frame["synthetic"].astype(bool)].copy()
    output = stratified_split(verified)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(output["split"].value_counts())


if __name__ == "__main__":
    main()
