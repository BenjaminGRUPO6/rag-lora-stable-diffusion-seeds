from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SeedDamageDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        root_dir: str | Path,
        label_map: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        path = self.root_dir / str(row["file_path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_map[str(row["main_label"])]
        return image, label
