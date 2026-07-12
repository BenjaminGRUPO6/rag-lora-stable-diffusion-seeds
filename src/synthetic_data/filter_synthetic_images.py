from __future__ import annotations

import pandas as pd

REVIEW_COLUMNS = [
    "file_path",
    "target_label",
    "prompt",
    "reviewer",
    "accepted",
    "visual_quality_1_5",
    "label_fidelity_1_5",
    "notes",
]


def empty_review_sheet() -> pd.DataFrame:
    return pd.DataFrame(columns=REVIEW_COLUMNS)
