from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe la configuración: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}
