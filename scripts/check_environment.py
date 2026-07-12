from __future__ import annotations

import importlib.util
import platform
import sys

MODULES = ["PIL", "pandas", "yaml", "sklearn", "pytest"]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.platform()}")
    missing = [module for module in MODULES if importlib.util.find_spec(module) is None]
    if missing:
        print("Faltan módulos:", ", ".join(missing))
        return 1
    print("Entorno base correcto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
