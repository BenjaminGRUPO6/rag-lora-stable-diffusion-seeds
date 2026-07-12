from __future__ import annotations

import importlib.util
import platform
import sys

PACKAGES = ["PIL", "yaml", "pandas", "numpy", "sklearn", "imagehash", "pytest"]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.platform()}")
    missing = [name for name in PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        print("Paquetes faltantes:", ", ".join(missing))
        return 1
    print("Entorno core correcto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
