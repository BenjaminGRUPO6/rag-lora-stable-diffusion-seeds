from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import run_demo


def test_app_packages_importable() -> None:
    """The application packages must be importable from the repository root."""
    import app
    import app.components
    import src

    assert app.__file__ is not None
    assert app.components.__file__ is not None
    assert src.__file__ is not None


def test_streamlit_entrypoint_exists_and_avoids_package_shadowing() -> None:
    """The Streamlit entrypoint must not be named app.py inside the app package."""
    assert run_demo.APP_PATH.name == "streamlit_app.py"
    assert run_demo.APP_PATH.exists()
    assert not (run_demo.PROJECT_ROOT / "app" / "app.py").exists()


def test_streamlit_command_uses_absolute_entrypoint() -> None:
    """The runner should work independently of the caller's current directory."""
    command = run_demo.streamlit_command(8501)

    assert command[:3] == [sys.executable, "-m", "streamlit"]
    assert Path(command[4]).is_absolute()
    assert Path(command[4]) == run_demo.APP_PATH
    assert "--server.port" in command


def test_run_demo_importable_from_other_working_directory(tmp_path: Path) -> None:
    """Importing the runner from another cwd should still resolve repository paths."""
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(run_demo.PROJECT_ROOT)!r}); "
        "from scripts import run_demo; "
        "assert run_demo.APP_PATH.exists(); "
        "assert run_demo.APP_PATH.is_absolute()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
