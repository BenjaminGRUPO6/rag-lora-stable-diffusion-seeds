from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from streamlit.testing.v1 import AppTest

from scripts import smoke_test_app
from scripts import run_demo
from src.synthetic_data.lora_evidence import MANDATORY_EXPLANATION

FORBIDDEN_LOG_MARKERS = (
    "ModuleNotFoundError",
    "No module named 'app'",
    'No module named "app"',
    "Traceback",
    "ImportError",
)


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


def test_streamlit_entrypoint_does_not_reference_legacy_app_module() -> None:
    """The official entrypoint must not import or execute the legacy app/app.py path."""
    source = run_demo.APP_PATH.read_text(encoding="utf-8")

    assert "app.app" not in source
    assert "app/app.py" not in source
    assert "app\\app.py" not in source
    assert "runpy" not in source
    assert "exec(" not in source
    assert "st.navigation" not in source
    assert "st.Page" not in source


def test_streamlit_command_uses_absolute_entrypoint() -> None:
    """The runner should work independently of the caller's current directory."""
    command = run_demo.streamlit_command(8501)

    assert command[:3] == [sys.executable, "-m", "streamlit"]
    assert Path(command[4]).is_absolute()
    assert Path(command[4]) == run_demo.APP_PATH
    assert "--server.port" in command
    assert "8501" in command


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


def test_streamlit_entrypoint_renders_title_without_apptest_exceptions() -> None:
    """Streamlit's test framework should render the app title without exceptions."""
    app_test = AppTest.from_file(str(run_demo.APP_PATH), default_timeout=30)
    app_test.run()

    assert len(app_test.exception) == 0
    assert any(title.value == "SeedCare-RAG" for title in app_test.title)


def test_streamlit_entrypoint_renders_required_tabs_and_graphics() -> None:
    """The Streamlit UI should expose the five requested tabs and local result PNGs."""
    app_test = AppTest.from_file(str(run_demo.APP_PATH), default_timeout=30)
    app_test.run()

    labels = [tab.label for tab in app_test.tabs]

    assert len(app_test.exception) == 0
    assert labels == [
        "A. Análisis",
        "B. Explicabilidad",
        "C. Evidencia RAG",
        "D. Resultados 1 vs Resultados 2",
        "Modelo generativo LoRA",
    ]
    assert len(app_test.image) >= 1


def test_streamlit_lora_tab_is_visual_and_does_not_load_generation_pipeline() -> None:
    """The LoRA tab must stay separate from classifier inference and SD loading."""
    source = run_demo.APP_PATH.read_text(encoding="utf-8")

    assert "Modelo generativo LoRA" in source
    assert "MANDATORY_EXPLANATION" in source
    assert MANDATORY_EXPLANATION == (
        "El LoRA genera imágenes sintéticas de semillas. No clasifica la imagen "
        "cargada y no modifica la confianza del clasificador ResNet18."
    )
    assert "StableDiffusionPipeline" not in source
    assert "from diffusers" not in source
    assert "safetensors.torch" not in source


def test_streamlit_entrypoint_exposes_gradcam_fallback_copy() -> None:
    """The explainability tab should render a fallback state before analysis."""
    app_test = AppTest.from_file(str(run_demo.APP_PATH), default_timeout=30)
    app_test.run()

    captions = [caption.value for caption in app_test.caption]
    infos = [info.value for info in app_test.info]

    assert len(app_test.exception) == 0
    assert any("Ejecuta un analisis para generar Grad-CAM" in value for value in captions)
    assert any("Grad-CAM es una explicacion aproximada" in value for value in infos)


def test_streamlit_subprocess_runs_from_other_directory_without_import_errors(
    tmp_path: Path,
) -> None:
    """The absolute entrypoint should run from another cwd without app import failures."""
    port = smoke_test_app.find_free_port()
    command = run_demo.streamlit_command(port)
    process = subprocess.Popen(
        command,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        http_status = wait_for_http_status(port=port, timeout=30.0)
        process_alive = process.poll() is None
        port_listening = smoke_test_app.is_port_listening(port)
    finally:
        stdout, stderr = smoke_test_app.stop_and_collect(process)

    combined_output = f"{stdout}\n{stderr}"
    assert http_status == 200
    assert process_alive
    assert port_listening
    assert not any(marker in combined_output for marker in FORBIDDEN_LOG_MARKERS), combined_output


def wait_for_http_status(port: int, timeout: float) -> int:
    """Wait until Streamlit returns the first root HTTP status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/", timeout=2.0) as response:
                return int(response.status)
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    return 0
