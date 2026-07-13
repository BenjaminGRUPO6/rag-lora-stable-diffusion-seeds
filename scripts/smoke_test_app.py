from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "app_smoke_test"
EXPECTED_TITLE = "SeedCare-RAG"


@dataclass(frozen=True)
class SmokeSummary:
    """Serializable result of the Streamlit startup smoke test."""

    status: str
    command: list[str]
    port: int
    process_alive: bool
    port_listening: bool
    http_status: int
    http_ok: bool
    rendered_title_found: bool
    traceback_found: bool
    stdout_log: str
    stderr_log: str


def parse_args() -> argparse.Namespace:
    """Parse smoke test command line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test the SeedCare-RAG Streamlit app.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Run the smoke test and persist logs plus summary JSON."""
    args = parse_args()
    summary = run_smoke_test(timeout=args.timeout, output_dir=args.output_dir)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0 if summary.status == "PASS" else 1


def run_smoke_test(timeout: float = 30.0, output_dir: Path = DEFAULT_OUTPUT_DIR) -> SmokeSummary:
    """Start Streamlit, validate HTTP startup, render the app, and save logs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    port = find_free_port()
    command = streamlit_command(port)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    http_status = 0
    http_ok = False
    port_listening = False
    process_alive = False
    rendered_title_found = False
    try:
        http_status = wait_for_http_status(port=port, timeout=timeout)
        http_ok = http_status == 200
        port_listening = is_port_listening(port)
        process_alive = process.poll() is None
        rendered_title_found = render_contains_title(APP_PATH, EXPECTED_TITLE)
    finally:
        stdout, stderr = stop_and_collect(process)

    traceback_found = "Traceback (most recent call last)" in f"{stdout}\n{stderr}"
    status = (
        "PASS"
        if http_ok and port_listening and process_alive and rendered_title_found and not traceback_found
        else "FAIL"
    )
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    summary = SmokeSummary(
        status=status,
        command=sanitize_command(command),
        port=port,
        process_alive=process_alive,
        port_listening=port_listening,
        http_status=http_status,
        http_ok=http_ok,
        rendered_title_found=rendered_title_found,
        traceback_found=traceback_found,
        stdout_log=relative_to_project(stdout_path),
        stderr_log=relative_to_project(stderr_path),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def streamlit_command(port: int) -> list[str]:
    """Build the Streamlit startup command."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]


def sanitize_command(command: list[str]) -> list[str]:
    """Return a display-safe command without private absolute paths."""
    sanitized: list[str] = []
    for item in command:
        if item == str(APP_PATH):
            sanitized.append("app/streamlit_app.py")
        elif item == sys.executable:
            sanitized.append("python")
        else:
            sanitized.append(item.replace(str(PROJECT_ROOT), "."))
    return sanitized


def relative_to_project(path: Path) -> str:
    """Return a POSIX relative path for generated artifacts."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def wait_for_http_status(port: int, timeout: float) -> int:
    """Wait until the Streamlit root URL returns an HTTP status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/", timeout=2.0) as response:
                return int(response.status)
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    return 0


def is_port_listening(port: int) -> bool:
    """Return true when localhost accepts TCP connections on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def render_contains_title(app_path: Path, expected_title: str) -> bool:
    """Render the Streamlit script with AppTest and check for the app title."""
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.run()
    return any(title.value == expected_title for title in app.title)


def stop_and_collect(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Stop Streamlit and return captured stdout and stderr."""
    if process.poll() is None:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
    else:
        stdout, stderr = process.communicate(timeout=5)
    return stdout or "", stderr or ""


def find_free_port() -> int:
    """Return an available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
