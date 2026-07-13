from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"
DEFAULT_PORT = 8501


def main(argv: list[str] | None = None) -> int:
    """Start the persistent Streamlit demo until the user presses Ctrl+C."""
    parser = argparse.ArgumentParser(description="Run the SeedCare-RAG Streamlit demo.")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to use. Defaults to {DEFAULT_PORT}.",
    )
    args = parser.parse_args(argv)

    command = streamlit_command(args.port)
    print(f"SeedCare-RAG demo: http://127.0.0.1:{args.port}", flush=True)
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        return int(process.wait())
    except KeyboardInterrupt:
        stop_process(process)
        print("\nStreamlit detenido por Ctrl+C.", flush=True)
        return 130


def streamlit_command(port: int) -> list[str]:
    """Build the Windows-compatible persistent Streamlit command."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """Stop the Streamlit child process started by the demo wrapper."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
