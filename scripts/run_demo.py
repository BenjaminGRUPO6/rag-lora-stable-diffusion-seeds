from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


APP_PATH = Path("app/app.py")


def main(argv: list[str] | None = None) -> int:
    """Start the Streamlit demo or run a finite startup smoke test."""
    parser = argparse.ArgumentParser(description="Run the SeedCare-RAG Streamlit demo.")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Keep Streamlit running for interactive use.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to use. Defaults to a free local port.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=25.0,
        help="Seconds to wait for Streamlit startup during smoke test.",
    )
    args = parser.parse_args(argv)

    port = args.port or find_free_port()
    command = streamlit_command(port)
    if args.serve:
        print(f"SeedCare-RAG demo: http://localhost:{port}")
        return subprocess.call(command)
    return smoke_test(command, port, args.timeout)


def streamlit_command(port: int) -> list[str]:
    """Build a Windows-compatible Streamlit command."""
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


def smoke_test(command: list[str], port: int, timeout: float) -> int:
    """Start Streamlit, wait for health endpoint, then stop it."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output: list[str] = []
    started_at = time.monotonic()
    try:
        while time.monotonic() - started_at < timeout:
            if process.poll() is not None:
                if process.stdout is not None:
                    output.extend(process.stdout.readlines())
                print("Streamlit termino antes de iniciar correctamente.")
                print("".join(output[-40:]))
                return process.returncode or 1
            if healthcheck(port):
                print(f"Streamlit inicio correctamente en http://localhost:{port}")
                return 0
            time.sleep(0.5)
        print(f"Streamlit no respondio antes de {timeout:.1f} segundos.")
        return 1
    finally:
        stop_process(process)


def healthcheck(port: int) -> bool:
    """Return true when Streamlit health endpoint responds."""
    try:
        with urlopen(f"http://localhost:{port}/_stcore/health", timeout=1.0) as response:
            return 200 <= int(response.status) < 500
    except URLError:
        return False
    except TimeoutError:
        return False


def stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate a subprocess without leaving Streamlit running."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def find_free_port() -> int:
    """Return an available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
