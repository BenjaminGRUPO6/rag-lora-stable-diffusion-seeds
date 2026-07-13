from __future__ import annotations

from pathlib import Path

from scripts import smoke_test_app


def test_smoke_streamlit_command_uses_expected_entrypoint() -> None:
    command = smoke_test_app.streamlit_command(8765)

    assert command[:4] == [smoke_test_app.sys.executable, "-m", "streamlit", "run"]
    assert Path(command[4]) == smoke_test_app.APP_PATH
    assert "--server.address" in command
    assert "127.0.0.1" in command
    assert "--server.port" in command
    assert "8765" in command


def test_find_free_port_returns_listenable_port() -> None:
    port = smoke_test_app.find_free_port()

    assert isinstance(port, int)
    assert port > 0
