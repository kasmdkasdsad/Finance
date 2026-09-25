import socket
import threading
import time

import httpx
import pytest
import uvicorn

from quantpulse.api.app import create_app
from quantpulse.config import Settings


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def api_server(tmp_path_factory):
    """A real QuantPulse API (offline mode) served on a local port for the Streamlit pages."""
    db = tmp_path_factory.mktemp("frontend") / "ui.db"
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{db}",
        enable_live_data=False,
        polling_enabled=False,
        log_level="WARNING",
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("API server did not start")
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    # Train today's model and research once, so page tests exercise rendering rather than the background wait
    # (the progress display for a model that is still training has its own test).
    with httpx.Client(base_url=url, timeout=600) as client:
        for path in ("/api/v1/model/report", "/api/v1/model/research"):
            client.get(path, params={"wait": 600}).raise_for_status()
    yield url
    server.should_exit = True
    thread.join(timeout=10)
