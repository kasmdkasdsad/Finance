"""The desktop launcher end to end: it really starts the API and the Streamlit UI from the project's virtual
environment, reuses them on a second start, reports their status and stops them cleanly — and nothing it
does places an order.

Everything runs on free ports, with the launcher's logs/state in a temporary folder (QP_LAUNCHER_HOME) and
the application's settings in a temporary .env (QP_ENV_FILE): no live data, no Alpaca keys, no network.
"""

import json
import os
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "launcher" / "quantpulse_launcher.py"
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

pytestmark = pytest.mark.skipif(not PYTHON.exists(), reason="needs the project's .venv")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str) -> tuple[int, bytes] | None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=3) as r:
            return r.status, r.read()
    except OSError:
        return None


@pytest.fixture
def launch(tmp_path):
    home = tmp_path / "launcher-home"
    env_file = tmp_path / ".env"
    token = "tok-e2e-5d4c3b2a1f"
    env_file.write_text(
        "\n".join(
            [
                f"QP_DATABASE_URL=sqlite+aiosqlite:///{(tmp_path / 'e2e.db').as_posix()}",
                "QP_POLLING_ENABLED=false",
                "QP_ENABLE_LIVE_DATA=false",
                f"QP_API_TOKEN={token}",
                "QP_ALPACA_PAPER=true",
                "QP_ALPACA_TRADING_ENABLED=true",
                "QP_TRADING_DRY_RUN=false",
                "QP_TRADING_KILL_SWITCH=false",
                "QP_TRADING_SCHEDULER_ENABLED=false",
            ]
        )
    )
    ports = {"api": free_port(), "ui": free_port()}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QP_", "APCA_", "ALPACA_"))}
    env.update(QP_ENV_FILE=str(env_file), QP_LAUNCHER_HOME=str(home))

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(PYTHON),
                str(LAUNCHER),
                *args,
                "--api-port",
                str(ports["api"]),
                "--ui-port",
                str(ports["ui"]),
            ],
            cwd=str(tmp_path),  # never the project folder: the launcher must not rely on it
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    run.ports, run.home, run.token = ports, home, token
    yield run
    run("stop", "--no-dialogs")  # never leave servers behind


def servers(home: Path) -> dict:
    return json.loads((home / "launcher-state.json").read_text())


def test_start_reuse_status_and_stop(launch):
    api, ui = launch.ports["api"], launch.ports["ui"]
    first = launch("start", "--no-browser", "--no-splash", "--no-dialogs")
    assert first.returncode == 0, first.stderr
    assert get(f"http://127.0.0.1:{api}/health")[1] == b'{"status":"ok","version":"1.0.0"}'
    assert get(f"http://127.0.0.1:{ui}/_stcore/health") == (200, b"ok")
    state = servers(launch.home)
    assert {state["api"]["port"], state["ui"]["port"]} == {api, ui}
    assert "Alpaca PAPER account" in first.stderr and "scheduler off" in first.stderr
    assert (
        "mode dry_run" in first.stderr and "keys missing" in first.stderr
    )  # no keys here: nothing can be sent

    again = launch("start", "--page", "trading", "--no-browser", "--no-splash", "--no-dialogs")
    assert again.returncode == 0 and "reused: api, ui" in again.stderr
    assert servers(launch.home) == state  # no second copy was started
    assert f"http://127.0.0.1:{ui}/trading" in again.stderr

    status = launch("status", "--no-dialogs")
    assert f"API: running on port {api} (pid {state['api']['pid']}, started by the launcher)" in status.stdout

    stop = launch("stop", "--no-dialogs")
    assert "QuantPulse stopped: UI (stopped), API (stopped)." in stop.stderr
    assert (
        get(f"http://127.0.0.1:{api}/health") is None and get(f"http://127.0.0.1:{ui}/_stcore/health") is None
    )

    api_log = (launch.home / "api.log").read_text()
    assert "Application shutdown complete" in api_log  # a clean shutdown, not a kill
    requests = re.findall(r'"([A-Z]+) (/[^ ]*) HTTP', api_log)
    assert requests and {m for m, _ in requests} == {"GET"}  # starting QuantPulse never sends an order
    assert not any(p.startswith(("/api/v1/trading/run", "/api/v1/trading/test-order")) for _, p in requests)
    for log in launch.home.glob("*.log"):
        assert launch.token not in log.read_text(), f"the API token leaked into {log.name}"


def test_a_port_held_by_another_program_is_reported_and_nothing_starts(launch):
    holder = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(launch.ports["api"]), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if get(f"http://127.0.0.1:{launch.ports['api']}/") is not None:
                break
            __import__("time").sleep(0.1)
        out = launch("start", "--no-browser", "--no-splash", "--no-dialogs")
        assert out.returncode == 1 and f"Port {launch.ports['api']} is used by another program" in out.stderr
        assert get(f"http://127.0.0.1:{launch.ports['ui']}/_stcore/health") is None
    finally:
        holder.terminate()
        holder.wait(timeout=10)
