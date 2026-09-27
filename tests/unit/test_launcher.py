"""The Windows desktop launcher (launcher/quantpulse_launcher.py): pure pieces, probes and safety audits."""

import ast
import http.server
import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "launcher" / "quantpulse_launcher.py"


def load():
    spec = importlib.util.spec_from_file_location("quantpulse_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


ql = load()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """A throwaway local HTTP server answering GETs from a dict {path: (status, body)}."""

    def __init__(self, routes):
        self.seen = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append((self.command, self.path, dict(self.headers)))
                status, body = routes.get(self.path.split("?")[0], (404, b"not found"))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return None

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(routes):
        made.append(_Server(routes))
        return made[-1]

    yield make
    for s in made:
        s.close()


# --------------------------------------------------------------------------- settings and secrets
def test_env_files_are_read_like_the_application_reads_them(tmp_path):
    f = tmp_path / ".env"
    f.write_bytes(
        b"\xef\xbb\xbf# comment\nQP_API_TOKEN=tok-123456789\nexport QP_TRADING_DRY_RUN=false\n"
        b'QP_QUOTED="a b # not a comment"\nqp_lower=x  # trailing comment\nnot a line\n'
    )
    got = ql.read_env_file(f)
    assert got == {
        "QP_API_TOKEN": "tok-123456789",
        "QP_TRADING_DRY_RUN": "false",
        "QP_QUOTED": "a b # not a comment",
        "QP_LOWER": "x",
    }
    assert ql.env_file(tmp_path, {}) == f
    assert ql.env_file(tmp_path, {"QP_ENV_FILE": str(tmp_path / "other.env")}) == tmp_path / "other.env"
    merged = ql.effective_settings(tmp_path, {"QP_API_TOKEN": "from-the-environment"})
    assert merged["QP_API_TOKEN"] == "from-the-environment" and merged["QP_TRADING_DRY_RUN"] == "false"


def test_secrets_are_scrubbed_from_every_log_line_and_dialog():
    settings = {
        "QP_ALPACA_API_KEY_ID": "PKABCDEF123456",
        "QP_ALPACA_API_SECRET_KEY": "s3cr3t-value-xyz",
        "QP_API_TOKEN": "tok-987654321",
        "QP_TRADING_DRY_RUN": "false",
        "QP_SOME_KEY": "true",  # too short to be a secret: left alone
    }
    secrets = ql.secret_values(settings)
    assert secrets == {"PKABCDEF123456", "s3cr3t-value-xyz", "tok-987654321"}
    r = ql.Redactor(secrets)
    assert r.redact("key PKABCDEF123456 secret s3cr3t-value-xyz ok") == "key [redacted] secret [redacted] ok"


def test_the_ui_gets_the_api_address_and_token_and_nothing_trading_related(monkeypatch):
    monkeypatch.delenv("QP_API_TOKEN", raising=False)
    settings = {"QP_API_TOKEN": "tok-from-dotenv", "QP_TRADING_SCHEDULER_ENABLED": "false"}
    api = ql.child_env("api", settings, 8000)
    ui = ql.child_env("ui", settings, 8000)
    assert "QP_API_TOKEN" not in api and "QP_API_URL" not in api  # the API reads .env itself
    assert ui["QP_API_URL"] == "http://127.0.0.1:8000" and ui["QP_API_TOKEN"] == "tok-from-dotenv"
    added = {k for env in (api, ui) for k in env if k not in __import__("os").environ}
    assert added <= {"PYTHONUNBUFFERED", "PYTHONIOENCODING", "QP_API_URL", "QP_API_TOKEN"}
    assert not any(k.startswith(("QP_TRADING", "QP_ALPACA")) for k in added)


def test_server_commands_bind_this_machine_only(tmp_path):
    py = tmp_path / "python"
    assert ql.api_command(py, 8000) == [
        str(py), "-m", "uvicorn", "--factory", "quantpulse.api.app:app_factory",
        "--host", "127.0.0.1", "--port", "8000",
    ]  # fmt: skip
    ui = ql.ui_command(py, 8501, tmp_path)
    assert ui[:5] == [str(py), "-m", "streamlit", "run", str(tmp_path / "frontend" / "app.py")]
    opts = dict(zip(ui[5::2], ui[6::2], strict=True))
    assert opts == {
        "--server.address": "127.0.0.1",
        "--server.port": "8501",
        "--server.headless": "true",
        "--browser.gatherUsageStats": "false",
    }


# --------------------------------------------------------------------------- probes
def test_probes_tell_quantpulse_from_other_programs(server):
    qp = server(
        {
            "/health": (200, b'{"status":"ok","version":"1.0.0"}'),
            "/openapi.json": (200, json.dumps({"info": {"title": "QuantPulse Terminal API"}}).encode()),
        }
    )
    other = server({"/health": (200, b'{"status":"ok"}'), "/openapi.json": (200, b'{"info":{"title":"X"}}')})
    web = server({"/": (200, b"<html>")})
    ui = server({"/_stcore/health": (200, b"ok")})
    assert ql.probe_api(qp.port) == "quantpulse"
    assert ql.probe_api(other.port) == "other" and ql.probe_api(web.port) == "other"
    assert ql.probe_api(free_port()) == "free"
    assert ql.probe_ui(ui.port) == "streamlit" and ql.probe_ui(web.port) == "other"
    assert ql.probe_ui(free_port()) == "free"
    assert {m for s in (qp, other, web, ui) for m, _, _ in s.seen} == {"GET"}


def test_trading_status_is_read_with_the_token_and_warns_about_the_scheduler(server):
    status = {
        "paper": True,
        "mode": "paper",
        "trading_enabled": True,
        "dry_run": False,
        "kill_switch": {"active": False},
        "scheduler_enabled": False,
        "broker_configured": True,
    }
    s = server({ql.TRADING_STATUS_PATH: (200, json.dumps(status).encode())})
    summary, warnings = ql.trading_status(s.port, {"QP_API_TOKEN": "tok-abcdef"})
    assert summary.startswith("Alpaca PAPER account · mode paper") and "scheduler off" in summary
    assert warnings == []
    [(method, path, headers)] = s.seen
    assert method == "GET" and path == ql.TRADING_STATUS_PATH
    assert {k.lower(): v for k, v in headers.items()}["x-api-key"] == "tok-abcdef"
    s2 = server({ql.TRADING_STATUS_PATH: (200, json.dumps({**status, "scheduler_enabled": True}).encode())})
    _, warnings = ql.trading_status(s2.port, {})
    assert len(warnings) == 1 and "QP_TRADING_SCHEDULER_ENABLED=false" in warnings[0]
    s3 = server({ql.TRADING_STATUS_PATH: (401, b'{"detail":"bad key"}')})
    assert ql.trading_status(s3.port, {}) == ("trading status unavailable (HTTP 401)", [])


# --------------------------------------------------------------------------- installation checks
def test_a_missing_project_or_virtual_environment_is_explained(tmp_path):
    with pytest.raises(ql.LaunchError, match="QuantPulse was not found"):
        ql.check_installation(tmp_path)
    for rel in ("pyproject.toml", "src/quantpulse/__init__.py", "frontend/app.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("")
    with pytest.raises(ql.LaunchError, match=r"virtual environment was not found(.|\n)*py -3.11 -m venv"):
        ql.check_installation(tmp_path)


def test_the_real_project_passes_the_installation_check():
    assert ql.check_installation(ROOT) == ql.venv_python(ROOT)


def test_log_tails_drop_traceback_markers_and_secrets(tmp_path):
    ql.REDACTOR.secrets = ["tok-secret-123"]
    try:
        log = tmp_path / "api.log"
        log.write_text("line 1\n  x = f()\n      ^^^^^\nValueError: token tok-secret-123 refused\n")
        assert ql.tail(log) == "line 1\n  x = f()\nValueError: token [redacted] refused"
    finally:
        ql.REDACTOR.secrets = []


# --------------------------------------------------------------------------- safety audit of the source
def test_the_launcher_can_only_read():
    """The launcher has no way to trade: every HTTP request it can make is a GET, and it never names an
    endpoint that runs a cycle, sends a test order or touches orders."""
    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "method":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value == "GET"
                assert kw.arg != "data", "a request body would mean a non-GET request"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.lower()
            for forbidden in (
                "/trading/run",
                "test-order",
                "/v2/orders",
                "kill-switch",
                "cancel-all",
                "close-all",
            ):
                assert forbidden not in text, f"the launcher mentions {forbidden!r}"
    assert "urlopen(" not in source.replace("_OPENER.open(", "")  # one opener, used by http_get only
    assert "QP_TRADING_SCHEDULER_ENABLED=true" not in source.replace(
        "(QP_TRADING_SCHEDULER_ENABLED=true)", ""
    )


def test_the_launcher_adds_no_trading_setting_to_any_environment():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'env["QP_TRADING' not in source and 'env["QP_ALPACA' not in source
    assert "QP_TRADING_KILL_SWITCH" not in source and "QP_ALPACA_TRADING_ENABLED" not in source


def test_a_reused_pid_is_never_mistaken_for_our_server():
    import os

    me = {"pid": os.getpid(), "module": "pytest", "created": ql.process_created(os.getpid())}
    python = Path(sys.executable)
    if os.name != "nt":  # on Windows the image must be the venv's python.exe, which pytest may not be
        assert ql.is_ours(me, python)
    assert not ql.is_ours({**me, "created": (me["created"] or 0) + 1}, python)  # same pid, another process
    assert not ql.is_ours({"pid": 2**22 + 12345, "module": "pytest"}, python)  # no such process
    assert not ql.is_ours(None, python)
