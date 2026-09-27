"""QuantPulse Terminal desktop launcher.

Double-clicking the **QuantPulse Terminal** desktop shortcut runs ``launch.bat``, which runs this script with
the project's own virtual environment (``.venv``). It:

1. checks the project folder and its virtual environment;
2. starts the FastAPI backend (``uvicorn --factory quantpulse.api.app:app_factory``) and the Streamlit UI
   (``streamlit run frontend/app.py``) from that environment, in hidden windows, both bound to
   ``127.0.0.1`` only — unless they are already running (a second double-click never starts a second
   copy);
3. waits until ``/health`` and Streamlit's health check answer, then opens the UI in the default browser;
4. reads the paper-trading status (``GET /api/v1/trading/status``) and records a one-line summary.

**It never trades.** It never runs a strategy cycle, never sends, cancels or changes an order and never
changes a setting: the only requests it makes are GETs to ``/health``, ``/openapi.json``,
``/api/v1/trading/status`` and Streamlit's ``/_stcore/health``. The API and the UI read their settings
from ``.env`` exactly as when started by hand (the launcher adds nothing trading-related to their
environment; the UI is only given the API address and the ``QP_API_TOKEN`` from ``.env`` so its requests
are authorised). Secrets are never logged: every line of ``logs/launcher.log`` is scrubbed of them.

Commands (``launch.bat``, ``trading-control.bat`` and ``stop.bat`` call these)::

    quantpulse_launcher.py start [--page trading] [--no-browser] [--no-splash] [--no-dialogs]
    quantpulse_launcher.py stop
    quantpulse_launcher.py status

Standard library only. Windows first (hidden consoles, message boxes, a start-up window, graceful Ctrl+C
shutdown); it also runs on macOS and Linux, which is how it is tested.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import logging
import logging.handlers
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# logs and the launcher's state; QP_LAUNCHER_HOME moves them (the tests use it, never a real session's files)
LOG_DIR = Path(os.environ.get("QP_LAUNCHER_HOME") or ROOT / "logs").resolve()
STATE_FILE = LOG_DIR / "launcher-state.json"
LOCK_FILE = LOG_DIR / "launcher.lock"
ICON = ROOT / "assets" / "quantpulse.ico"
APP_TITLE = "QuantPulse Terminal"
HOST = "127.0.0.1"
API_PORT = 8000
UI_PORT = 8501
API_START_TIMEOUT = 180.0  # the first start creates / migrates the database
UI_START_TIMEOUT = 120.0
STOP_TIMEOUT = 20.0
STATUS_TIMEOUT = 25.0
WINDOWS = os.name == "nt"
SECRET_WORDS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
CHILD_LOG_MAX_BYTES = 5_000_000
PAGES = {"trading": "trading"}
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
TRADING_STATUS_PATH = "/api/v1/trading/status"

log = logging.getLogger("quantpulse.launcher")
Progress = Callable[[str], None]


class LaunchError(Exception):
    """Something the user must fix; the message says what and how."""


# ----------------------------------------------------------------------------- settings (.env)
def env_file(root: Path = ROOT, environ: dict[str, str] | None = None) -> Path | None:
    """The .env the application reads (same order as quantpulse.config: QP_ENV_FILE, then the project)."""
    env = os.environ if environ is None else environ
    explicit = env.get("QP_ENV_FILE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    candidate = root / ".env"
    return candidate if candidate.is_file() else None


def read_env_file(path: Path | None) -> dict[str, str]:
    """``KEY=value`` pairs (comments, ``export``, quotes and a UTF-8 byte-order mark handled)."""
    values: dict[str, str] = {}
    if path is None or not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key.strip().upper()] = value
    return values


def effective_settings(root: Path = ROOT, environ: dict[str, str] | None = None) -> dict[str, str]:
    """What the application sees: the .env values, overridden by the process environment."""
    env = dict(os.environ if environ is None else environ)
    merged = read_env_file(env_file(root, env))
    merged.update({k.upper(): v for k, v in env.items()})
    return merged


def secret_values(settings: dict[str, str]) -> set[str]:
    return {
        v
        for k, v in settings.items()
        if any(w in k for w in SECRET_WORDS) and len(v) >= 6 and not v.isdigit()
    }


class Redactor(logging.Filter):
    """Removes every known secret value from log records (and from dialog text)."""

    def __init__(self, secrets: set[str] | None = None) -> None:
        super().__init__()
        self.secrets = sorted(secrets or set(), key=len, reverse=True)

    def redact(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "[redacted]")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = self.redact(record.getMessage()), None
        return True


REDACTOR = Redactor()


def setup_logging(settings: dict[str, str]) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    REDACTOR.secrets = sorted(secret_values(settings), key=len, reverse=True)
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "launcher.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    handler.addFilter(REDACTOR)
    log.addHandler(handler)
    if sys.stderr is not None:  # a console (not pythonw)
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        console.addFilter(REDACTOR)
        log.addHandler(console)


# ----------------------------------------------------------------------------- dialogs
class Dialogs:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def show(self, text: str, kind: str = "info") -> None:
        text = REDACTOR.redact(text)
        if not self.enabled:
            return
        if WINDOWS:
            import ctypes

            icon = {"info": 0x40, "warning": 0x30, "error": 0x10}[kind]
            ctypes.windll.user32.MessageBoxW(
                None, text, APP_TITLE, icon | 0x10000 | 0x40000
            )  # foreground, topmost
        elif sys.stderr is not None:
            print(f"[{kind}] {text}", file=sys.stderr)


# ----------------------------------------------------------------------------- local HTTP (GET only)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy


def http_get(
    url: str, headers: dict[str, str] | None = None, timeout: float = 2.0
) -> tuple[int, bytes] | None:
    """GET ``url``; ``(status, body)``, or ``None`` when nothing answered. (The launcher has no other verb.)"""
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read(500_000)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(2_000)
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return None


def port_open(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def probe_api(port: int) -> str:
    """``quantpulse`` (the QuantPulse API answers), ``other`` (something else holds the port) or ``free``."""
    got = http_get(f"http://{HOST}:{port}/health")
    if got is None:
        return "other" if port_open(port) else "free"
    status, body = got
    try:
        health = json.loads(body)
    except ValueError:
        return "other"
    if status == 200 and isinstance(health, dict) and health.get("status") == "ok":
        doc = http_get(f"http://{HOST}:{port}/openapi.json", timeout=10)
        try:
            title = json.loads(doc[1]).get("info", {}).get("title", "") if doc and doc[0] == 200 else ""
        except (ValueError, AttributeError):
            title = ""
        if "QuantPulse" in title:
            return "quantpulse"
    return "other"


def probe_ui(port: int) -> str:
    """``streamlit`` (a Streamlit app answers), ``other`` or ``free``."""
    got = http_get(f"http://{HOST}:{port}/_stcore/health")
    if got is None:
        return "other" if port_open(port) else "free"
    return "streamlit" if got[0] == 200 and got[1].strip() == b"ok" else "other"


# ----------------------------------------------------------------------------- processes
def venv_python(root: Path = ROOT) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")


def api_command(python: Path, port: int) -> list[str]:
    return [
        str(python),
        "-m",
        "uvicorn",
        "--factory",
        "quantpulse.api.app:app_factory",
        "--host",
        HOST,
        "--port",
        str(port),
    ]


def ui_command(python: Path, port: int, root: Path = ROOT) -> list[str]:
    return [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(root / "frontend" / "app.py"),
        "--server.address",
        HOST,  # this machine only: the UI can act on the paper account with the API token
        "--server.port",
        str(port),
        "--server.headless",
        "true",  # the launcher opens the browser itself (once)
        "--browser.gatherUsageStats",
        "false",
    ]


def child_env(kind: str, settings: dict[str, str], api_port: int) -> dict[str, str]:
    """Environment for a server: the launcher's own, unchanged — plus, for the UI only, the API address and
    the API token from .env so the UI's requests are authorised. Nothing trading-related is ever added."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if kind == "ui":
        env["QP_API_URL"] = f"http://{HOST}:{api_port}"
        token = settings.get("QP_API_TOKEN", "").strip()
        if token and not env.get("QP_API_TOKEN"):
            env["QP_API_TOKEN"] = token
    return env


def _rotate(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > CHILD_LOG_MAX_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:  # still open elsewhere: keep appending
        pass


def spawn(name: str, command: list[str], env: dict[str, str], log_path: Path) -> int:
    """Start a server in the background (no window on Windows) with its output in ``log_path``."""
    _rotate(log_path)
    with open(log_path, "ab") as out:
        out.write(
            f"\n===== {name} started by the launcher {datetime.now():%Y-%m-%d %H:%M:%S} =====\n".encode()
        )
        out.flush()
        kwargs: dict[str, Any] = {
            "cwd": str(ROOT),  # the same working directory as `make api` / `streamlit run` from the repo
            "env": env,
            "stdin": subprocess.DEVNULL,
            "stdout": out,
            "stderr": subprocess.STDOUT,
        }
        if WINDOWS:
            # its own console without a window: nothing appears on screen (Windows Terminal is never
            # involved), and "stop" can still deliver Ctrl+C to that console for a clean shutdown
            kwargs.update(creationflags=CREATE_NO_WINDOW)
        else:
            kwargs.update(start_new_session=True)
        proc = subprocess.Popen(command, **kwargs)
    log.info("started %s (pid %s): %s", name, proc.pid, " ".join(command[1:]))
    return proc.pid


def _win_kernel32() -> Any:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    return k32


def process_image(pid: int) -> str | None:
    """The executable of a live process (``None`` if it is gone)."""
    if pid <= 0:
        return None
    if WINDOWS:
        import ctypes
        from ctypes import wintypes

        k32 = _win_kernel32()
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
                return None
            size = wintypes.DWORD(32768)
            buf = ctypes.create_unicode_buffer(size.value)
            if not k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return ""
            return buf.value
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return None
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        with contextlib.suppress(OSError):
            if (proc / "stat").read_text().split(")")[-1].split()[0] == "Z":  # a zombie has exited
                return None
            return (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return ""


def process_created(pid: int) -> int | None:
    """When the process started (an opaque number; with the pid it identifies one process for good)."""
    if pid <= 0:
        return None
    if WINDOWS:
        import ctypes
        from ctypes import wintypes

        k32 = _win_kernel32()
        handle = k32.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            k32.CloseHandle(handle)
    with contextlib.suppress(OSError, IndexError, ValueError):
        return int(Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[19])
    return None


def is_ours(entry: dict[str, Any] | None, python: Path) -> bool:
    """Whether a recorded process is still the server the launcher started (not a reused pid)."""
    if not entry:
        return False
    pid = int(entry.get("pid", 0))
    image = process_image(pid)
    if image is None:
        return False
    created = entry.get("created")
    if created is not None and process_created(pid) not in (None, created):
        return False  # the pid now belongs to another process
    if not image:  # alive, but its path cannot be read (e.g. macOS)
        return True
    if WINDOWS:
        return os.path.normcase(os.path.abspath(image)) == os.path.normcase(os.path.abspath(python))
    return str(entry.get("module", "")) in image


def send_ctrl_c(pid: int) -> None:
    """Windows: deliver Ctrl+C to the server's own (hidden) console, from a short-lived helper process
    that has no console of its own — so the launcher's console, if any, is never touched."""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_ctrl_c", str(pid)],
        creationflags=DETACHED_PROCESS if WINDOWS else 0,
        timeout=15,
        check=False,
    )


def _ctrl_c_helper(pid: int) -> int:
    k32 = _win_kernel32()
    k32.FreeConsole()
    if not k32.AttachConsole(pid):
        return 1
    k32.SetConsoleCtrlHandler(None, True)  # ignore it ourselves
    k32.GenerateConsoleCtrlEvent(0, 0)  # CTRL_C_EVENT to every process on that console
    time.sleep(0.5)
    k32.FreeConsole()
    return 0


def stop_process(pid: int, name: str, timeout: float = STOP_TIMEOUT) -> str:
    """Ask the server to shut down (Ctrl+C / SIGINT: it finishes requests and closes the database), then
    force it after ``timeout`` seconds. ``stopped``, ``forced`` or ``gone``."""
    if process_image(pid) is None:
        return "gone"
    log.info("stopping %s (pid %s)", name, pid)
    try:
        if WINDOWS:
            send_ctrl_c(pid)
        else:
            os.killpg(pid, signal.SIGINT)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not signal %s: %s", name, exc)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process_image(pid) is None:
            return "stopped"
        time.sleep(0.25)
    log.warning("%s did not stop within %.0fs: forcing it", name, timeout)
    if WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    else:
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
    for _ in range(40):
        if process_image(pid) is None:
            break
        time.sleep(0.25)
    return "forced"


def load_state() -> dict[str, Any]:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any]) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def tail(path: Path, lines: int = 6) -> str:
    """The last meaningful lines of a server log (traceback caret markers dropped), secrets removed."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    useful = [t.rstrip() for t in text if t.strip() and set(t.strip()) - set("^~ ")]
    return REDACTOR.redact("\n".join(useful[-lines:]))


# ----------------------------------------------------------------------------- single instance
@contextlib.contextmanager
def single_instance() -> Iterator[bool]:
    """True for the only launcher currently starting QuantPulse; False if another one already is."""
    if WINDOWS:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = wintypes.HANDLE
        name = "Local\\QuantPulseTerminalLauncher-" + format(zlib.crc32(str(LOG_DIR).lower().encode()), "08x")
        handle = k32.CreateMutexW(None, False, name)
        first = ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS
        try:
            yield first
        finally:
            if handle:
                k32.CloseHandle(handle)
        return
    import fcntl

    LOG_DIR.mkdir(exist_ok=True)
    with open(LOCK_FILE, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------- start
@dataclass
class Options:
    api_port: int = API_PORT
    ui_port: int = UI_PORT
    page: str | None = None
    browser: bool = True
    splash: bool = True
    dialogs: bool = True


@dataclass
class StartResult:
    url: str
    started: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    trading: str = ""
    warnings: list[str] = field(default_factory=list)


def check_installation(root: Path = ROOT) -> Path:
    required = ("pyproject.toml", "src/quantpulse/__init__.py", "frontend/app.py")
    missing = [p for p in required if not (root / p).exists()]
    if missing:
        raise LaunchError(
            f"QuantPulse was not found at:\n{root}\n\nMissing: {', '.join(missing)}.\n"
            "Keep the launcher folder inside the QuantPulse project (…\\Finance\\launcher)."
        )
    python = venv_python(root)
    if not python.exists():
        raise LaunchError(
            f"The QuantPulse virtual environment was not found:\n{root / '.venv'}\n\n"
            "Create it once in PowerShell:\n"
            f"  cd {root}\n"
            "  py -3.11 -m venv .venv\n"
            '  .venv\\Scripts\\pip install -e ".[frontend]" -c constraints.txt'
        )
    check = subprocess.run(
        [str(python), "-c", "import quantpulse, uvicorn, streamlit"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=180,
        creationflags=CREATE_NO_WINDOW if WINDOWS else 0,
        check=False,
    )
    if check.returncode != 0:
        last = (check.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise LaunchError(
            f"The virtual environment is missing packages ({last}).\n\nInstall them once in PowerShell:\n"
            f"  cd {root}\n"
            '  .venv\\Scripts\\pip install -e ".[frontend]" -c constraints.txt'
        )
    return python


def wait_until(
    name: str, ready: Callable[[], bool], pid: int | None, log_path: Path, timeout: float, progress: Progress
) -> None:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    while time.monotonic() < deadline:
        if ready():
            log.info("%s is ready (%.1fs)", name, time.monotonic() - started)
            return
        if pid is not None and process_image(pid) is None:
            raise LaunchError(
                f"The {name} stopped while starting:\n\n{tail(log_path)}\n\nFull log: {log_path}"
            )
        progress(f"Waiting for the {name}… {time.monotonic() - started:.0f}s")
        time.sleep(0.5)
    raise LaunchError(
        f"The {name} did not answer within {timeout:.0f} seconds:\n\n{tail(log_path)}\n\nFull log: {log_path}"
    )


def trading_status(api_port: int, settings: dict[str, str]) -> tuple[str, list[str]]:
    """A read-only summary of the paper-trading switches (GET /api/v1/trading/status) and warnings."""
    token = settings.get("QP_API_TOKEN", "").strip()
    got = http_get(
        f"http://{HOST}:{api_port}{TRADING_STATUS_PATH}",
        headers={"X-API-Key": token} if token else None,
        timeout=STATUS_TIMEOUT,
    )
    if got is None or got[0] != 200:
        code = "no answer" if got is None else f"HTTP {got[0]}"
        return f"trading status unavailable ({code})", []
    try:
        st = json.loads(got[1])
    except ValueError:
        return "trading status unreadable", []

    def yes(v: Any) -> str:
        return "on" if v else "off"

    summary = (
        f"Alpaca {'PAPER' if st.get('paper') else 'NOT PAPER'} account · mode {st.get('mode')} · "
        f"trading enabled {yes(st.get('trading_enabled'))} · dry run {yes(st.get('dry_run'))} · "
        f"kill switch {yes((st.get('kill_switch') or {}).get('active'))} · "
        f"scheduler {yes(st.get('scheduler_enabled'))} · keys {'set' if st.get('broker_configured') else 'missing'}"
    )
    warnings: list[str] = []
    if st.get("paper") is not True:
        warnings.append("The API did not report a PAPER account. Do not trade until this is explained.")
    if st.get("scheduler_enabled"):
        warnings.append(
            "The trading scheduler is ON (QP_TRADING_SCHEDULER_ENABLED=true): during market hours QuantPulse "
            "runs strategy cycles on its own"
            + (
                " and may send paper orders."
                if st.get("mode") == "paper"
                else " (dry runs while the mode is dry run)."
            )
            + "\n\nThe launcher did not change this. To turn it off, set QP_TRADING_SCHEDULER_ENABLED=false in "
            ".env and start QuantPulse again."
        )
    return summary, warnings


def start_services(opts: Options, progress: Progress) -> StartResult:
    settings = effective_settings()
    url = f"http://{HOST}:{opts.ui_port}/" + PAGES.get(opts.page or "", "")
    result = StartResult(url=url)
    state = load_state()
    python = venv_python()

    progress("Checking what is already running…")
    api = probe_api(opts.api_port)
    ui = probe_ui(opts.ui_port)
    api_ours = is_ours(state.get("api"), python) and state["api"].get("port") == opts.api_port
    ui_ours = is_ours(state.get("ui"), python) and state["ui"].get("port") == opts.ui_port
    if api == "other" and not api_ours:
        raise LaunchError(
            f"Port {opts.api_port} is used by another program, so the QuantPulse API cannot start.\n\n"
            "Close that program (or find it with: netstat -ano | findstr :%d) and try again." % opts.api_port
        )
    if ui == "other" and not ui_ours:
        raise LaunchError(
            f"Port {opts.ui_port} is used by another program, so the QuantPulse UI cannot start.\n\n"
            "Close that program (or find it with: netstat -ano | findstr :%d) and try again." % opts.ui_port
        )

    started_pids: dict[str, int] = {}
    try:
        if (api == "free" and not api_ours) or (ui == "free" and not ui_ours):
            progress("Checking the QuantPulse installation…")
            python = check_installation()
        # --- API
        if api == "quantpulse" or api_ours:
            result.reused.append("api")
            log.info("the API is already running on port %s: not starting another", opts.api_port)
            api_pid = int(state["api"]["pid"]) if api_ours else None
        else:
            progress("Starting the QuantPulse API…")
            api_pid = spawn(
                "api",
                api_command(python, opts.api_port),
                child_env("api", settings, opts.api_port),
                LOG_DIR / "api.log",
            )
            started_pids["api"] = api_pid
            state["api"] = {
                "pid": api_pid,
                "created": process_created(api_pid),
                "port": opts.api_port,
                "module": "uvicorn",
                "started": _now(),
            }
            save_state(state)
            result.started.append("api")
        wait_until(
            "QuantPulse API",
            lambda: probe_api(opts.api_port) == "quantpulse",
            api_pid,
            LOG_DIR / "api.log",
            API_START_TIMEOUT,
            progress,
        )
        # --- UI
        if ui == "streamlit" or ui_ours:
            result.reused.append("ui")
            log.info("the UI is already running on port %s: not starting another", opts.ui_port)
            ui_pid = int(state["ui"]["pid"]) if ui_ours else None
        else:
            progress("Starting the QuantPulse UI…")
            ui_pid = spawn(
                "ui",
                ui_command(python, opts.ui_port),
                child_env("ui", settings, opts.api_port),
                LOG_DIR / "ui.log",
            )
            started_pids["ui"] = ui_pid
            state["ui"] = {
                "pid": ui_pid,
                "created": process_created(ui_pid),
                "port": opts.ui_port,
                "module": "streamlit",
                "started": _now(),
            }
            save_state(state)
            result.started.append("ui")
        wait_until(
            "QuantPulse UI",
            lambda: probe_ui(opts.ui_port) == "streamlit",
            ui_pid,
            LOG_DIR / "ui.log",
            UI_START_TIMEOUT,
            progress,
        )
    except BaseException:
        for name, pid in started_pids.items():  # leave nothing half-started behind
            stop_process(pid, name, timeout=10)
            state.pop(name, None)
        save_state(state)
        raise

    progress("Opening QuantPulse…")
    return result


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- start-up window
def run_with_splash(work: Callable[[Progress], StartResult], enabled: bool) -> StartResult:
    """Run ``work`` while a small start-up window shows its progress. Without Tk (or if the window cannot
    be built for any reason) ``work`` simply runs without it: the window is never a reason to fail."""
    if not enabled:
        return work(lambda s: None)
    root: Any = None
    try:
        import tkinter as tk

        root = tk.Tk()
        status = _build_splash(root)
    except Exception as exc:
        log.info("no start-up window (%s)", type(exc).__name__)
        with contextlib.suppress(Exception):
            if root is not None:
                root.destroy()
        return work(lambda s: None)

    messages: queue.Queue[str] = queue.Queue()
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["result"] = work(messages.put)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            messages.put("__done__")

    def poll() -> None:
        try:
            while True:
                msg = messages.get_nowait()
                if msg == "__done__":
                    root.destroy()
                    return
                with contextlib.suppress(Exception):
                    status.configure(text=msg)
        except queue.Empty:
            root.after(100, poll)

    threading.Thread(target=worker, daemon=True).start()
    root.after(100, poll)
    root.mainloop()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _build_splash(root: Any) -> Any:
    """The start-up window: title, PAPER notice, current step and a progress bar. Returns the step label."""
    import tkinter as tk
    from tkinter import ttk

    bg, fg, accent, amber, muted = "#0b1f3a", "#e8eef7", "#2ee6a6", "#ffb020", "#8aa2c2"
    root.title(APP_TITLE)
    root.overrideredirect(True)
    root.configure(bg=bg)
    root.attributes("-topmost", True)
    with contextlib.suppress(Exception):
        root.iconbitmap(str(ICON))
    w, h = 460, 190
    root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")
    frame = tk.Frame(root, bg=bg, highlightthickness=1, highlightbackground="#23466f")
    frame.pack(fill="both", expand=True)
    title_font = ("Segoe UI Semibold", 17) if WINDOWS else ("Helvetica", 17, "bold")
    body_font = ("Segoe UI", 10) if WINDOWS else ("Helvetica", 10)
    small_font = ("Segoe UI", 8) if WINDOWS else ("Helvetica", 8)
    tk.Label(frame, text=APP_TITLE, bg=bg, fg=fg, font=title_font).pack(anchor="w", padx=22, pady=(20, 0))
    tk.Label(
        frame, text="ALPACA PAPER TRADING · SIMULATED MONEY", bg=bg, fg=amber, font=(*small_font, "bold")
    ).pack(anchor="w", padx=22)
    status = tk.Label(frame, text="Starting…", bg=bg, fg=fg, font=body_font)
    status.pack(anchor="w", padx=22, pady=(18, 6))
    style = ttk.Style(root)
    with contextlib.suppress(Exception):
        style.theme_use("clam")
    with contextlib.suppress(Exception):
        style.configure(
            "QP.Horizontal.TProgressbar",
            troughcolor="#16345c",
            background=accent,
            bordercolor=bg,
            lightcolor=accent,
            darkcolor=accent,
        )
    bar = ttk.Progressbar(frame, mode="indeterminate", length=w - 44, style="QP.Horizontal.TProgressbar")
    bar.pack(anchor="w", padx=22)
    bar.start(12)
    tk.Label(frame, text="Starting QuantPulse never places a trade.", bg=bg, fg=muted, font=small_font).pack(
        anchor="w", padx=22, pady=(12, 0)
    )
    root.update_idletasks()
    return status


# ----------------------------------------------------------------------------- commands
def cmd_start(opts: Options) -> int:
    dialogs = Dialogs(opts.dialogs)
    log.info("----- start (%s) -----", ROOT)
    with single_instance() as first:
        if not first:
            log.info("another launcher is already starting QuantPulse")
            dialogs.show("QuantPulse is already starting. Your browser will open when it is ready.", "info")
            return 0
        result = run_with_splash(lambda progress: start_services(opts, progress), opts.splash)
    what = ", ".join(result.started) or "nothing (already running)"
    log.info("started: %s; reused: %s; UI at %s", what, ", ".join(result.reused) or "none", result.url)
    if opts.browser:
        opened = False
        with contextlib.suppress(Exception):
            opened = webbrowser.open(result.url, new=2)
        log.info("browser %s %s", "opened at" if opened else "could not be opened for", result.url)
        if not opened:
            dialogs.show(f"QuantPulse is running. Open this address in your browser:\n{result.url}", "info")
    # after the browser is open, so a slow answer never delays the UI (a read-only GET)
    result.trading, result.warnings = trading_status(opts.api_port, effective_settings())
    log.info("trading status (read-only): %s", result.trading)
    for warning in result.warnings:
        log.warning(warning.replace("\n", " "))
        dialogs.show(warning, "warning")
    return 0


def cmd_stop(opts: Options) -> int:
    dialogs = Dialogs(opts.dialogs)
    log.info("----- stop -----")
    state = load_state()
    python = venv_python()
    stopped: list[str] = []
    notes: list[str] = []
    for name, label, port, probe, running in (
        ("ui", "UI", opts.ui_port, probe_ui, "streamlit"),
        ("api", "API", opts.api_port, probe_api, "quantpulse"),
    ):
        entry = state.get(name)
        if is_ours(entry, python):
            assert entry is not None
            how = stop_process(int(entry["pid"]), name)
            stopped.append(f"{label} ({how})")
            state.pop(name, None)
        else:
            state.pop(name, None)
            if probe(entry.get("port", port) if entry else port) == running:
                notes.append(
                    f"The {label} on port {port} was not started by this launcher (a Command Prompt window?): "
                    "close that window to stop it."
                )
    save_state(state)
    if stopped:
        text = "QuantPulse stopped: " + ", ".join(stopped) + "."
    else:
        text = "QuantPulse was not running (nothing started by the launcher)."
    log.info(text)
    for n in notes:
        log.info(n)
    dialogs.show("\n\n".join([text, *notes]), "info")
    return 0


def cmd_status(opts: Options) -> int:
    settings = effective_settings()
    state = load_state()
    python = venv_python()
    lines = []
    for name, label, port, probe, running in (
        ("api", "API", opts.api_port, probe_api, "quantpulse"),
        ("ui", "UI", opts.ui_port, probe_ui, "streamlit"),
    ):
        up = probe(port) == running
        owner = (
            f"pid {state[name]['pid']}, started by the launcher"
            if is_ours(state.get(name), python)
            else "not started by the launcher"
        )
        lines.append(
            f"{label}: {'running' if up else 'not running'} on port {port}" + (f" ({owner})" if up else "")
        )
    if probe_api(opts.api_port) == "quantpulse":
        lines.append(trading_status(opts.api_port, settings)[0])
    text = "\n".join(lines)
    log.info("status: %s", text.replace("\n", " | "))
    if sys.stdout is not None:
        print(text)
    else:
        Dialogs(opts.dialogs).show(text, "info")
    return 0


def parse_args(argv: list[str] | None) -> tuple[str, Options, list[str]]:
    parser = argparse.ArgumentParser(
        prog="quantpulse_launcher", description="Start or stop QuantPulse Terminal."
    )
    parser.add_argument("command", nargs="?", default="start", choices=["start", "stop", "status", "_ctrl_c"])
    parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--page", choices=sorted(PAGES), help="open this page (e.g. trading)")
    parser.add_argument("--api-port", type=int, default=API_PORT)
    parser.add_argument("--ui-port", type=int, default=UI_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--no-splash", action="store_true", help="no start-up window")
    parser.add_argument("--no-dialogs", action="store_true", help="no message boxes (log only)")
    args = parser.parse_args(argv)
    opts = Options(
        api_port=args.api_port,
        ui_port=args.ui_port,
        page=args.page,
        browser=not args.no_browser,
        splash=not args.no_splash,
        dialogs=not args.no_dialogs,
    )
    return args.command, opts, args.extra


def _windows_setup() -> None:
    import ctypes

    with contextlib.suppress(Exception):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("QuantPulse.Terminal.Launcher")
    with contextlib.suppress(Exception):
        ctypes.windll.shcore.SetProcessDpiAwareness(1)


def main(argv: list[str] | None = None) -> int:
    command, opts, extra = parse_args(argv)
    if command == "_ctrl_c":
        return _ctrl_c_helper(int(extra[0]))
    setup_logging(effective_settings())
    if WINDOWS:
        _windows_setup()
    dialogs = Dialogs(opts.dialogs)
    try:
        if command == "stop":
            return cmd_stop(opts)
        if command == "status":
            return cmd_status(opts)
        return cmd_start(opts)
    except LaunchError as exc:
        log.error("%s", str(exc).replace("\n", " | "))
        dialogs.show(f"{exc}\n\nDetails: {LOG_DIR / 'launcher.log'}", "error")
        return 1
    except Exception as exc:
        log.exception("unexpected launcher error")
        dialogs.show(
            f"QuantPulse could not start: {type(exc).__name__}: {exc}\n\nDetails: {LOG_DIR / 'launcher.log'}",
            "error",
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
