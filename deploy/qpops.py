#!/usr/bin/env python3
"""qpops — QuantPulse's server operations on one Linux VM (built for Oracle Cloud Always Free; any host works).

Standard library only (the host's python3). ``./qp`` calls it and the systemd timers in deploy/systemd/ run it:

  watchdog       every minute: is the Brain supervisor alive — not just the process? A stalled supervisor (or an
                 API that stopped answering) gets a graceful restart of the api container, at most 3 per 6 hours;
                 the watchdog's only request to QuantPulse is one GET, so it can never cause an order
  auto-update    every 10 minutes: deploy the newest commit of the followed branch only when GitHub CI passed
                 every required job (ARM64 included) on exactly that commit; build and check it before
                 switching, verify after, roll back automatically when the new version does not come up
  deploy         the same now, inside market hours too (./qp update); CI must still have passed
  ci-gate [SHA]  what the gate decides about SHA (default: the branch's newest commit)
  backup         a verified database dump; the nightly one is uploaded to Object Storage through a write-only
                 link; an alert when anything fails
  restore-test   restore the newest dump into a scratch database and check it (weekly); an alert when it fails
  status         the whole server at a glance (./qp status)
  sample         record CPU and memory use (for Oracle's idle check); the watchdog does it every minute

Host-only settings and secrets live in deploy/ops.env (chmod 600, ignored by git, never given to a container):
see deploy/ops.env.example. Nothing here prints a secret.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------------------------- constants
WORKFLOW = ".github/workflows/ci.yml"
# every job of that workflow; a commit is deployed only when each one passed on exactly that commit
REQUIRED_JOBS = ("test (3.11)", "test (3.12)", "postgres", "docker", "docker (arm64)")
NEW_YORK = ZoneInfo("America/New_York")
QUIET_FROM, QUIET_TO = dtime(8, 0), dtime(17, 30)  # weekdays, New York: no automatic deploy in between

API = "http://127.0.0.1:8000"
WATCHDOG_PATH = "/api/v1/system/watchdog"
# the only QuantPulse endpoints this program ever calls, all GET and read-only
API_READS = frozenset({"/health", WATCHDOG_PATH, "/api/v1/system/health", "/api/v1/system/health?fresh=true"})

API_START_GRACE = timedelta(minutes=5)  # a fresh api container is left alone this long
UNREACHABLE_CHECKS = 3  # consecutive minutes without an answer before a restart
STALLED_CHECKS = 2  # consecutive "stalled" verdicts before a restart
MAX_RESTARTS, RESTART_WINDOW = 3, timedelta(hours=6)
PRE_SWITCH_TRIES = 3  # a commit whose build or checks failed this often is not tried again automatically
VERIFY_FOR = timedelta(minutes=10)  # a new version must be ticking within this long after the switch
INTERRUPTED_AFTER = timedelta(minutes=45)  # a deploy still "switching" this long ago was cut off (a reboot)
HEALTHY_AFTER_DEPLOY = {"ok", "paused", "not_applicable"}
KEEP_DUMPS_DAYS, KEEP_MANUAL_DAYS, KEEP_AT_LEAST = 14, 60, 3
RESTORE_DB = "qp_restore_test"
SAMPLES_KEEP = timedelta(days=7)
IDLE_THRESHOLD = 20.0  # percent: Oracle's idle rule for Always Free instances (CPU p95, network, memory)
IDLE_MIN_SAMPLES = 24 * 60  # a day of minutes before the idle warning speaks
ALERT_REPEAT = timedelta(hours=6)
SHA = re.compile(r"^[0-9a-f]{40}$")
# Alpaca's live-money trading API and its broker API — the same rule as the API's own preflight
LIVE_ENDPOINT = re.compile(r"(?<![\w.-])(api|broker-api)\.alpaca\.markets", re.IGNORECASE)
HOST_PREFIXES = ("QP_", "APCA_", "ALPACA_")


def utcnow() -> datetime:
    return datetime.now(UTC)


def log(message: str) -> None:
    print(f"{utcnow():%Y-%m-%d %H:%M:%S} UTC  {message}", flush=True)


def redact(text: str) -> str:
    """Remove anything secret-shaped from a message: pre-authenticated request paths, tokens in URLs."""
    text = re.sub(r"/p/[^/\s]+/", "/p/<redacted>/", text)
    text = re.sub(r"(https?://)[^\s/@]+@", r"\1<redacted>@", text)
    return re.sub(r"(ntfy\.sh/|hc-ping\.com/)[^\s\"']+", r"\1<redacted>", text)


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines (``#`` comments, optional quotes); a missing file is empty."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def in_quiet_hours(now: datetime) -> bool:
    """US market hours, widened (the Brain's pre-market preparation and the close): weekdays 08:00–17:30 New
    York. No holiday calendar here, so a holiday counts as a trading day — the safe side."""
    local = now.astimezone(NEW_YORK)
    return local.weekday() < 5 and QUIET_FROM <= local.time() < QUIET_TO


@dataclass
class Gate:
    ok: bool
    state: str  # passed | pending | failed | unknown
    reason: str
    run_url: str | None = None
    checked_at: str = ""


class OpsError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------- the host
class Host:
    """Runs commands on this server (tests replace it)."""

    def run(
        self, args: list[str], *, env: dict[str, str] | None = None, timeout: float = 600, check: bool = True,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:  # fmt: skip
        merged = {**os.environ, **(env or {})}
        try:
            proc = subprocess.run(args, env=merged, capture_output=True, text=True, timeout=timeout,
                                  input=input_text, check=False)  # fmt: skip
        except subprocess.TimeoutExpired:  # a command that hangs is a failure, never a crash of the tool
            proc = subprocess.CompletedProcess(args, 124, "", f"timed out after {timeout:.0f} s")
        if check and proc.returncode != 0:
            tail = redact((proc.stderr or proc.stdout or "").strip()[-600:])
            raise OpsError(f"{' '.join(args[:4])} … failed ({proc.returncode}): {tail}")
        return proc

    def pipe(self, first: list[str], second: list[str], *, timeout: float = 3600) -> None:
        """``first | second`` (a ``git archive`` into a ``docker build``)."""
        with subprocess.Popen(first, stdout=subprocess.PIPE) as producer:
            consumer = subprocess.run(second, stdin=producer.stdout, capture_output=True, text=True,
                                      timeout=timeout, check=False)  # fmt: skip
            if producer.stdout is not None:
                producer.stdout.close()
            producer.wait(timeout=60)
        if producer.returncode != 0 or consumer.returncode != 0:
            tail = redact((consumer.stderr or consumer.stdout or "").strip()[-800:])
            raise OpsError(f"the image build failed: {tail}")


# --------------------------------------------------------------------------------------------- operations
class Ops:
    def __init__(
        self,
        deploy_dir: Path,
        host: Host | None = None,
        urlopen: Callable[..., Any] | None = None,
        now: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.dir = deploy_dir
        self.repo_dir = deploy_dir.parent
        self.state_dir = deploy_dir / "state"
        self.host = host or Host()
        self._urlopen = urlopen or urllib.request.urlopen
        self.now = now
        self.sleep = sleep
        self.app_env = read_env(deploy_dir / ".env")  # the containers' settings (the API token, alert URLs)
        self.cfg = read_env(deploy_dir / "ops.env")  # host-only settings and secrets

    # ------------------------------------------------------------------ small helpers
    def setting(self, name: str, default: str = "") -> str:
        return os.environ.get(name) or self.cfg.get(name) or default

    def flag(self, name: str, default: bool) -> bool:
        value = self.setting(name, "true" if default else "false").strip().lower()
        return value in ("1", "true", "yes", "on")

    def load(self, name: str) -> dict[str, Any]:
        path = self.state_dir / name
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return {}

    def save(self, name: str, data: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash leaves the old or the new file, never half of one

    @contextlib.contextmanager
    def lock(self, wait: bool = False) -> Iterator[bool]:
        """One deploy, backup or restore at a time (the watchdog stays away while one runs)."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / "ops.lock", "a+") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def dc(self, *args: str, env: dict[str, str] | None = None, timeout: float = 600,
           check: bool = True) -> subprocess.CompletedProcess[str]:  # fmt: skip
        return self.host.run(["docker", "compose", "-f", str(self.dir / "compose.yaml"), *args], env=env,
                             timeout=timeout, check=check)  # fmt: skip

    def git(self, *args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
        return self.host.run(["git", "-C", str(self.repo_dir), *args], env=env, check=check).stdout.strip()

    # ------------------------------------------------------------------ alerts
    def alert(self, title: str, message: str, severity: str = "warning", *, key: str | None = None) -> None:
        """ntfy and/or the webhook from deploy/.env (or QP_OPS_NTFY_URL); never raises. With ``key``, the same
        alert is not repeated within 6 hours."""
        log(f"ALERT [{severity}] {title}: {redact(message)}")
        if key is not None:
            sent = self.load("alerts.json")
            last = sent.get(key)
            if last and self.now() - datetime.fromisoformat(last) < ALERT_REPEAT:
                return
            sent[key] = self.now().isoformat()
            self.save("alerts.json", sent)
        ntfy = self.setting("QP_OPS_NTFY_URL") or self.app_env.get("QP_ALERT_NTFY_URL", "")
        hook = self.app_env.get("QP_ALERT_WEBHOOK_URL", "")
        body = f"{message}\n(server {socket.gethostname()})"
        if ntfy:
            prio = {"info": "default", "warning": "high", "critical": "urgent"}.get(severity, "high")
            headers = {"Title": f"QuantPulse server: {title}"[:200].encode("ascii", "replace").decode(),
                       "Priority": prio, "Tags": "warning" if severity != "info" else "information_source"}  # fmt: skip
            self._send(ntfy, body.encode(), headers)
        if hook:
            payload = json.dumps(
                {"text": f"QuantPulse server: {title}\n{body}", "content": f"{title}: {body}"}
            )
            self._send(hook, payload.encode(), {"Content-Type": "application/json"})

    def _send(self, url: str, data: bytes, headers: dict[str, str]) -> None:
        try:
            with self._urlopen(
                urllib.request.Request(url, data=data, headers=headers, method="POST"), timeout=15
            ):
                pass
        except Exception as exc:  # an alert that cannot be delivered never stops the work
            log(f"alert delivery failed ({type(exc).__name__})")

    def ping(self, url: str, ok: bool) -> None:
        """A dead man's switch (healthchecks.io): success, or ``/fail``."""
        if not url:
            return
        try:
            target = url.rstrip("/") + ("" if ok else "/fail")
            with self._urlopen(urllib.request.Request(target, method="GET"), timeout=15):
                pass
        except Exception as exc:
            log(f"heartbeat ping failed ({type(exc).__name__})")

    # ------------------------------------------------------------------ QuantPulse's API (read-only)
    def api_get(self, path: str, timeout: float = 20) -> tuple[int, Any]:
        """GET one of the read-only endpoints in ``API_READS`` — the only way this program talks to QuantPulse.
        Anything else is refused before a request is made."""
        if path not in API_READS:
            raise OpsError(f"not a read-only endpoint this program may call: {path}")
        req = urllib.request.Request(API + path, method="GET",
                                     headers={"X-API-Key": self.app_env.get("QP_API_TOKEN", "")})  # fmt: skip
        try:
            with self._urlopen(req, timeout=timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, None
        try:
            return status, json.loads(raw.decode() or "null")
        except ValueError:
            return status, None

    # ------------------------------------------------------------------ GitHub
    def repo_slug(self) -> str:
        slug = self.setting("QP_DEPLOY_REPO")
        if slug:
            return slug
        url = self.git("remote", "get-url", "origin")
        m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
        if not m:
            raise OpsError(
                "set QP_DEPLOY_REPO=owner/name in deploy/ops.env (the git remote is not on GitHub)"
            )
        return m.group(1)

    def branch(self) -> str:
        branch = self.setting("QP_DEPLOY_BRANCH")
        if not branch and (self.dir / ".deploy-branch").exists():
            branch = (self.dir / ".deploy-branch").read_text().strip()
        if not branch:
            raise OpsError("set QP_DEPLOY_BRANCH in deploy/ops.env (the branch this server follows)")
        return branch

    def _github(self, url: str) -> Any:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "quantpulse-server"}  # fmt: skip
        token = self.setting("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with self._urlopen(urllib.request.Request(url, headers=headers, method="GET"), timeout=20) as r:
            return json.loads(r.read().decode())

    def ci_gate(self, sha: str, branch: str | None = None) -> Gate:
        """Deploy ``sha`` only if the CI workflow's push run for exactly this commit (on the followed branch)
        completed with success and every required job — ARM64 included — succeeded. Anything else (no run yet,
        still running, a failure, a job missing or skipped, GitHub unreachable, an unexpected answer) is not a
        pass."""
        at = self.now().isoformat()
        if not SHA.match(sha):
            return Gate(False, "unknown", f"not a full commit id: {sha!r}", checked_at=at)
        base = f"https://api.github.com/repos/{self.repo_slug()}"
        try:
            listing = self._github(
                f"{base}/actions/workflows/ci.yml/runs?head_sha={sha}&event=push&per_page=50"
            )
            runs = [
                r
                for r in listing.get("workflow_runs", [])
                if r.get("head_sha") == sha
                and r.get("event") == "push"
                and str(r.get("path", "")).split("@")[0] == WORKFLOW
                and (branch is None or r.get("head_branch") == branch)
            ]
            if not runs:
                return Gate(False, "pending", "no CI run for this commit yet", checked_at=at)
            if any(r.get("status") != "completed" for r in runs):
                return Gate(False, "pending", "CI is still running", checked_at=at)
            latest = max(runs, key=lambda r: (int(r.get("run_number", 0)), int(r.get("run_attempt", 1))))
            url = latest.get("html_url")
            if latest.get("conclusion") != "success":
                return Gate(False, "failed", f"CI concluded {latest.get('conclusion')!r}", url, at)
            jobs = self._github(f"{base}/actions/runs/{int(latest['id'])}/jobs?filter=latest&per_page=100")
            by_name = {j.get("name"): j for j in jobs.get("jobs", [])}
            for name in REQUIRED_JOBS:
                job = by_name.get(name)
                if job is None:
                    return Gate(False, "failed", f"required CI job {name!r} did not run", url, at)
                if job.get("status") != "completed" or job.get("conclusion") != "success":
                    return Gate(
                        False, "failed", f"required CI job {name!r}: {job.get('conclusion')}", url, at
                    )
            odd = [n for n, j in by_name.items() if j.get("conclusion") not in ("success", "skipped")]
            if odd:
                return Gate(False, "failed", f"CI jobs not successful: {', '.join(map(str, odd))}", url, at)
            return Gate(
                True, "passed", f"CI passed on {sha[:12]} (every required job, ARM64 included)", url, at
            )
        except urllib.error.HTTPError as exc:
            return Gate(False, "unknown", f"GitHub answered {exc.code}: cannot confirm CI", checked_at=at)
        except Exception as exc:
            return Gate(False, "unknown", f"cannot reach GitHub ({type(exc).__name__}): cannot confirm CI",
                        checked_at=at)  # fmt: skip

    def _git_auth_env(self) -> dict[str, str]:
        """A private repository's token reaches git through its environment, never a command line."""
        token = self.setting("GITHUB_TOKEN")
        if not token:
            return {}
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        return {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}", "GIT_TERMINAL_PROMPT": "0"}  # fmt: skip

    def fetch(self, branch: str) -> str:
        self.git("fetch", "--quiet", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                 env=self._git_auth_env())  # fmt: skip
        return self.git("rev-parse", f"refs/remotes/origin/{branch}")

    # ------------------------------------------------------------------ deploys
    def deployed(self) -> str:
        return self.git("rev-parse", "HEAD")

    def schema_revision(self) -> str:
        out = self.dc("exec", "-T", "db", "psql", "-U", "quantpulse", "-d", "quantpulse", "-tAc",
                      "SELECT version_num FROM alembic_version", check=False)  # fmt: skip
        return out.stdout.strip() if out.returncode == 0 else ""

    def auto_update(self) -> str:
        if not self.flag("QP_AUTO_UPDATE", True):
            return "auto-update is off (QP_AUTO_UPDATE=false)"
        if (self.state_dir / "stopped").exists():
            return "stopped by a person (./qp down): not deploying; ./qp start resumes"
        with self.lock() as got:
            if not got:
                return "another deploy, backup or restore is running"
            resumed = self._resume_interrupted()
            if resumed:
                return resumed
            if (self.state_dir / "pin").exists():
                return f"pinned by a rollback to {(self.state_dir / 'pin').read_text().strip()[:12]}: ./qp update resumes"
            if self.setting("QP_AUTO_UPDATE_WINDOW", "closed") != "any" and in_quiet_hours(self.now()):
                return "market hours (08:00–17:30 New York, weekdays): no automatic deploy now"
            problems = self.host_preflight()
            if problems:
                self.alert("Server preflight FAILED: nothing is deployed", "; ".join(problems), "critical",
                           key="host-preflight")  # fmt: skip
                return "server preflight failed: " + "; ".join(problems)
            branch = self.branch()
            target = self.fetch(branch)
            current = self.deployed()
            if target == current:
                return f"up to date ({current[:12]})"
            if not self._is_ancestor(current, target):
                self.alert("Automatic deploy stopped", f"{branch} no longer contains the deployed commit "
                           f"{current[:12]} (history rewritten): deploy by hand with ./qp update",
                           key=f"rewritten:{target}")  # fmt: skip
                return "the branch history was rewritten: not deploying automatically"
            failed = self.load("deploy.json").get("failed", {})
            if target in failed:
                return f"{target[:12]} failed to deploy before ({failed[target]}): waiting for a newer commit"
            gate = self.ci_gate(target, branch)
            if not gate.ok:
                if gate.state == "failed":
                    self.alert("New commit not deployed: CI did not pass", f"{target[:12]}: {gate.reason}",
                               key=f"ci:{target}")  # fmt: skip
                return f"not deploying {target[:12]}: {gate.reason}"
            before = self._verdict()
            if before not in HEALTHY_AFTER_DEPLOY:
                self.alert("Automatic deploy postponed", f"the running version is not healthy ({before}): a "
                           "person should look first", key=f"unhealthy:{target}")  # fmt: skip
                return f"the running version is not healthy ({before}): not deploying on top of it"
            return self._deploy(target, current, gate, why="automatic")

    def _is_ancestor(self, older: str, newer: str) -> bool:
        return self.host.run(["git", "-C", str(self.repo_dir), "merge-base", "--is-ancestor", older, newer],
                             check=False).returncode == 0  # fmt: skip

    def deploy_now(self, target: str | None = None) -> str:
        """``./qp update``: the same gated deploy, now (inside market hours too); clears a rollback pin."""
        with self.lock(wait=True):
            problems = self.host_preflight()
            if problems:
                raise OpsError("server preflight failed: " + "; ".join(problems))
            resumed = self._resume_interrupted()
            if resumed:
                log(resumed)
            branch = self.branch()
            fetched = self.fetch(branch)
            target = target or fetched
            if not self._is_ancestor(target, fetched):
                raise OpsError(f"{target[:12]} is not on {branch}")
            current = self.deployed()
            if target != current and self._is_ancestor(target, current):
                raise OpsError(
                    f"{target[:12]} is older than what runs ({current[:12]}): use ./qp rollback to go back"
                )
            gate = self.ci_gate(target, branch)
            if not gate.ok:
                raise OpsError(f"not deploying {target[:12]}: {gate.reason} (only commits whose CI passed are "
                               "deployed)")  # fmt: skip
            (self.state_dir / "pin").unlink(missing_ok=True)
            state = self.load("deploy.json")
            state.get("failed", {}).pop(target, None)
            self.save("deploy.json", state)
            if target == current:
                return f"already running {current[:12]}"
            return self._deploy(target, current, gate, why="manual")

    def _deploy(self, target: str, current: str, gate: Gate, why: str) -> str:
        """Everything that can fail runs before the switch, while the old version keeps running: a backup, the
        image build, the preflight on the new image. The switch is one ``docker compose up``; the new version
        must then be healthy and ticking within 10 minutes, or the old one comes back (schema included)."""
        short, state = target[:12], self.load("deploy.json")
        log(f"deploying {short} ({why}; {gate.reason})")
        try:
            # the switch checks the commit out: an edited file in the checkout would stop it half-way
            changed = self.git("status", "--porcelain", "--untracked-files=no")
            if changed:
                raise OpsError(f"the checkout has local changes ({changed.splitlines()[0].strip()} …): commit or "
                               "undo them (git -C /opt/quantpulse status) — settings belong in deploy/.env")  # fmt: skip
            target_compose = self.git("show", f"{target}:deploy/compose.yaml", check=False)
            if not compose_forces_paper(target_compose):
                raise OpsError("the new version's deploy/compose.yaml does not force QP_DEPLOYMENT=cloud and "
                               "QP_ALPACA_PAPER=true: refused")  # fmt: skip
            dump = self.backup(kind="predeploy")["file"]
            self.build(target)
            self.dc("run", "--rm", "--no-deps", "-T", "api", "quantpulse-preflight",
                    env={"QP_IMAGE_TAG": short}, timeout=300)  # fmt: skip
        except OpsError as exc:  # nothing has changed: the old version still runs
            tries = state.setdefault("attempts", {})
            tries[target] = int(tries.get(target, 0)) + 1
            if tries[target] >= PRE_SWITCH_TRIES or why == "manual":
                state.setdefault("failed", {})[target] = f"before the switch: {exc}"[:300]
            self.save("deploy.json", state)
            self.alert("Deploy failed before the switch (nothing changed)", f"{short} (attempt {tries[target]}): "
                       f"{exc}", "warning")  # fmt: skip
            return f"not deployed: {exc}"
        old_schema = self.schema_revision()
        state.update(phase="switching", target=target, previous=current, previous_schema=old_schema,
                     predeploy_dump=dump, started_at=self.now().isoformat())  # fmt: skip
        self.save("deploy.json", state)
        self._retag(target)
        self.git("checkout", "--quiet", "--detach", target)
        self.dc("up", "-d", "--no-build", "api", "dashboard", timeout=900, check=False)
        state["phase"] = "verifying"
        self.save("deploy.json", state)
        ok, verdict = self.wait_ticking(commit=target)
        if ok:
            state.update(phase="done", deployed=target, deployed_at=self.now().isoformat(), gate=asdict(gate))
            state.setdefault("history", []).append(
                {"commit": target, "at": self.now().isoformat(), "why": why}
            )
            state["history"] = state["history"][-30:]
            self.save("deploy.json", state)
            self._prune_images()
            self.alert("Deployed", f"{short} is running and the Brain supervisor is {verdict}", "info")
            return f"deployed {short}"
        minutes = int(VERIFY_FOR.total_seconds() // 60)
        return self.rollback_failed(
            state, f"the new version was not healthy within {minutes} minutes ({verdict})"
        )

    def build(self, target: str) -> None:
        """The production image of exactly ``target`` (from git, not from the working tree), tagged by commit."""
        short = target[:12]
        have = (
            self.host.run(["docker", "image", "inspect", f"quantpulse:{short}"], check=False).returncode == 0
        )
        if have:
            return
        log(f"building quantpulse:{short} (several minutes on a small ARM server) ...")
        self.host.pipe(
            ["git", "-C", str(self.repo_dir), "archive", "--format=tar", target],
            ["docker", "build", "--build-arg", f"QP_GIT_COMMIT={target}", "-t", f"quantpulse:{short}", "-"],
        )

    def _retag(self, target: str) -> None:
        if self.host.run(["docker", "image", "inspect", "quantpulse:current"], check=False).returncode == 0:
            self.host.run(["docker", "tag", "quantpulse:current", "quantpulse:previous"])
        self.host.run(["docker", "tag", f"quantpulse:{target[:12]}", "quantpulse:current"])

    def _prune_images(self) -> None:
        """Keep the images of the running and the previous version; drop older ones (disk is small)."""
        state = self.load("deploy.json")
        keep = {
            "current",
            "previous",
            str(state.get("deployed", ""))[:12],
            str(state.get("previous", ""))[:12],
        }
        out = self.host.run(
            ["docker", "image", "ls", "quantpulse", "--format", "{{.Tag}}"], check=False
        ).stdout
        for tag in out.split():
            if tag not in keep and tag != "<none>":
                self.host.run(["docker", "image", "rm", f"quantpulse:{tag}"], check=False)
        self.host.run(["docker", "image", "prune", "-f"], check=False)

    def wait_ticking(self, within: timedelta = VERIFY_FOR, commit: str | None = None) -> tuple[bool, str]:
        """The API answers and the supervisor is ticking (or deliberately paused/off). "blocked" — startup
        recovery refusing to resume — is a failure here: the version before it was healthy. With ``commit``,
        the answer must come from that version (the commit is built into the image), never from the old one."""
        deadline, verdict = self.now() + within, "no answer"
        while True:
            verdict, running = self._verdict_of()
            if commit is not None and verdict in HEALTHY_AFTER_DEPLOY and running != commit:
                verdict = f"the old version still answers ({str(running)[:12]})"
            elif verdict in HEALTHY_AFTER_DEPLOY:
                return True, verdict
            if self.now() >= deadline:
                return False, verdict
            self.sleep(15)

    def _verdict(self) -> str:
        return self._verdict_of()[0]

    def _verdict_of(self) -> tuple[str, str | None]:
        try:
            status, body = self.api_get(WATCHDOG_PATH)
        except Exception as exc:
            return f"no answer ({type(exc).__name__})", None
        if status != 200 or not isinstance(body, dict):
            return f"no answer (HTTP {status})", None
        commit = body.get("commit")
        return str(body.get("verdict")), str(commit) if commit else None

    def rollback_failed(self, state: dict[str, Any], why: str) -> str:
        """Back to the version that ran before: the schema is undone by the new image (it knows its own
        migrations) — or, if that fails, the pre-deploy backup is restored — then the old image starts."""
        target, previous = str(state.get("target")), str(state.get("previous"))
        log(f"rolling back {target[:12]} → {previous[:12]}: {why}")
        self.dc("stop", "api", "dashboard", timeout=300, check=False)
        schema_note = "schema unchanged"
        old_schema = str(state.get("previous_schema") or "")
        if old_schema and self.schema_revision() not in ("", old_schema):
            down = self.dc("run", "--rm", "--no-deps", "-T", "api", "quantpulse-migrate", old_schema, "--downgrade",
                           timeout=600, check=False)  # fmt: skip
            schema_note = f"schema back to {old_schema}"
            if down.returncode != 0:
                dump = str(state.get("predeploy_dump") or "")
                self.restore(dump, confirm=True)
                schema_note = f"schema downgrade failed: the pre-deploy backup {dump} was restored"
        if self.host.run(["docker", "image", "inspect", "quantpulse:previous"], check=False).returncode == 0:
            self.host.run(["docker", "tag", "quantpulse:previous", "quantpulse:current"])
        self.git("checkout", "--quiet", "--detach", previous)
        self.dc("up", "-d", "--no-build", timeout=900, check=False)
        ok, verdict = self.wait_ticking()
        state.setdefault("failed", {})[target] = why[:300]
        state.update(phase="rolled_back" if ok else "rollback_unhealthy", deployed=previous,
                     rolled_back_at=self.now().isoformat())  # fmt: skip
        self.save("deploy.json", state)
        severity = "warning" if ok else "critical"
        self.alert("Deploy failed: rolled back", f"{target[:12]}: {why}. Back on {previous[:12]} ({schema_note}); "
                   f"supervisor {verdict}.", severity)  # fmt: skip
        return f"rolled back to {previous[:12]} ({verdict})"

    def _resume_interrupted(self) -> str | None:
        """A deploy cut off mid-switch (a reboot, a kill): verify what runs now; roll back if it is not healthy."""
        state = self.load("deploy.json")
        if state.get("phase") not in ("switching", "verifying"):
            return None
        started = datetime.fromisoformat(str(state.get("started_at")))
        if self.now() - started < INTERRUPTED_AFTER:
            return None
        ok, verdict = self.wait_ticking(timedelta(minutes=2))
        if ok and self.deployed() == state.get("target"):
            state.update(phase="done", deployed=state.get("target"), deployed_at=self.now().isoformat())
            self.save("deploy.json", state)
            return f"an interrupted deploy of {str(state.get('target'))[:12]} turned out healthy: kept"
        return self.rollback_failed(
            state, f"the deploy was interrupted and the result is not healthy ({verdict})"
        )

    def host_preflight(self) -> list[str]:
        """The server's own hard preflight, before anything starts or is deployed (``./qp start``, every deploy).
        A second wall in front of the API's preflight, which refuses to start on its own: deploy/ops.env and the
        host's environment never reach the container, so they are checked here. Problems name files and
        variables, never values; an empty list is a pass."""
        problems: list[str] = []
        for name in (".env", "ops.env"):
            path = self.dir / name
            if not path.exists():
                continue
            for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = raw.strip()
                if line and not line.startswith("#") and LIVE_ENDPOINT.search(line):
                    key = line.split("=", 1)[0].strip() if "=" in line else f"line {number}"
                    problems.append(f"deploy/{name}: {key} names Alpaca's live-money API")
        for key, value in os.environ.items():
            if key.startswith(HOST_PREFIXES) and LIVE_ENDPOINT.search(value):
                problems.append(f"the server's environment variable {key} names Alpaca's live-money API")
        if not (self.dir / ".env").exists():
            problems.append("deploy/.env is missing (./qp setup makes it)")
        elif self.app_env.get("QP_ALPACA_PAPER", "").strip().lower() != "true":
            problems.append("deploy/.env: QP_ALPACA_PAPER must be exactly true")
        deployment = self.app_env.get("QP_DEPLOYMENT", "cloud").strip().lower()
        if deployment != "cloud":
            problems.append(
                f"deploy/.env: QP_DEPLOYMENT={deployment} is ambiguous on a server (cloud is forced)"
            )
        compose = self.dir / "compose.yaml"
        if not compose_forces_paper(compose.read_text(encoding="utf-8") if compose.exists() else ""):
            problems.append("deploy/compose.yaml does not force QP_DEPLOYMENT=cloud and QP_ALPACA_PAPER=true")
        access = self.setting("QP_DASHBOARD_ACCESS", "tailscale").strip().lower()
        if access not in ("tailscale", "public"):
            problems.append(f"deploy/ops.env: QP_DASHBOARD_ACCESS={access} (tailscale or public)")
        return problems

    def tailscale(self) -> dict[str, Any]:
        """How the dashboard is reached: on the tailnet only (``tailscale serve``), never the public internet
        (Tailscale Funnel or the caddy profile) unless QP_DASHBOARD_ACCESS=public."""
        out: dict[str, Any] = {"access": self.setting("QP_DASHBOARD_ACCESS", "tailscale").strip().lower()}
        if shutil.which("tailscale") is None:
            return {**out, "installed": False}
        status = self.host.run(["tailscale", "status", "--json"], check=False, timeout=15)
        try:
            info = json.loads(status.stdout) if status.returncode == 0 and status.stdout.strip() else {}
        except ValueError:
            info = {}
        serve = self.host.run(["tailscale", "serve", "status"], check=False, timeout=15)
        text = serve.stdout if serve.returncode == 0 else ""
        return {**out, "installed": True, "state": info.get("BackendState"),
                "name": str((info.get("Self") or {}).get("DNSName") or "").rstrip("."),
                "serving": "8501" in text, "funnel": "funnel on" in text.lower()}  # fmt: skip

    def record_start(self) -> None:
        """``./qp start`` built and started the working tree's commit by hand: what runs now."""
        state = self.load("deploy.json")
        head = self.deployed()
        state.update(phase="done", deployed=head, deployed_at=self.now().isoformat())
        state.pop("gate", None)  # built from the working tree: its CI result is looked up when asked
        state.setdefault("history", []).append(
            {"commit": head, "at": self.now().isoformat(), "why": "./qp start"}
        )
        state["history"] = state["history"][-30:]
        self.save("deploy.json", state)

    def pin(self, commit: str) -> None:
        """After a person's rollback: automatic deploys pause until ``./qp update``."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "pin").write_text(commit + "\n")

    # ------------------------------------------------------------------ the watchdog
    def watchdog(self) -> str:
        """Once a minute. The only request to QuantPulse is ``GET /api/v1/system/watchdog``; the only action is
        ``docker compose restart api`` — the graceful stop a deploy does — when the supervisor is stalled or the
        API has stopped answering. A restart never sends an order: the new process reconciles with Alpaca and
        runs the safety audit before anything else, and the single-supervisor lease is kept throughout."""
        self.sample()
        self._idle_warning()
        if not self.flag("QP_WATCHDOG", True):
            return "the watchdog is off (QP_WATCHDOG=false)"
        if (self.state_dir / "stopped").exists():
            return "stopped by a person (./qp down): nothing to watch"
        with self.lock() as got:
            if not got:
                return "a deploy, backup or restore is running: not watching now"
            return self._watch()

    def _watch(self) -> str:
        st = self.load("watchdog.json")
        container = self._api_container()
        if container is None or container.get("State") != "running":
            st.update(unreachable=0, stalled=0)
            self.save("watchdog.json", st)
            return "the api container is not running (stopped by a person, or Docker is restarting it)"
        started = container.get("StartedAt")
        if started and self.now() - datetime.fromisoformat(started) < API_START_GRACE:
            return "the api container started less than 5 minutes ago: waiting"
        try:
            status, body = self.api_get(WATCHDOG_PATH)
        except Exception as exc:
            status, body = 0, None
            log(f"watchdog: no answer from the API ({type(exc).__name__})")
        cause: str | None = None
        if status == 200 and isinstance(body, dict) and "verdict" in body:
            st["unreachable"] = 0
            st["last"] = {
                "at": self.now().isoformat(),
                "verdict": body["verdict"],
                "reason": body.get("reason"),
            }
            if body["verdict"] == "stalled" and body.get("restart") is True:
                st["stalled"] = int(st.get("stalled", 0)) + 1
                if st["stalled"] >= STALLED_CHECKS:
                    cause = f"the Brain supervisor is stalled: {body.get('reason')}"
            else:
                st["stalled"] = 0
        elif status == 0 or status >= 500:  # no connection, a timeout, or the server failing
            st["unreachable"] = int(st.get("unreachable", 0)) + 1
            st["last"] = {
                "at": self.now().isoformat(),
                "verdict": "no answer",
                "reason": f"HTTP {status or 'none'}",
            }
            if st["unreachable"] >= UNREACHABLE_CHECKS:
                cause = f"the API has not answered for {st['unreachable']} minutes (HTTP {status or 'none'})"
        else:  # it answers (a wrong token, an older version without the endpoint): alive; a person must look
            st.update(unreachable=0, stalled=0)
            st["last"] = {"at": self.now().isoformat(), "verdict": "unknown", "reason": f"HTTP {status}"}
            self.alert("Watchdog cannot read the supervisor", f"GET {WATCHDOG_PATH} answered {status} (is "
                       "QP_API_TOKEN in deploy/.env the API's?); not restarting anything", key=f"http:{status}")  # fmt: skip
        if cause is None:
            self.save("watchdog.json", st)
            return f"{st['last']['verdict']}: {st['last'].get('reason') or ''}".strip()
        recent = [
            t for t in st.get("restarts", []) if self.now() - datetime.fromisoformat(t) < RESTART_WINDOW
        ]
        if len(recent) >= MAX_RESTARTS:
            st["restarts"] = recent
            self.save("watchdog.json", st)
            self.alert("Watchdog gave up: a person is needed", f"{cause}. Already restarted {len(recent)} times in "
                       "6 hours; not restarting again. See ./qp status and ./qp logs.", "critical",
                       key="watchdog-gave-up")  # fmt: skip
            return f"not restarting (limit reached): {cause}"
        recent.append(self.now().isoformat())
        st.update(restarts=recent, stalled=0, unreachable=0)
        self.save("watchdog.json", st)
        log(f"watchdog: restarting the api container — {cause}")
        done = self.dc("restart", "api", timeout=420, check=False)
        result = "restarted" if done.returncode == 0 else f"restart failed ({done.returncode})"
        self.alert("Watchdog restarted the API", f"{cause}. {result}; the Brain recovers (reconciles, audits) "
                   "before any order.", "warning")  # fmt: skip
        return f"{result}: {cause}"

    def _idle_warning(self) -> None:
        """Once in 6 hours at most: warn when a day or more of samples looks idle by Oracle's rule — the VM may
        be stopped (not deleted: start it again in the console, and QuantPulse recovers by itself)."""
        idle = self.idle_check()
        if idle["at_risk"] and idle["samples"] >= IDLE_MIN_SAMPLES:
            self.alert("Oracle may stop this VM as idle", f"over the last {idle['samples']} minutes CPU p95 was "
                       f"{idle['cpu_p95']}% and memory {idle['memory_now']}% — Oracle may stop an Always Free VM "
                       "whose CPU p95, network and memory all stay under 20% for 7 days. See deploy/ORACLE.md "
                       "(idle reclamation).", "warning", key="idle-risk")  # fmt: skip

    def _api_container(self) -> dict[str, Any] | None:
        out = self.dc("ps", "--all", "--format", "json", "api", check=False)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        rows = _json_lines(out.stdout)
        if not rows:
            return None
        row = rows[0]
        cid = row.get("ID") or row.get("Id")
        if cid:
            inspect = self.host.run(
                ["docker", "inspect", "--format", "{{json .State}}", str(cid)], check=False
            )
            if inspect.returncode == 0 and inspect.stdout.strip():
                state = json.loads(inspect.stdout)
                return {"State": state.get("Status"), "StartedAt": _docker_time(state.get("StartedAt")),
                        "Health": (state.get("Health") or {}).get("Status")}  # fmt: skip
        return {"State": row.get("State"), "StartedAt": None, "Health": row.get("Health")}

    # ------------------------------------------------------------------ backups
    def backup(self, kind: str = "manual") -> dict[str, Any]:
        """``pg_dump`` inside the backup container (the database password never leaves it), checked with
        ``pg_restore --list``; old local dumps pruned; the nightly one uploaded off the server."""
        if kind not in ("nightly", "manual", "predeploy"):
            raise OpsError(f"unknown backup kind {kind}")
        name = f"quantpulse-{kind}-{self.now():%Y%m%d-%H%M%S}.dump"
        record: dict[str, Any] = {"at": self.now().isoformat(), "kind": kind, "file": name}
        state = self.load("backup.json")
        try:
            script = (
                'set -e; f="/backups/$QP_NAME"; pg_dump -Fc -f "$f.part"; mv "$f.part" "$f"; '
                'n=$(pg_restore --list "$f" | grep -c "TABLE DATA" || true); echo "tables=$n"; '
                f"find /backups -name 'quantpulse-nightly-*.dump' -mtime +{KEEP_DUMPS_DAYS} -delete; "
                f"find /backups -name 'quantpulse-predeploy-*.dump' -mtime +{KEEP_DUMPS_DAYS} -delete; "
                f"find /backups -name 'quantpulse-manual-*.dump' -mtime +{KEEP_MANUAL_DAYS} -delete"
            )
            out = self.dc("exec", "-T", "-e", f"QP_NAME={name}", "backup", "sh", "-c", script, timeout=3600)
            found = re.search(r"tables=(\d+)", out.stdout)
            tables = int(found.group(1)) if found else 0
            if tables == 0:
                raise OpsError("the dump holds no table data")
            path = self.dir / "backups" / name
            record.update(tables=tables, bytes=path.stat().st_size, sha256=_sha256(path), verified=True)
            if kind == "nightly":
                record["upload"] = self.upload(path)
            state["last_ok"] = record
            state.pop("last_error", None)
            self.save("backup.json", state)
            log(f"backup {name}: {tables} tables, {record['bytes'] / 1e6:.1f} MB, verified")
            if kind == "nightly":
                self.ping(self.setting("QP_BACKUP_HEARTBEAT_URL"), True)
            return record
        except Exception as exc:
            record["error"] = redact(f"{type(exc).__name__}: {exc}")[:400]
            state["last_error"] = record
            self.save("backup.json", state)
            self.alert("Database backup FAILED", f"{kind} backup {name}: {record['error']}", "critical")
            if kind == "nightly":
                self.ping(self.setting("QP_BACKUP_HEARTBEAT_URL"), False)
            raise OpsError(record["error"]) from exc

    def upload(self, path: Path) -> dict[str, Any]:
        """PUT the dump to Object Storage through a pre-authenticated request that may only *write* objects:
        this server can neither read nor list nor delete what is stored (the bucket's lifecycle rules keep the
        generations). The MD5 is checked by Object Storage on arrival."""
        par = self.setting("QP_BACKUP_PAR_URL")
        if not par:
            return {"skipped": "QP_BACKUP_PAR_URL is not set: kept on this server only"}
        now = self.now()
        tier = "monthly" if now.day == 1 else "weekly" if now.weekday() == 6 else "daily"
        obj = f"{tier}/{path.name}"
        md5 = hashlib.md5(usedforsecurity=False)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                md5.update(chunk)
        digest = base64.b64encode(md5.digest()).decode()
        headers = {"Content-Type": "application/octet-stream", "Content-MD5": digest,
                   "Content-Length": str(path.stat().st_size)}  # fmt: skip
        error: Exception = OpsError("not attempted")
        for attempt in range(3):
            try:
                with open(path, "rb") as fh:
                    req = urllib.request.Request(
                        par.rstrip("/") + "/" + obj, data=fh, headers=headers, method="PUT"
                    )
                    with self._urlopen(req, timeout=900) as r:
                        echoed = r.headers.get("opc-content-md5")
                        if r.status >= 300 or (echoed and echoed != digest):
                            raise OpsError(f"Object Storage answered {r.status}")
                return {"object": obj, "md5": digest, "at": self.now().isoformat()}
            except urllib.error.HTTPError as exc:
                error = OpsError(f"Object Storage answered {exc.code}")
            except Exception as exc:
                error = exc
            if attempt < 2:
                self.sleep(30 * (attempt + 1))
        raise OpsError(f"upload failed: {redact(str(error))}")

    def newest_dump(self) -> Path:
        dumps = sorted((self.dir / "backups").glob("quantpulse-*.dump"), key=lambda p: p.stat().st_mtime)
        if not dumps:
            raise OpsError("no backup in deploy/backups yet (./qp backup makes one)")
        return dumps[-1]

    def restore_test(self, dump: Path | None = None) -> dict[str, Any]:
        """Restore a dump into a scratch database next to the live one and check it: the restore must finish
        without an error, carry a schema version and the same tables as the live database, with rows in them.
        The scratch database is dropped afterwards; the live one is only read."""
        dump = dump or self.newest_dump()
        result: dict[str, Any] = {"at": self.now().isoformat(), "file": dump.name}
        state = self.load("backup.json")
        script = (
            f"set -e; dropdb --if-exists {RESTORE_DB}; createdb {RESTORE_DB}; "
            f'pg_restore --exit-on-error --no-owner --no-privileges -d {RESTORE_DB} "/backups/$QP_NAME"; '
            f"echo rev=$(psql -d {RESTORE_DB} -tAc 'SELECT version_num FROM alembic_version'); "
            f'echo tables=$(psql -d {RESTORE_DB} -tAc "SELECT count(*) FROM information_schema.tables '
            "WHERE table_schema='public'\"); "
            'echo live_tables=$(psql -tAc "SELECT count(*) FROM information_schema.tables WHERE '
            "table_schema='public'\"); "
            f'echo rows=$(psql -d {RESTORE_DB} -tAc "SELECT (SELECT count(*) FROM alembic_version) + '
            'COALESCE((SELECT sum(n_live_tup) FROM pg_stat_user_tables), 0)"); '
            f"dropdb {RESTORE_DB}"
        )
        try:
            with self.lock(wait=True):
                out = self.dc("exec", "-T", "-e", f"QP_NAME={dump.name}", "backup", "sh", "-c", script, timeout=3600,
                              check=False)  # fmt: skip
            values = dict(re.findall(r"^(\w+)=(.*)$", out.stdout, flags=re.M))
            result.update(values)
            problems = []
            if out.returncode != 0:
                problems.append(f"the restore failed: {redact((out.stderr or '').strip()[-300:])}")
            if not values.get("rev"):
                problems.append("no schema version in the restored database")
            if values.get("tables") != values.get("live_tables"):
                problems.append(f"{values.get('tables')} tables restored, the live database has "
                                f"{values.get('live_tables')}")  # fmt: skip
            if problems:
                self.dc("exec", "-T", "backup", "dropdb", "--if-exists", RESTORE_DB, check=False)
                raise OpsError("; ".join(problems))
            result["ok"] = True
            state["restore_test"] = result
            self.save("backup.json", state)
            log(
                f"restore test passed: {dump.name} (schema {values.get('rev')}, {values.get('tables')} tables)"
            )
            return result
        except Exception as exc:
            result.update(ok=False, error=redact(str(exc))[:400])
            state["restore_test"] = result
            self.save("backup.json", state)
            self.alert("Backup restore test FAILED", f"{dump.name}: {result['error']}", "critical")
            raise OpsError(result["error"]) from exc

    def restore(self, dump: str, confirm: bool = False) -> None:
        """Replace the live database with ``dump`` (the API must be stopped; ./qp restore asks first)."""
        if not confirm:
            raise OpsError("restore needs confirmation")
        name = Path(dump).name
        if not (self.dir / "backups" / name).exists():
            raise OpsError(f"deploy/backups/{name} does not exist")
        self.dc("exec", "-T", "backup", "pg_restore", "--clean", "--if-exists", "--no-owner", "-d", "quantpulse",
                f"/backups/{name}", timeout=3600, check=False)  # fmt: skip

    # ------------------------------------------------------------------ host measurements
    def sample(self) -> dict[str, float]:
        """CPU (since the last sample) and memory use, kept for 7 days: Oracle reclaims an Always Free instance
        whose CPU p95, network and memory all stay under 20% for 7 days."""
        cpu = _cpu_times()
        prev = self.load("cpu.json")
        busy = 0.0
        if prev.get("total") and cpu[1] > prev["total"]:
            busy = 100.0 * (1 - (cpu[0] - prev["idle"]) / (cpu[1] - prev["total"]))
        self.save("cpu.json", {"idle": cpu[0], "total": cpu[1]})
        mem = _memory()
        point = {"t": self.now().timestamp(), "cpu": round(busy, 1), "mem": round(mem["used_pct"], 1)}
        path = self.state_dir / "samples.jsonl"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(point) + "\n")
        if path.stat().st_size > 1_500_000:  # ~7 days of minutes: keep the last week only
            cutoff = (self.now() - SAMPLES_KEEP).timestamp()
            kept = [line for line in path.read_text().splitlines() if json.loads(line)["t"] >= cutoff]
            path.write_text("\n".join(kept) + "\n")
        return point

    def idle_check(self) -> dict[str, Any]:
        path = self.state_dir / "samples.jsonl"
        cutoff = (self.now() - SAMPLES_KEEP).timestamp()
        points = []
        if path.exists():
            for line in path.read_text().splitlines():
                with contextlib.suppress(ValueError):
                    p = json.loads(line)
                    if p["t"] >= cutoff:
                        points.append(p)
        cpu = sorted(p["cpu"] for p in points)
        mem = sorted(p["mem"] for p in points)
        p95 = cpu[int(0.95 * (len(cpu) - 1))] if cpu else None
        mem_p95 = mem[int(0.95 * (len(mem) - 1))] if mem else None
        now_mem = _memory()["used_pct"]
        # Oracle: idle when CPU p95, network AND memory all stay under 20% (network is never near 20% of the link
        # here, so CPU or memory must carry it); judged on the recorded window, like Oracle's 7 days
        at_risk = bool(points) and (p95 or 0.0) < IDLE_THRESHOLD and (mem_p95 or 0.0) < IDLE_THRESHOLD
        return {"samples": len(points), "cpu_p95": p95, "memory_now": round(now_mem, 1), "memory_p95": mem_p95,
                "threshold": IDLE_THRESHOLD, "at_risk": at_risk}  # fmt: skip

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"at": self.now().isoformat(), "host": socket.gethostname()}
        load1, _, _ = os.getloadavg()
        mem = _memory()
        disk = shutil.disk_usage("/")
        out["cpu"] = {"cores": os.cpu_count(), "load1": round(load1, 2), "busy_pct": self.sample()["cpu"]}
        out["memory"] = mem
        out["disk"] = {
            "total_gb": round(disk.total / 1e9, 1),
            "used_pct": round(100 * disk.used / disk.total, 1),
        }
        out["idle"] = self.idle_check()
        ps = self.dc("ps", "--all", "--format", "json", check=False)
        out["docker"] = [
            {"service": r.get("Service"), "state": r.get("State"), "health": r.get("Health") or ""}
            for r in _json_lines(ps.stdout)
        ] if ps.returncode == 0 else f"docker compose ps failed: {redact(ps.stderr.strip()[-200:])}"  # fmt: skip
        ready = self.dc("exec", "-T", "db", "pg_isready", "-U", "quantpulse", "-d", "quantpulse", check=False)
        size = self.dc("exec", "-T", "db", "psql", "-U", "quantpulse", "-d", "quantpulse", "-tAc",
                       "SELECT pg_size_pretty(pg_database_size('quantpulse'))", check=False)  # fmt: skip
        out["postgres"] = {"ready": ready.returncode == 0, "size": size.stdout.strip() if size.returncode == 0 else None,
                           "schema": self.schema_revision() or None}  # fmt: skip
        try:
            code, _ = self.api_get("/health", timeout=5)
            out["api"] = {"health": code}
            code, wd = self.api_get(WATCHDOG_PATH)
            out["supervisor"] = wd if code == 200 else {"verdict": f"no answer (HTTP {code})"}
            code, health = self.api_get("/api/v1/system/health?fresh=true", timeout=60)
            out["health"] = health if code == 200 else None
        except Exception as exc:
            out["api"] = {"health": f"no answer ({type(exc).__name__})"}
            out["supervisor"], out["health"] = {"verdict": "no answer"}, None
        backup = self.load("backup.json")
        out["backup"] = {"last_ok": backup.get("last_ok"), "last_error": backup.get("last_error"),
                         "restore_test": backup.get("restore_test"),
                         "offsite": bool(self.setting("QP_BACKUP_PAR_URL"))}  # fmt: skip
        deploy = self.load("deploy.json")
        try:
            head = self.deployed()
        except OpsError:
            head = ""
        gate = deploy.get("gate") if deploy.get("deployed") == head else None
        if head and (gate is None or gate.get("state") != "passed"):
            gate = asdict(self.ci_gate(head, self.setting("QP_DEPLOY_BRANCH") or None))
        out["deploy"] = {"commit": head, "phase": deploy.get("phase"), "deployed_at": deploy.get("deployed_at"),
                         "ci": gate, "pinned": (self.state_dir / "pin").exists(),
                         "auto_update": self.flag("QP_AUTO_UPDATE", True),
                         "window": self.setting("QP_AUTO_UPDATE_WINDOW", "closed"),
                         "failed": list(deploy.get("failed", {}))[-3:]}  # fmt: skip
        out["dashboard"] = self.tailscale()
        if isinstance(out["docker"], list):
            out["dashboard"]["caddy"] = any(
                d["service"] == "caddy" and d["state"] == "running" for d in out["docker"]
            )
        out["preflight"] = self.host_preflight()
        wd = self.load("watchdog.json")
        recent = [
            t for t in wd.get("restarts", []) if self.now() - datetime.fromisoformat(t) < RESTART_WINDOW
        ]
        out["watchdog"] = {"enabled": self.flag("QP_WATCHDOG", True), "restarts_6h": len(recent),
                           "last": wd.get("last")}  # fmt: skip
        return out


# --------------------------------------------------------------------------------------------- helpers
def compose_forces_paper(text: str) -> bool:
    """The api service's environment in deploy/compose.yaml forces cloud mode and paper trading."""
    return bool(re.search(r"^\s+QP_DEPLOYMENT:\s*cloud\s*$", text, re.M)) and bool(
        re.search(r'^\s+QP_ALPACA_PAPER:\s*"true"', text, re.M)
    )


def _json_lines(text: str) -> list[dict[str, Any]]:
    """``docker compose ps --format json`` prints one object per line (newer) or one array (older)."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _docker_time(value: str | None) -> str | None:
    if not value or value.startswith("0001"):
        return None
    m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:\d\d)$", value)
    if not m:
        return None
    return m.group(1) + ("+00:00" if m.group(3) == "Z" else m.group(3))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_times() -> tuple[float, float]:
    try:
        fields = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)  # idle + iowait
    return idle, sum(fields[:8])


def _memory() -> dict[str, float]:
    info: dict[str, float] = {}
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            info[key] = float(rest.split()[0]) * 1024
    total = info.get("MemTotal", 0.0)
    avail = info.get("MemAvailable", total)
    swap_total, swap_free = info.get("SwapTotal", 0.0), info.get("SwapFree", 0.0)
    return {
        "total_gb": round(total / 1e9, 2),
        "used_pct": round(100 * (total - avail) / total, 1) if total else 0.0,
        "swap_used_pct": round(100 * (swap_total - swap_free) / swap_total, 1) if swap_total else 0.0,
    }


def _when(value: str | None) -> str:
    if not value:
        return "never"
    with contextlib.suppress(ValueError):
        return f"{datetime.fromisoformat(value).astimezone(UTC):%Y-%m-%d %H:%M} UTC"
    return value


def render_status(s: dict[str, Any]) -> str:
    """``./qp status`` for a person (``--json`` gives the raw data)."""
    lines = [f"QuantPulse server {s['host']} — {_when(s['at'])} (Alpaca PAPER only)"]
    cpu, mem, disk, idle = s["cpu"], s["memory"], s["disk"], s["idle"]
    lines.append(f"  host        CPU {cpu['busy_pct']:.0f}% (load {cpu['load1']}, {cpu['cores']} cores) · memory "
                 f"{mem['used_pct']:.0f}% of {mem['total_gb']} GB (swap {mem['swap_used_pct']:.0f}%) · disk "
                 f"{disk['used_pct']:.0f}% of {disk['total_gb']} GB")  # fmt: skip
    p95 = "n/a" if idle["cpu_p95"] is None else f"{idle['cpu_p95']:.0f}%"
    mem95 = "n/a" if idle["memory_p95"] is None else f"{idle['memory_p95']:.0f}%"
    verdict = "AT RISK of idle reclamation" if idle["at_risk"] else "not idle"
    lines.append(f"  idle check  7 d ({idle['samples']} samples): CPU p95 {p95}, memory p95 {mem95} (now "
                 f"{idle['memory_now']:.0f}%) vs Oracle's {idle['threshold']:.0f}% → {verdict}")  # fmt: skip
    if isinstance(s["docker"], list):
        parts = [
            f"{d['service']} {d['state']}" + (f" ({d['health']})" if d["health"] else "") for d in s["docker"]
        ]
        lines.append("  docker      " + (" · ".join(parts) or "no containers (./qp start)"))
    else:
        lines.append(f"  docker      {s['docker']}")
    pg = s["postgres"]
    lines.append(f"  postgres    {'accepting connections' if pg['ready'] else 'NOT READY'} · "
                 f"{pg['size'] or '?'} · schema {pg['schema'] or '?'}")  # fmt: skip
    sup = s["supervisor"] or {}
    lines.append(f"  api         /health {s['api'].get('health')} · commit "
                 f"{(sup.get('commit') or '?')[:12]} · started {_when(sup.get('started_at'))}")  # fmt: skip
    lines.append(f"  supervisor  {str(sup.get('verdict', '?')).upper()}: {sup.get('reason', '')} · "
                 f"{'leader' if sup.get('leader') else 'not leader'} · last tick {_when(sup.get('last_tick_at'))}"
                 f" ({sup.get('last_result') or '-'})"[:220])  # fmt: skip
    cycle = sup.get("last_cycle") or {}
    lines.append("  last cycle  " + (f"#{cycle.get('id')} {cycle.get('kind')} {cycle.get('status')} "
                                      f"{_when(cycle.get('started_at'))}" if cycle else "none yet"))  # fmt: skip
    beat = sup.get("alert_heartbeat") or {}
    lines.append("  heartbeat   " + (f"last healthy ping delivered {_when(beat.get('last_healthy_delivered_at'))}; "
                                     f"last ping {_when(beat.get('at'))} "
                                     f"({'healthy' if beat.get('healthy') else 'UNHEALTHY'}, "
                                     f"{'delivered' if beat.get('delivered') else 'NOT delivered'})"
                                     if beat else "no ping yet (QP_HEARTBEAT_URL unset, or none due yet)"))  # fmt: skip
    b = s["backup"]
    ok, err, rt = b.get("last_ok") or {}, b.get("last_error"), b.get("restore_test") or {}
    upload = ok.get("upload") or {}
    where = upload.get("object") or upload.get("skipped") or "local only"
    text = (f"last {_when(ok.get('at'))} ({ok.get('kind')}, {ok.get('bytes', 0) / 1e6:.1f} MB, {where})"
            if ok else "none yet")  # fmt: skip
    if err and (not ok or err.get("at", "") > ok.get("at", "")):
        text += f" · LAST ATTEMPT FAILED {_when(err.get('at'))}: {err.get('error')}"
    text += (f" · restore test {_when(rt.get('at'))} {'OK' if rt.get('ok') else 'FAILED'}" if rt
             else " · no restore test yet")  # fmt: skip
    lines.append(f"  backup      {text}" + ("" if b["offsite"] else " · OFF-SITE NOT CONFIGURED"))
    d = s["deploy"]
    ci = d.get("ci") or {}
    lines.append(f"  deployed    {d['commit'][:12] or '?'} at {_when(d.get('deployed_at'))} · CI {ci.get('state')}: "
                 f"{ci.get('reason')} · auto-update {'on' if d['auto_update'] else 'OFF'} ({d['window']})"
                 + (" · PINNED by a rollback" if d["pinned"] else "")
                 + (f" · phase {d['phase']}" if d.get("phase") not in (None, "done") else ""))  # fmt: skip
    dash = s.get("dashboard") or {}
    if (dash.get("funnel") or dash.get("caddy")) and dash.get("access") != "public":
        how = (
            "Tailscale Funnel is on: sudo tailscale funnel reset"
            if dash.get("funnel")
            else "the caddy profile runs"
        )
        text = f"PUBLIC ({how}) although QP_DASHBOARD_ACCESS=tailscale"
    elif not dash.get("installed"):
        text = "Tailscale is not installed (bootstrap-oracle.sh installs it)"
    elif dash.get("serving"):
        text = f"tailnet only: https://{dash.get('name') or '?'} (Tailscale {dash.get('state')})"
    else:
        text = f"not published on the tailnet yet (./qp tailscale); Tailscale {dash.get('state')}"
    lines.append(f"  dashboard   {text}")
    if s.get("preflight"):
        lines.append("  PREFLIGHT   FAILED: " + "; ".join(s["preflight"]))
    w = s["watchdog"]
    lines.append(f"  watchdog    {'on' if w['enabled'] else 'OFF'} · {w['restarts_6h']} restart(s) in 6 h"
                 + (f" · last check {(w.get('last') or {}).get('verdict')}" if w.get("last") else ""))  # fmt: skip
    h = s.get("health")
    if h:
        lines.append("")
        lines.append(
            f"HEALTH: {str(h.get('status')).upper()}   (checked {str(h.get('checked_at'))[:19]} UTC)"
        )
        for name, p in (h.get("parts") or {}).items():
            lines.append(f"  {p.get('status', '?'):>8}  {name:<15} {p.get('detail', '')}")
        for blocker in h.get("order_blockers") or []:
            lines.append(f"  NEW BRAIN ORDERS HELD: {blocker}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ops = Ops(Path(__file__).resolve().parent)
    cmd, args = (argv[0] if argv else "help"), argv[1:]
    try:
        if cmd == "watchdog":
            print(ops.watchdog())
        elif cmd == "auto-update":
            print(ops.auto_update())
        elif cmd == "deploy":
            print(ops.deploy_now(args[0] if args else None))
        elif cmd == "ci-gate":
            sha = args[0] if args else ops.fetch(ops.branch())
            gate = ops.ci_gate(ops.git("rev-parse", sha), ops.branch())
            print(json.dumps(asdict(gate), indent=1))
            return 0 if gate.ok else 1
        elif cmd == "backup":
            kind = args[0] if args else "manual"
            with ops.lock(wait=True):
                record = ops.backup(kind)
            print(json.dumps(record, indent=1))
        elif cmd == "restore-test":
            print(json.dumps(ops.restore_test(Path(args[0]) if args else None), indent=1))
        elif cmd == "status":
            data = ops.status()
            print(json.dumps(data, indent=1, default=str) if "--json" in args else render_status(data))
        elif cmd == "sample":
            print(json.dumps(ops.sample()))
        elif cmd == "host-preflight":
            problems = ops.host_preflight()
            for problem in problems:
                print(f"FAIL  {problem}")
            print("server preflight: " + ("FAILED — nothing is started" if problems else "PASS"))
            return 1 if problems else 0
        elif cmd == "record-start":
            ops.record_start()
        elif cmd == "pin":
            ops.pin(ops.deployed())
            print("automatic deploys paused until ./qp update")
        else:
            print(__doc__)
            return 0 if cmd in ("help", "-h", "--help") else 2
    except OpsError as exc:
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
