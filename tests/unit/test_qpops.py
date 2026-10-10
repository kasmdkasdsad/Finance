"""The server's operations (deploy/qpops.py), against a simulated server: git, Docker, the QuantPulse API, GitHub
and Object Storage are fakes — nothing here runs a container, reaches GitHub or touches a broker.

* the CI gate deploys a commit only when every required job (ARM64 included) passed on exactly that commit;
* a deploy builds and checks before it switches, verifies after, and rolls back (schema included) by itself;
* the watchdog restarts a stalled supervisor, gracefully and rate-limited — and can never cause an order: its only
  request to QuantPulse is one GET;
* backups are verified, uploaded write-only with an integrity check, and a failure alerts.
"""

import importlib.util
import io
import json
import re
import subprocess
import sys
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("qpops", ROOT / "deploy" / "qpops.py")
assert _spec and _spec.loader
qpops = importlib.util.module_from_spec(_spec)
sys.modules["qpops"] = qpops
_spec.loader.exec_module(qpops)

OLD, NEW, BAD = "a" * 40, "b" * 40, "c" * 40
COMPOSE = (ROOT / "deploy" / "compose.yaml").read_text()
REPO = "kasmdkasdsad/Finance"
BRANCH = "claude/keen-tesla-y7rion"
NTFY = "https://ntfy.sh/qp-secret-topic"
PAR = "https://objectstorage.eu-frankfurt-1.oraclecloud.com/p/SECRET-PAR-TOKEN/n/ns/b/qp-backups/o/"
TOKEN = "t" * 64
# a Saturday: outside market hours, so automatic deploys may run
SATURDAY = datetime(2026, 10, 3, 15, 0, tzinfo=UTC)


class Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def now(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        self.at += timedelta(seconds=seconds)


class Response:
    def __init__(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.status, self._body, self.headers = status, body, headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def job(name: str, conclusion: str = "success") -> dict:
    return {"name": name, "status": "completed", "conclusion": conclusion}


GREEN_JOBS = [job(n) for n in qpops.REQUIRED_JOBS]


def run(sha: str, *, number: int = 1, status: str = "completed", conclusion: str = "success",
        branch: str = BRANCH) -> dict:  # fmt: skip
    return {"id": 1000 + number, "run_number": number, "run_attempt": 1, "head_sha": sha, "event": "push",
            "path": ".github/workflows/ci.yml", "head_branch": branch, "status": status, "conclusion": conclusion,
            "html_url": f"https://github.com/{REPO}/actions/runs/{1000 + number}"}  # fmt: skip


class Server:
    """A simulated VM: the git checkout, Docker images and containers, the database schema, the QuantPulse API,
    GitHub's CI results and Object Storage. Records every command and every HTTP request."""

    def __init__(self, deploy_dir: Path, clock: Clock) -> None:
        self.dir, self.clock = deploy_dir, clock
        self.head, self.remote = OLD, OLD
        self.images = {"current": OLD, OLD[:12]: OLD}
        self.running: str | None = OLD  # the commit of the image the api container runs
        self.schema = {OLD: "0024", NEW: "0025", BAD: "0025"}
        self.db_schema = "0024"
        self.broken = {BAD}  # versions whose supervisor never comes up
        self.preflight_ok, self.build_ok, self.downgrade_ok = True, True, True
        self.ci: dict[str, tuple[list[dict], list[dict]]] = {}
        self.verdict = "ok"  # what the running API's supervisor reports
        self.api_status = 200
        self.api_answers = True
        self.started_at = clock.now() - timedelta(hours=2)
        self.container_state = "running"
        self.commands: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.requests: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.uploads: dict[str, bytes] = {}
        self.upload_status = 200
        self.dump_tables = 12
        self.restarts = 0
        self.restore_values = "rev=0024\ntables=40\nlive_tables=40\nrows=1234\n"
        self.restore_rc = 0
        self.dirty = ""  # `git status --porcelain` of the checkout
        self.compose_of: dict[str, str] = {}  # a commit's deploy/compose.yaml (default: the real one)
        self.tailscale = {"BackendState": "Running", "Self": {"DNSName": "quantpulse.tail1234.ts.net."}}
        self.serve = "https://quantpulse.tail1234.ts.net (tailnet only)\n|-- / proxy http://127.0.0.1:8501\n"
        self.serve_json: dict = {
            "TCP": {"443": {"HTTPS": True}},
            "Web": {
                "quantpulse.tail1234.ts.net:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8501"}}}
            },
        }
        self.up_fails = False  # `docker compose up` fails: the old container keeps running

    # -------------------------------------------------------------- commands
    def respond(self, a: list[str], env: dict[str, str]) -> tuple[int, str, str]:
        if a[:1] == ["git"]:
            return self._git(a[3:])
        if a[:2] == ["docker", "compose"]:
            return self._compose(a[4:], env)
        if a[:1] == ["docker"]:
            return self._docker(a[1:])
        if a[:3] == ["tailscale", "status", "--json"]:
            return 0, json.dumps(self.tailscale), ""
        if a[:4] == ["tailscale", "serve", "status", "--json"]:
            return 0, json.dumps(self.serve_json), ""
        if a[:3] == ["tailscale", "serve", "status"]:
            return 0, self.serve, ""
        raise AssertionError(f"unexpected command {a}")

    def _git(self, a: list[str]) -> tuple[int, str, str]:
        if a == ["rev-parse", "HEAD"]:
            return 0, self.head + "\n", ""
        if a[0] == "fetch":
            return 0, "", ""
        if a[0] == "rev-parse" and a[1].startswith("refs/remotes/origin/"):
            return 0, self.remote + "\n", ""
        if a[0] == "rev-parse":
            return 0, a[1] + "\n", ""
        if a[:2] == ["merge-base", "--is-ancestor"]:
            order = [OLD, NEW, BAD]
            return (0 if order.index(a[2]) <= order.index(a[3]) else 1), "", ""
        if a[:3] == ["checkout", "--quiet", "--detach"]:
            self.head = a[3]
            return 0, "", ""
        if a[:3] == ["remote", "get-url", "origin"]:
            return 0, f"https://github.com/{REPO}.git\n", ""
        if a[:2] == ["status", "--porcelain"]:
            return 0, self.dirty, ""
        if a[0] == "show" and a[1].endswith(":deploy/compose.yaml"):  # a version's compose file
            return 0, self.compose_of.get(a[1].split(":")[0], COMPOSE), ""
        raise AssertionError(f"unexpected git {a}")

    def _compose(self, a: list[str], env: dict[str, str]) -> tuple[int, str, str]:
        while a[:1] == ["--profile"]:  # a compose profile (the reader's) changes nothing here
            a = a[2:]
        if a[:2] == ["exec", "-T"] and "backup" in a and "sh" in a:
            name = next(x.split("=", 1)[1] for x in a if x.startswith("QP_NAME="))
            script = a[-1]
            if "pg_dump" in script:
                (self.dir / "backups").mkdir(exist_ok=True)
                (self.dir / "backups" / name).write_bytes(b"PGDMP" + name.encode() * 50)
                return 0, f"tables={self.dump_tables}\n", ""
            if "pg_restore" in script:
                return (
                    self.restore_rc,
                    self.restore_values,
                    "pg_restore: error: bad archive" if self.restore_rc else "",
                )
        if a[:3] == ["exec", "-T", "backup"]:  # dropdb after a failed restore test, a restore
            return 0, "", ""
        if a[:3] == ["exec", "-T", "db"]:
            if "pg_isready" in a:
                return 0, "accepting connections", ""
            if any("alembic_version" in x for x in a):
                return 0, self.db_schema + "\n", ""
            if any("pg_database_size" in x for x in a):
                return 0, "412 MB\n", ""
        if a[:4] == ["run", "--rm", "--no-deps", "-T"]:
            if "quantpulse-preflight" in a:
                assert env.get("QP_IMAGE_TAG"), "the preflight must run on the new image"
                return (0 if self.preflight_ok else 1), "", "" if self.preflight_ok else "FAIL paper_endpoint"
            if "quantpulse-migrate" in a:
                if self.downgrade_ok:
                    self.db_schema = a[a.index("quantpulse-migrate") + 1]
                    return 0, "", ""
                return 1, "", "downgrade failed"
        if a[:2] == ["up", "-d"]:
            if self.up_fails:
                return 1, "", "Error response from daemon: no space left on device"
            self.running = self.images["current"]
            self.started_at = self.clock.now()
            self.db_schema = self.schema[self.running]  # the new version migrates at start-up
            self.container_state = "running"
            return 0, "", ""
        if a[0] == "stop":
            self.running, self.container_state = None, "exited"
            return 0, "", ""
        if a[:2] == ["restart", "api"]:
            self.restarts += 1
            self.started_at = self.clock.now()
            return 0, "", ""
        if a[:4] == ["ps", "--all", "--format", "json"]:
            if a[-1] == "api":
                return (
                    0,
                    json.dumps({"ID": "cid", "Service": "api", "State": self.container_state}) + "\n",
                    "",
                )
            rows = [
                {"Service": s, "State": "running", "Health": "healthy"} for s in ("db", "api", "dashboard")
            ]
            return 0, "\n".join(json.dumps(r) for r in rows), ""
        raise AssertionError(f"unexpected compose {a}")

    def _docker(self, a: list[str]) -> tuple[int, str, str]:
        if a[:2] == ["image", "inspect"]:
            return (0 if a[2].split(":")[1] in self.images else 1), "", ""
        if a[0] == "tag":
            self.images[a[2].split(":")[1]] = self.images[a[1].split(":")[1]]
            return 0, "", ""
        if a[:2] == ["image", "ls"]:
            return 0, "\n".join(self.images) + "\n", ""
        if a[:2] in (["image", "rm"], ["image", "prune"]):
            if a[1] == "rm":
                self.images.pop(a[2].split(":")[1], None)
            return 0, "", ""
        if a[0] == "inspect":
            started = self.started_at.strftime("%Y-%m-%dT%H:%M:%S.123456789Z")
            return 0, json.dumps({"Status": self.container_state, "StartedAt": started}), ""
        raise AssertionError(f"unexpected docker {a}")

    # -------------------------------------------------------------- HTTP
    def urlopen(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        data = req.data.read() if hasattr(req.data, "read") else req.data
        self.requests.append((method, url, dict(req.header_items()), data))
        if url.startswith(qpops.API):
            return self._api(url[len(qpops.API) :], req)
        if url.startswith("https://api.github.com/"):
            return self._github(url)
        if url.startswith(PAR.rstrip("/")):
            if self.upload_status >= 400:
                raise urllib.error.HTTPError(url, self.upload_status, "error", {}, io.BytesIO())  # type: ignore[arg-type]
            import base64
            import hashlib

            self.uploads[url[len(PAR) :]] = data
            md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
            return Response(200, b"", {"opc-content-md5": md5})
        return Response(200)  # ntfy, the webhook, a heartbeat

    def _api(self, path: str, req):
        if not self.api_answers or self.running is None:
            raise urllib.error.URLError("connection refused")
        if self.api_status != 200:
            raise urllib.error.HTTPError(req.full_url, self.api_status, "error", {}, io.BytesIO())  # type: ignore[arg-type]
        if path == qpops.WATCHDOG_PATH:
            verdict = "blocked" if self.running in self.broken else self.verdict
            body = {"verdict": verdict, "reason": "test", "restart": verdict == "stalled", "leader": True,
                    "commit": self.running, "started_at": self.started_at.isoformat(), "last_tick_at": None,
                    "last_result": "idle", "last_cycle": {"id": 7, "kind": "full", "status": "completed",
                                                          "started_at": self.started_at.isoformat()},
                    "alert_heartbeat": {"at": self.clock.now().isoformat(), "healthy": True, "delivered": True,
                                        "last_healthy_delivered_at": self.clock.now().isoformat()}}  # fmt: skip
            return Response(200, json.dumps(body).encode())
        if path == "/health":
            return Response(200, b'{"status":"ok"}')
        if path.startswith("/api/v1/system/health"):
            body = {"status": "ok", "checked_at": self.clock.now().isoformat(),
                    "parts": {"database": {"status": "ok", "detail": "answers"}}, "order_blockers": []}  # fmt: skip
            return Response(200, json.dumps(body).encode())
        raise AssertionError(f"unexpected API path {path}")

    def _github(self, url: str):
        sha = re.search(r"head_sha=([0-9a-f]+)", url)
        if sha:
            runs, _ = self.ci.get(sha.group(1), ([], []))
            return Response(200, json.dumps({"workflow_runs": runs}).encode())
        run_id = int(re.search(r"/runs/(\d+)/jobs", url).group(1))  # type: ignore[union-attr]
        for runs, jobs in self.ci.values():
            if any(r["id"] == run_id for r in runs):
                return Response(200, json.dumps({"jobs": jobs}).encode())
        return Response(200, json.dumps({"jobs": []}).encode())

    # -------------------------------------------------------------- views
    def compose_calls(self) -> list[list[str]]:
        return [c[4:] for c in self.commands if c[:2] == ["docker", "compose"]]

    def api_requests(self) -> list[tuple[str, str]]:
        return [(m, u[len(qpops.API) :]) for m, u, _, _ in self.requests if u.startswith(qpops.API)]


class FakeHost(qpops.Host):
    def __init__(self, server: Server) -> None:
        self.s = server
        self.builds: list[tuple[list[str], list[str]]] = []

    def run(self, args, *, env=None, timeout=600, check=True, input_text=None):
        self.s.commands.append(list(args))
        self.s.envs.append(dict(env or {}))
        rc, out, err = self.s.respond(list(args), dict(env or {}))
        if check and rc != 0:
            raise qpops.OpsError(f"{' '.join(args[:4])} … failed ({rc}): {err}")
        return subprocess.CompletedProcess(args, rc, out, err)

    def pipe(self, first, second, *, timeout=3600):
        self.builds.append((first, second))
        if not self.s.build_ok:
            raise qpops.OpsError("the image build failed: pip could not install")
        tag = second[second.index("-t") + 1].split(":")[1]
        self.s.images[tag] = first[-1]


@pytest.fixture
def world(tmp_path):
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / ".env").write_text(
        f"QP_API_TOKEN={TOKEN}\nQP_ALERT_NTFY_URL={NTFY}\nPOSTGRES_PASSWORD=pw\nQP_ALPACA_PAPER=true\n"
    )
    (deploy / "compose.yaml").write_text(COMPOSE)
    (deploy / "ops.env").write_text(
        f"QP_DEPLOY_REPO={REPO}\nQP_DEPLOY_BRANCH={BRANCH}\nQP_BACKUP_PAR_URL={PAR}\n"
    )
    clock = Clock(SATURDAY)
    server = Server(deploy, clock)
    ops = qpops.Ops(deploy, host=FakeHost(server), urlopen=server.urlopen, now=clock.now, sleep=clock.sleep)
    return ops, server, clock


def alerts(server: Server) -> list[str]:
    return [h.get("Title", "") for m, u, h, _ in server.requests if u == NTFY]


# ================================================================================================ the CI gate
def test_the_gate_passes_only_a_commit_whose_every_required_job_passed(world):
    ops, server, _ = world
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    gate = ops.ci_gate(NEW, BRANCH)
    assert gate.ok and gate.state == "passed" and "ARM64" in gate.reason
    assert all(m == "GET" for m, *_ in server.requests)  # reading CI results only


@pytest.mark.parametrize(
    ("runs", "jobs", "state", "why"),
    [
        ([], [], "pending", "no CI run for this commit yet"),
        ([run(NEW, status="in_progress", conclusion=None)], [], "pending", "still running"),
        ([run(NEW, conclusion="failure")], GREEN_JOBS, "failed", "'failure'"),
        ([run(NEW, conclusion="cancelled")], GREEN_JOBS, "failed", "'cancelled'"),
        (
            [run(NEW)],
            [j for j in GREEN_JOBS if j["name"] != "docker (arm64)"],
            "failed",
            "'docker (arm64)' did not",
        ),
        ([run(NEW)], [*GREEN_JOBS[:-1], job("docker (arm64)", "failure")], "failed", "docker (arm64)"),
        ([run(NEW)], [*GREEN_JOBS[:-1], job("docker (arm64)", "skipped")], "failed", "docker (arm64)"),
        ([run(NEW)], [*GREEN_JOBS, job("extra", "failure")], "failed", "extra"),
        # the same commit on another branch says nothing about this one
        ([run(NEW, branch="main")], GREEN_JOBS, "pending", "no CI run"),
        # a re-run still going makes the answer ambiguous: wait
        (
            [run(NEW, number=1), run(NEW, number=2, status="queued", conclusion=None)],
            GREEN_JOBS,
            "pending",
            "running",
        ),
    ],
)
def test_anything_short_of_a_full_pass_is_not_deployed(world, runs, jobs, state, why):
    ops, server, _ = world
    server.ci[NEW] = (runs, jobs)
    gate = ops.ci_gate(NEW, BRANCH)
    assert not gate.ok and gate.state == state and why in gate.reason, gate


def test_a_later_passing_run_of_the_same_commit_counts(world):
    ops, server, _ = world
    server.ci[NEW] = ([run(NEW, number=1, conclusion="failure"), run(NEW, number=2)], GREEN_JOBS)
    assert ops.ci_gate(NEW, BRANCH).ok


def test_github_unreachable_or_refusing_is_never_a_pass(world):
    ops, _, _ = world

    def refused(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "rate limited", {}, io.BytesIO())  # type: ignore[arg-type]

    ops._urlopen = refused
    gate = ops.ci_gate(NEW, BRANCH)
    assert (gate.ok, gate.state) == (False, "unknown") and "403" in gate.reason

    def down(req, timeout=None):
        raise urllib.error.URLError("no route")

    ops._urlopen = down
    assert ops.ci_gate(NEW, BRANCH).state == "unknown"
    assert ops.ci_gate("b" * 12, BRANCH).state == "unknown"  # never a short or guessed commit id


def test_the_required_jobs_are_exactly_the_ci_workflows_jobs():
    """Keep the gate in step with .github/workflows/ci.yml: a job added there must be required here."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    names = set()
    for job_id, spec in workflow["jobs"].items():
        name = spec.get("name", job_id)
        matrix = (spec.get("strategy") or {}).get("matrix") or {}
        if matrix:
            ((_, values),) = matrix.items()
            names |= {f"{name} ({v})" for v in values}
        else:
            names.add(name)
    assert names == set(qpops.REQUIRED_JOBS)
    arm = next(s for s in workflow["jobs"].values() if s.get("name") == "docker (arm64)")
    assert "linux/arm64" in json.dumps(arm) and arm["needs"] == "test"
    assert workflow.get("on", workflow.get(True)) == {"push": {"branches": ["main", "claude/**"]}}


# ================================================================================================ deploys
def test_a_verified_commit_is_built_checked_switched_and_verified(world):
    ops, server, _ = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    assert ops.auto_update() == f"deployed {NEW[:12]}"
    calls = server.compose_calls()
    # before the switch: a backup, the build of exactly that commit, the preflight on the new image
    backup = next(i for i, c in enumerate(calls) if c[:2] == ["exec", "-T"] and "backup" in c)
    preflight = next(i for i, c in enumerate(calls) if "quantpulse-preflight" in c)
    switch = next(i for i, c in enumerate(calls) if c[:2] == ["up", "-d"])
    assert backup < preflight < switch
    (first, second), = ops.host.builds  # fmt: skip
    assert first[-3:] == ["archive", "--format=tar", NEW] and f"QP_GIT_COMMIT={NEW}" in second
    assert server.head == NEW and server.running == NEW and server.images["previous"] == OLD
    state = ops.load("deploy.json")
    assert state["phase"] == "done" and state["deployed"] == NEW and state["gate"]["state"] == "passed"
    assert any("Deployed" in t for t in alerts(server))


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (lambda s, o: None, "up to date"),
        (lambda s, o: setattr(s, "remote", NEW), "no CI run for this commit yet"),
        (
            lambda s, o: (
                setattr(s, "remote", NEW),
                s.ci.update({NEW: ([run(NEW, conclusion="failure")], [])}),
            ),
            "CI concluded 'failure'",
        ),
        (
            lambda s, o: (
                setattr(s, "remote", NEW),
                s.ci.update({NEW: ([run(NEW)], GREEN_JOBS)}),
                setattr(s, "verdict", "blocked"),
            ),
            "not healthy",
        ),
        (
            lambda s, o: (
                setattr(s, "remote", NEW),
                (o.state_dir / "pin").parent.mkdir(parents=True, exist_ok=True),
                (o.state_dir / "pin").write_text(OLD),
            ),
            "pinned by a rollback",
        ),
        (
            lambda s, o: (
                setattr(s, "remote", NEW),
                o.state_dir.mkdir(parents=True, exist_ok=True),
                (o.state_dir / "stopped").touch(),
            ),
            "stopped by a person",
        ),
    ],
)
def test_nothing_is_deployed_unless_everything_is_right(world, setup, expected):
    ops, server, _ = world
    setup(server, ops)
    out = ops.auto_update()
    assert expected in out, out
    assert not ops.host.builds and not [c for c in server.compose_calls() if c[:2] == ["up", "-d"]]
    assert server.head == OLD and server.running in (OLD, None)


def test_no_automatic_deploy_in_market_hours_but_a_person_may_deploy_a_verified_commit(world):
    ops, server, clock = world
    clock.at = datetime(2026, 9, 30, 15, 0, tzinfo=UTC)  # Wednesday 11:00 New York
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    assert "market hours" in ops.auto_update() and server.head == OLD
    assert ops.deploy_now() == f"deployed {NEW[:12]}"
    server.ci[BAD] = ([run(BAD, conclusion="failure")], [])
    server.remote = BAD
    with pytest.raises(qpops.OpsError, match="only commits whose CI passed are deployed"):
        ops.deploy_now()


def test_a_failed_ci_is_alerted_once(world):
    ops, server, clock = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW, conclusion="failure")], GREEN_JOBS)
    for _ in range(5):
        assert "CI concluded 'failure'" in ops.auto_update()
        clock.sleep(600)
    assert sum("CI did not pass" in t for t in alerts(server)) == 1 and server.head == OLD


def test_a_version_that_does_not_come_up_is_rolled_back_schema_included(world):
    ops, server, _ = world
    server.remote = BAD
    server.ci[BAD] = ([run(BAD)], GREEN_JOBS)
    out = ops.auto_update()
    assert out.startswith(f"rolled back to {OLD[:12]}"), out
    migrate = [c for c in server.compose_calls() if "quantpulse-migrate" in c]
    assert migrate and migrate[0][-2:] == [
        "0024",
        "--downgrade",
    ]  # undone by the new image, before the old starts
    assert server.head == OLD and server.running == OLD and server.db_schema == "0024"
    state = ops.load("deploy.json")
    assert state["phase"] == "rolled_back" and BAD in state["failed"]
    assert any("rolled back" in t for t in alerts(server))
    # the failed commit is not tried again automatically
    before = len(server.commands)
    assert "failed to deploy before" in ops.auto_update()
    assert not [c for c in server.commands[before:] if c[:2] == ["docker", "compose"]]


def test_a_downgrade_that_fails_restores_the_backup_taken_before_the_deploy(world):
    ops, server, _ = world
    server.remote = BAD
    server.ci[BAD] = ([run(BAD)], GREEN_JOBS)
    server.downgrade_ok = False
    ops.auto_update()
    restores = [c for c in server.compose_calls() if "pg_restore" in c and "--clean" in c]
    dump = ops.load("deploy.json")["predeploy_dump"]
    assert restores and restores[0][-1] == f"/backups/{dump}" and dump.startswith("quantpulse-predeploy-")


@pytest.mark.parametrize("broken", ["build_ok", "preflight_ok"])
def test_a_failure_before_the_switch_changes_nothing(world, broken):
    ops, server, _ = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    setattr(server, broken, False)
    assert ops.auto_update().startswith("not deployed")
    assert server.head == OLD and server.running == OLD and server.images["current"] == OLD
    assert not [c for c in server.compose_calls() if c[:2] in (["up", "-d"], ["stop", "api"])]
    assert ops.load("deploy.json")["attempts"][NEW] == 1 and NEW not in ops.load("deploy.json").get(
        "failed", {}
    )
    ops.auto_update()
    ops.auto_update()
    assert NEW in ops.load("deploy.json")["failed"]  # three strikes: a person looks


def test_a_deploy_cut_off_by_a_reboot_is_finished_or_undone(world):
    ops, server, clock = world
    ops.save("deploy.json", {"phase": "verifying", "target": BAD, "previous": OLD, "previous_schema": "0024",
                             "started_at": (clock.now() - timedelta(hours=1)).isoformat()})  # fmt: skip
    server.head, server.running, server.db_schema = BAD, BAD, "0025"
    server.images.update(current=BAD, previous=OLD)
    assert ops.auto_update().startswith(f"rolled back to {OLD[:12]}")
    assert server.running == OLD and server.db_schema == "0024"


def test_the_deploy_path_never_asks_quantpulse_for_more_than_a_read(world):
    ops, server, _ = world
    server.remote = BAD
    server.ci[BAD] = ([run(BAD)], GREEN_JOBS)
    ops.auto_update()
    assert server.api_requests() and all(r == ("GET", qpops.WATCHDOG_PATH) for r in server.api_requests())


# ================================================================================================ the watchdog
def test_a_healthy_supervisor_is_left_alone(world):
    ops, server, clock = world
    for _ in range(10):
        clock.sleep(60)
        assert ops.watchdog().startswith("ok")
    assert server.restarts == 0 and not alerts(server)


def test_a_stalled_supervisor_is_restarted_gracefully_and_rate_limited(world):
    ops, server, clock = world
    server.verdict = "stalled"
    clock.sleep(60)
    assert ops.watchdog().startswith("stalled")  # once could be a moment: confirm on the next check
    assert server.restarts == 0
    clock.sleep(60)
    assert ops.watchdog().startswith("restarted")
    assert server.restarts == 1 and ["restart", "api"] in server.compose_calls()
    assert any("Watchdog restarted the API" in t for t in alerts(server))
    for _ in range(3):
        clock.sleep(6 * 60)  # past the start-up grace each time
        ops.watchdog()
        clock.sleep(60)
        ops.watchdog()
    assert server.restarts == 3  # at most 3 in 6 hours
    assert any("gave up" in t for t in alerts(server))


def test_an_api_that_stops_answering_is_restarted_after_three_minutes(world):
    ops, server, clock = world
    server.api_answers = False
    for minute in range(1, 4):
        clock.sleep(60)
        ops.watchdog()
        assert server.restarts == (1 if minute == 3 else 0)


@pytest.mark.parametrize("state", ["blocked", "standby", "paused", "not_applicable", "starting"])
def test_fail_closed_and_deliberate_states_are_never_restarted(world, state):
    ops, server, clock = world
    server.verdict = state
    for _ in range(10):
        clock.sleep(60)
        ops.watchdog()
    assert server.restarts == 0


def test_an_answer_it_cannot_read_is_alive_not_dead(world):
    ops, server, clock = world
    server.api_status = 401  # a wrong token, or an older version without the endpoint (404)
    for _ in range(5):
        clock.sleep(60)
        ops.watchdog()
    assert server.restarts == 0 and sum("cannot read" in t for t in alerts(server)) == 1


def test_a_stopped_or_just_started_api_is_left_alone(world):
    ops, server, clock = world
    server.api_answers = False
    server.container_state = "exited"  # a person stopped it, or Docker is restarting it
    for _ in range(5):
        clock.sleep(60)
        ops.watchdog()
    server.container_state = "running"
    server.started_at = clock.now()  # starting up: 5 minutes' grace
    for _ in range(4):
        clock.sleep(60)
        assert "less than 5 minutes" in ops.watchdog()
    assert server.restarts == 0


def test_the_watchdog_cannot_cause_an_order(world):
    """Whatever the API answers — healthy, stalled, silent, refusing — the watchdog makes one kind of request to
    QuantPulse (GET /api/v1/system/watchdog) and one kind of change (restart the api container gracefully)."""
    ops, server, clock = world
    scenarios = [("verdict", "ok"), ("verdict", "stalled"), ("verdict", "blocked"), ("api_status", 500),
                 ("api_status", 401), ("api_answers", False)]  # fmt: skip
    for attr, value in scenarios:
        setattr(server, attr, value)
        for _ in range(4):
            clock.sleep(6 * 60)
            ops.watchdog()
        server.verdict, server.api_status, server.api_answers = "ok", 200, True
    assert server.restarts > 0  # the scenarios did drive restarts
    assert set(server.api_requests()) == {("GET", qpops.WATCHDOG_PATH)}
    others = {u for m, u, _, _ in server.requests if not u.startswith(qpops.API)}
    assert others <= {NTFY}  # alerts only: never GitHub, never Alpaca
    allowed = {("ps", "--all"), ("restart", "api")}
    for c in server.commands:
        if c[:2] == ["docker", "compose"]:
            assert tuple(c[4:6]) in allowed, c
        else:
            assert c[:3] == ["docker", "inspect", "--format"], c
    for path in ("/api/v1/brain/kill-switch", "/api/v1/brain/run", "/api/v1/trading/run", "/v2/orders"):
        with pytest.raises(qpops.OpsError, match="not a read-only endpoint"):
            ops.api_get(path)


def test_the_operations_code_has_no_way_to_write_to_quantpulse_or_a_broker():
    source = (ROOT / "deploy" / "qpops.py").read_text()
    # no broker at all: no Alpaca host to call (the live-endpoint rule is a regex), no order path
    assert "alpaca.markets" not in source and "APCA-" not in source and "/v2/" not in source
    assert "/orders" not in source
    # it never reads the Alpaca keys (only the QP_ALPACA_PAPER flag, for the server preflight)
    for key in ("QP_ALPACA_API_KEY_ID", "QP_ALPACA_API_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        assert key not in source
    assert "kill-switch" not in source and "/brain/" not in source and "/trading/" not in source
    # every request to the API goes through api_get, which is GET-only
    api_requests = re.findall(r"Request\(\s*API \+ path, method=\"(\w+)\"", source)
    assert api_requests == ["GET"]
    assert source.count("API +") == 1


# ================================================================================================ backups
def test_the_nightly_backup_is_verified_and_uploaded_write_only(world, capsys):
    ops, server, clock = world
    record = ops.backup("nightly")
    assert record["verified"] and record["tables"] == 12
    obj = record["upload"]["object"]
    assert obj.startswith("daily/quantpulse-nightly-") and obj in server.uploads
    local = (ops.dir / "backups" / record["file"]).read_bytes()
    assert server.uploads[obj] == local  # the bytes that arrived are the bytes dumped
    put = next(r for r in server.requests if r[0] == "PUT")
    assert put[2]["Content-md5"] == record["upload"]["md5"]  # Object Storage checks the MD5 on arrival
    assert ops.load("backup.json")["last_ok"]["file"] == record["file"]
    assert "SECRET-PAR-TOKEN" not in capsys.readouterr().out + json.dumps(ops.load("backup.json"))
    clock.at = datetime(2026, 11, 1, 7, 15, tzinfo=UTC)  # the 1st of a month: kept longer (bucket rules)
    assert ops.backup("nightly")["upload"]["object"].startswith("monthly/")
    clock.at = datetime(2026, 10, 4, 7, 15, tzinfo=UTC)  # a Sunday
    assert ops.backup("nightly")["upload"]["object"].startswith("weekly/")


@pytest.mark.parametrize("failure", ["upload", "empty"])
def test_a_failed_backup_alerts_and_fails_the_heartbeat(world, failure, capsys):
    ops, server, _ = world
    ops.cfg["QP_BACKUP_HEARTBEAT_URL"] = "https://hc-ping.com/backup-uuid"
    if failure == "upload":
        server.upload_status = 503
    else:
        server.dump_tables = 0
    with pytest.raises(qpops.OpsError):
        ops.backup("nightly")
    assert any("Database backup FAILED" in t for t in alerts(server))
    assert any(u == "https://hc-ping.com/backup-uuid/fail" for _, u, _, _ in server.requests)
    assert ops.load("backup.json")["last_error"]["error"]
    assert "SECRET-PAR-TOKEN" not in capsys.readouterr().out + json.dumps(ops.load("backup.json"))


def test_the_restore_test_checks_the_restored_database_and_alerts_when_it_fails(world):
    ops, server, _ = world
    ops.backup("manual")
    result = ops.restore_test()
    assert result["ok"] and result["rev"] == "0024" and result["tables"] == result["live_tables"] == "40"
    server.restore_rc, server.restore_values = 1, "rev=\ntables=3\nlive_tables=40\n"
    with pytest.raises(qpops.OpsError, match="the restore failed"):
        ops.restore_test()
    assert any("restore test FAILED" in t for t in alerts(server))
    assert ops.load("backup.json")["restore_test"]["ok"] is False


# ================================================================================================ status
def test_the_status_shows_the_whole_server(world):
    ops, server, _ = world
    server.ci[OLD] = ([run(OLD)], GREEN_JOBS)
    ops.backup("nightly")
    ops.restore_test()
    data = ops.status()
    text = qpops.render_status(data)
    for needle in ("CPU", "memory", "disk", "idle check", "Oracle's 20%", "docker      db running (healthy)",
                   "postgres    accepting connections · 412 MB · schema 0024", "/health 200",
                   "supervisor  OK", "last cycle  #7 full completed", "heartbeat   last healthy ping delivered",
                   "backup      last", "restore test", "OK", f"deployed    {OLD[:12]}", "CI passed",
                   "watchdog    on", "HEALTH: OK"):  # fmt: skip
        assert needle in text, (needle, text)
    assert TOKEN not in text and "SECRET-PAR-TOKEN" not in json.dumps(data, default=str)


def test_the_idle_check_reads_seven_days_of_samples(world):
    ops, _, clock = world
    for _ in range(30):
        ops.sample()
        clock.sleep(60)
    idle = ops.idle_check()
    assert idle["samples"] == 30 and idle["threshold"] == 20.0 and isinstance(idle["at_risk"], bool)


def test_the_watchdog_warns_once_when_oracle_could_see_the_vm_as_idle(world):
    ops, server, clock = world
    ops.state_dir.mkdir(parents=True, exist_ok=True)
    start = clock.now().timestamp()

    def samples(cpu: float, mem: float) -> None:
        lines = [
            json.dumps({"t": start - 60 * i, "cpu": cpu, "mem": mem}) for i in range(qpops.IDLE_MIN_SAMPLES)
        ]
        (ops.state_dir / "samples.jsonl").write_text("\n".join(lines) + "\n")

    samples(cpu=3.0, mem=8.0)  # a quiet day: all under 20%
    for _ in range(3):
        ops.watchdog()
    assert sum("Oracle may stop this VM as idle" in t for t in alerts(server)) == 1  # once, not every minute
    assert ops.idle_check()["at_risk"]
    samples(cpu=3.0, mem=31.0)  # memory above 20%: not idle by Oracle's rule
    assert not ops.idle_check()["at_risk"]


def test_a_switch_that_did_not_happen_is_not_mistaken_for_a_deploy(world):
    """`docker compose up` failed, so the old version still answers "ok": the new commit must be the one answering."""
    ops, server, _ = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    server.up_fails = True
    out = ops.auto_update()
    assert out.startswith("rolled back"), out
    assert "the old version still answers" in ops.load("deploy.json")["failed"][NEW]


def test_local_changes_in_the_checkout_stop_a_deploy_before_anything_changes(world):
    ops, server, _ = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    server.dirty = " M deploy/compose.yaml\n"
    assert "local changes" in ops.auto_update()
    assert not ops.host.builds and server.head == OLD and server.running == OLD
    assert not [c for c in server.compose_calls() if c[:2] == ["exec", "-T"]]  # not even a backup


def test_update_never_goes_backwards(world):
    ops, server, _ = world
    server.head, server.remote = NEW, NEW
    server.ci[OLD] = ([run(OLD)], GREEN_JOBS)
    with pytest.raises(qpops.OpsError, match=r"use \./qp rollback"):
        ops.deploy_now(OLD)


# ================================================================================================ server preflight
LIVE = "https://api.alpaca.markets"


def test_the_shipped_files_pass_the_server_preflight(tmp_path):
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / "compose.yaml").write_text(COMPOSE)
    (deploy / ".env").write_text(
        (ROOT / "deploy" / "cloud.env.example").read_text()
    )  # what ./qp setup starts from
    (deploy / "ops.env").write_text((ROOT / "deploy" / "ops.env.example").read_text())
    assert qpops.Ops(deploy).host_preflight() == []


@pytest.mark.parametrize(
    ("where", "line", "why"),
    [
        (
            ".env",
            f"QP_ALPACA_BASE_URL={LIVE}",
            "deploy/.env: QP_ALPACA_BASE_URL names Alpaca's live-money API",
        ),
        (
            ".env",
            "APCA_API_BASE_URL=https://broker-api.alpaca.markets/v1",
            "APCA_API_BASE_URL names Alpaca's live",
        ),
        (
            "ops.env",
            f"ALPACA_ENDPOINT={LIVE}",
            "deploy/ops.env: ALPACA_ENDPOINT names Alpaca's live-money API",
        ),
        (".env", "QP_ALPACA_PAPER=false", "QP_ALPACA_PAPER must be exactly true"),
        (".env", "QP_ALPACA_PAPER=yes", "QP_ALPACA_PAPER must be exactly true"),
        (".env", "QP_DEPLOYMENT=local", "QP_DEPLOYMENT=local is ambiguous"),
        ("ops.env", "QP_DASHBOARD_ACCESS=everyone", "QP_DASHBOARD_ACCESS=everyone"),
    ],
)
def test_the_server_preflight_refuses_anything_live_or_ambiguous(world, where, line, why):
    ops, _, _ = world
    path = ops.dir / where
    text = (
        path.read_text().replace("QP_ALPACA_PAPER=true\n", "")
        if "QP_ALPACA_PAPER" in line
        else path.read_text()
    )
    path.write_text(text + line + "\n")
    problems = qpops.Ops(ops.dir).host_preflight()
    assert any(why in p for p in problems), problems
    assert all(LIVE not in p and "broker-api" not in p for p in problems)  # names variables, never values


def test_the_server_preflight_reads_the_servers_own_environment_and_compose_file(world, monkeypatch):
    ops, _, _ = world
    monkeypatch.setenv("APCA_API_BASE_URL", LIVE)
    assert any("environment variable APCA_API_BASE_URL" in p for p in ops.host_preflight())
    monkeypatch.delenv("APCA_API_BASE_URL")
    (ops.dir / "compose.yaml").write_text(
        COMPOSE.replace('QP_ALPACA_PAPER: "true"', 'QP_ALPACA_PAPER: "false"')
    )
    assert any("compose.yaml does not force" in p for p in ops.host_preflight())
    (ops.dir / "compose.yaml").write_text(COMPOSE.replace("QP_DEPLOYMENT: cloud", "QP_DEPLOYMENT: local"))
    assert any("compose.yaml does not force" in p for p in ops.host_preflight())
    (ops.dir / "compose.yaml").write_text(COMPOSE)
    (ops.dir / ".env").write_text((ops.dir / ".env").read_text() + f"# never use {LIVE} here\n")
    assert ops.host_preflight() == []  # a comment configures nothing


def test_a_failed_server_preflight_deploys_nothing_and_alerts_once(world):
    ops, server, clock = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    (ops.dir / "ops.env").write_text((ops.dir / "ops.env").read_text() + f"ALPACA_BASE_URL={LIVE}\n")
    ops.cfg = qpops.read_env(ops.dir / "ops.env")
    for _ in range(3):
        assert ops.auto_update().startswith("server preflight failed")
        clock.sleep(600)
    assert sum("Server preflight FAILED" in t for t in alerts(server)) == 1
    assert not ops.host.builds and server.head == OLD and not server.compose_calls()
    with pytest.raises(qpops.OpsError, match="server preflight failed"):
        ops.deploy_now()


def test_a_new_version_that_stops_forcing_paper_is_never_switched_to(world):
    ops, server, _ = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    server.compose_of[NEW] = COMPOSE.replace('QP_ALPACA_PAPER: "true"', 'QP_ALPACA_PAPER: "false"')
    out = ops.auto_update()
    assert out.startswith("not deployed") and "does not force" in out, out
    assert server.head == OLD and server.running == OLD and not ops.host.builds


# ================================================================================================ dashboard access
def test_the_status_shows_the_dashboard_on_the_tailnet_only(world, monkeypatch):
    ops, server, _ = world
    monkeypatch.setattr(qpops.shutil, "which", lambda name: f"/usr/bin/{name}")
    text = qpops.render_status(ops.status())
    assert "dashboard   tailnet only: https://quantpulse.tail1234.ts.net (Tailscale Running)" in text
    assert "PREFLIGHT" not in text
    server.serve = "https://quantpulse.tail1234.ts.net (Funnel on)\n|-- / proxy http://127.0.0.1:8501\n"
    text = qpops.render_status(ops.status())
    assert "PUBLIC (Tailscale Funnel is on" in text
    server.serve = ""
    assert "not published on the tailnet yet" in qpops.render_status(ops.status())


def test_qp_runs_the_server_preflight_first_and_keeps_the_dashboard_on_tailscale():
    qp = (ROOT / "deploy" / "qp").read_text()
    start = qp[qp.index("  start)") : qp.index("  status)")]
    assert start.index("host-preflight") < start.index("dc build")  # before anything is built or started
    assert 'access:-tailscale}" == "public"' in start  # --public needs QP_DASHBOARD_ACCESS=public
    assert "tailscale serve --bg --https=443 http://127.0.0.1:8501" in qp  # tailnet only, never Funnel
    assert "funnel" not in qp.lower()


# ================================================================================================ the reader
READ_KEY = "r" * 64


def with_reader(world, published: bool = True, funnel: bool = False):
    """``./qp reader on`` has run: the read-only key in deploy/.env (and published on the tailnet)."""
    ops, server, clock = world
    env = ops.dir / ".env"
    env.write_text(env.read_text() + f"QP_API_READ_TOKEN={READ_KEY}\n")
    if published:
        host = "quantpulse.tail1234.ts.net:8443"
        server.serve_json["TCP"]["8443"] = {"HTTPS": True}
        server.serve_json["Web"][host] = {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8090"}}}
        if funnel:
            server.serve_json["AllowFunnel"] = {host: True}
    return qpops.Ops(ops.dir, host=ops.host, urlopen=server.urlopen, now=clock.now, sleep=clock.sleep)


def test_the_status_shows_the_reader_off_by_default_and_on_the_tailnet_only_when_on(world, monkeypatch):
    ops, server, _ = world
    monkeypatch.setattr(qpops.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert "reader      off (./qp reader on" in qpops.render_status(ops.status())
    on = with_reader(world)
    text = qpops.render_status(on.status())
    assert "reader      on: https://quantpulse.tail1234.ts.net:8443 (tailnet only;" in text
    assert READ_KEY not in text and READ_KEY not in json.dumps(on.status())  # never shown
    server.serve_json["AllowFunnel"] = {"quantpulse.tail1234.ts.net:8443": True}
    assert "reader      PUBLIC (Tailscale Funnel on port 8443)" in qpops.render_status(on.status())
    server.serve_json = {}
    assert "the reader is not published on the tailnet" in qpops.render_status(on.status())


def test_a_deploy_or_a_rollback_keeps_the_reader_on_the_new_version(world):
    ops = with_reader(world)
    _, server, _ = world
    server.remote = NEW
    server.ci[NEW] = ([run(NEW)], GREEN_JOBS)
    assert ops.auto_update() == f"deployed {NEW[:12]}"
    switch = next(c for c in server.compose_calls() if "up" in c)
    assert switch[:2] == ["--profile", "reader"] and switch[-3:] == ["api", "dashboard", "reader"]
    assert not world[0].reader_on  # without the read-only key: neither named nor its profile used


def test_a_rollback_brings_the_reader_back_too(world):
    ops = with_reader(world)
    _, server, _ = world
    server.remote = BAD  # a version that never comes up: it is rolled back
    server.ci[BAD] = ([run(BAD)], GREEN_JOBS)
    assert ops.auto_update().startswith(f"rolled back to {OLD[:12]}")
    ups = [c for c in server.compose_calls() if "up" in c]
    assert all(c[:2] == ["--profile", "reader"] for c in ups) and len(ups) == 2


def test_qp_reader_publishes_on_the_tailnet_only_and_off_revokes_the_key():
    qp = (ROOT / "deploy" / "qp").read_text()
    reader = qp[qp.index("  reader)") : qp.index("  down)")]
    assert (
        "sudo tailscale serve --bg --https=8443 http://127.0.0.1:8090" in reader
    )  # the tailnet, never public
    assert "openssl rand -hex 32" in reader  # a 64-character key, made on the server
    off = reader[reader.index("      off)") : reader.index("      key)")]
    assert (
        off.index("serve --https=8443 off")
        < off.index('set_var QP_API_READ_TOKEN ""')
        < off.index("--force-recreate api")
    )  # unpublished, the key removed, and the API restarted without it
