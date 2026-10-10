"""The server's backup and restore test, for real: pg_dump and pg_restore against PostgreSQL 16, through the same
code the server runs (deploy/qpops.py). Object Storage is a local fake that keeps what it receives.

Runs when ``QP_TEST_POSTGRES_URL`` points at a PostgreSQL server. ``QP_TEST_PG_TOOLS=docker`` (CI) runs the
tools inside postgres:16-alpine — the production backup container's image; otherwise the host's own tools.
"""

import base64
import hashlib
import importlib.util
import os
import shutil
import sys
import urllib.parse
from pathlib import Path

import pytest

from quantpulse.db import migrate

ROOT = Path(__file__).resolve().parents[2]
PG_URL = os.environ.get("QP_TEST_POSTGRES_URL", "")
MODE = os.environ.get("QP_TEST_PG_TOOLS", "local")
PAR = "https://objectstorage.test/p/SECRET/n/ns/b/backups/o/"

pytestmark = [
    pytest.mark.skipif(not PG_URL, reason="needs PostgreSQL (QP_TEST_POSTGRES_URL)"),
    pytest.mark.skipif(
        MODE == "local" and not shutil.which("pg_dump"), reason="needs pg_dump and pg_restore"
    ),
]

_spec = importlib.util.spec_from_file_location("qpops_real", ROOT / "deploy" / "qpops.py")
assert _spec and _spec.loader
qpops = importlib.util.module_from_spec(_spec)
sys.modules["qpops_real"] = qpops
_spec.loader.exec_module(qpops)


def pg_env(database: str) -> dict[str, str]:
    u = urllib.parse.urlsplit(PG_URL.replace("+asyncpg", ""))
    env = {"PGHOST": u.hostname or "localhost", "PGPORT": str(u.port or 5432), "PGUSER": u.username or "postgres",
           "PGDATABASE": database}  # fmt: skip
    if u.password:
        env["PGPASSWORD"] = u.password
    return env


class PgHost(qpops.Host):
    """``docker compose exec backup …`` → the PostgreSQL tools against the test server."""

    def __init__(self, backups: Path, database: str) -> None:
        self.backups, self.database = backups, database

    def run(self, args, *, env=None, timeout=600, check=True, input_text=None):
        assert args[:2] == ["docker", "compose"] and args[4:6] == ["exec", "-T"], args
        rest, extra = list(args[6:]), {}
        while rest[0] == "-e":
            key, value = rest[1].split("=", 1)
            extra[key] = value
            rest = rest[2:]
        assert rest[0] == "backup", rest  # only the backup container's tools are used
        return self.tools(rest[1:], extra, check=check, timeout=timeout)

    def tools(self, cmd, extra=None, *, check=True, timeout=600):
        env = {**pg_env(self.database), **(extra or {})}
        if MODE == "docker":
            flags = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
            full = ["docker", "run", "--rm", "--network", "host", "-v", f"{self.backups}:/backups", *flags,
                    "postgres:16-alpine", *cmd]  # fmt: skip
            return super().run(full, check=check, timeout=timeout)
        local = [c.replace("/backups", str(self.backups)) for c in cmd]
        return super().run(local, env=env, check=check, timeout=timeout)


class ObjectStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.alerts: list[str] = []

    def urlopen(self, req, timeout=None):
        from tests.unit.test_qpops import Response

        if req.full_url.startswith(PAR):
            data = req.data.read()
            md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
            assert req.get_header("Content-md5") == md5
            self.objects[req.full_url[len(PAR) :]] = data
            return Response(200, b"", {"opc-content-md5": md5})
        self.alerts.append(req.get_header("Title") or req.full_url)
        return Response(200)


@pytest.fixture
def server(tmp_path):
    source = f"qp_backup_src_{os.getpid()}"
    deploy = tmp_path / "deploy"
    (deploy / "backups").mkdir(parents=True)
    (deploy / "backups").chmod(0o777)  # the container's root user writes here
    (deploy / ".env").write_text("QP_ALERT_NTFY_URL=https://ntfy.sh/test-topic\n")
    (deploy / "ops.env").write_text(f"QP_BACKUP_PAR_URL={PAR}\n")
    host = PgHost(deploy / "backups", source)
    host.tools(["dropdb", "--if-exists", source])
    host.tools(["createdb", source])
    url = PG_URL.rstrip("/") + "/" + source
    migrate.upgrade(url)  # the real schema, at the head
    host.tools(["psql", "-v", "ON_ERROR_STOP=1", "-c",
                "INSERT INTO brain_state (key, value, updated_at) VALUES "
                "('backup-probe', '{\"kept\": 42}', now())"])  # fmt: skip
    storage = ObjectStorage()
    ops = qpops.Ops(deploy, host=host, urlopen=storage.urlopen, sleep=lambda s: None)
    yield ops, host, storage, source
    for db in (source, "qp_restore_check", qpops.RESTORE_DB):
        host.tools(["dropdb", "--if-exists", db], check=False)


def test_a_nightly_backup_restores_with_every_row(server):
    ops, host, storage, _ = server
    record = ops.backup("nightly")
    assert record["verified"] and record["tables"] > 30
    dump = ops.dir / "backups" / record["file"]
    obj = record["upload"]["object"]
    assert storage.objects[obj] == dump.read_bytes()  # what left the server is exactly the dump

    result = ops.restore_test()
    assert result["ok"] and result["rev"] == migrate.head_revision()
    assert result["tables"] == result["live_tables"] and int(result["tables"]) > 30
    listed = host.tools(
        ["psql", "-tAc", f"SELECT count(*) FROM pg_database WHERE datname = '{qpops.RESTORE_DB}'"]
    )
    assert listed.stdout.strip() == "0"  # the scratch database is gone again

    # the data itself: restored into a database of its own, the probe row is there
    host.tools(["createdb", "qp_restore_check"])
    host.tools(
        ["pg_restore", "--exit-on-error", "--no-owner", "-d", "qp_restore_check", f"/backups/{dump.name}"]
    )
    row = host.tools(["psql", "-d", "qp_restore_check", "-tAc",
                      "SELECT value FROM brain_state WHERE key = 'backup-probe'"])  # fmt: skip
    assert '"kept": 42' in row.stdout


def test_a_damaged_backup_fails_the_restore_test_and_alerts(server):
    ops, _, storage, _ = server
    record = ops.backup("manual")
    good = (ops.dir / "backups" / record["file"]).read_bytes()
    damaged = ops.dir / "backups" / "quantpulse-manual-29990101-000000.dump"
    damaged.write_bytes(good[: len(good) // 3])
    with pytest.raises(qpops.OpsError):
        ops.restore_test(damaged)
    assert any("restore test FAILED" in a for a in storage.alerts)
    assert ops.load("backup.json")["restore_test"]["ok"] is False


def test_the_backup_never_holds_a_database_password(server):
    ops, host, *_ = server
    role = f"qp_probe_{os.getpid()}"
    host.tools(["psql", "-c", f"DROP ROLE IF EXISTS {role}"])
    host.tools(["psql", "-c", f"CREATE ROLE {role} LOGIN PASSWORD 'probe-secret-8841'"])
    try:
        record = ops.backup("manual")
        host.tools(
            ["pg_restore", "-f", "/backups/as.sql", f"/backups/{record['file']}"]
        )  # the dump as SQL text
        sql = (ops.dir / "backups" / "as.sql").read_text()
        assert "brain_state" in sql  # it is the whole database ...
        assert (
            "probe-secret-8841" not in sql and "CREATE ROLE" not in sql
        )  # ... but never a role or a password
    finally:
        host.tools(["psql", "-c", f"DROP ROLE IF EXISTS {role}"], check=False)
