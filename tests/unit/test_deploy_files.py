"""The cloud deployment files keep their safety properties: paper forced, nothing public but the optional HTTPS
proxy, the dashboard without broker keys, no secret in any example or image."""

import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy"


def compose() -> dict:
    return yaml.safe_load((DEPLOY / "compose.yaml").read_text())


def env_of(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    return env if isinstance(env, dict) else dict(item.split("=", 1) for item in env)


def test_the_api_is_forced_into_cloud_paper_mode_on_postgres():
    api = compose()["services"]["api"]
    env = env_of(api)
    assert env["QP_DEPLOYMENT"] == "cloud" and env["QP_ALPACA_PAPER"] == "true"
    assert (
        env["QP_DATABASE_URL"].startswith("postgresql+asyncpg://")
        and "${POSTGRES_PASSWORD}" in env["QP_DATABASE_URL"]
    )
    assert env["QP_LOG_DIR"] == "/app/logs" and api["restart"] == "unless-stopped"


def test_nothing_but_the_optional_https_proxy_listens_publicly():
    services = compose()["services"]
    for name, svc in services.items():
        for port in svc.get("ports") or []:
            if name == "caddy":
                assert port in ("80:80", "443:443")
            else:
                assert str(port).startswith("127.0.0.1:"), (name, port)
    assert services["caddy"]["profiles"] == ["public"]  # only with ./qp start --public
    assert "8000" not in (DEPLOY / "Caddyfile").read_text()  # the API is never proxied


def test_the_dashboard_gets_no_broker_key_and_no_database_password():
    dashboard = compose()["services"]["dashboard"]
    assert "env_file" not in dashboard
    env = env_of(dashboard)
    assert set(env) <= {"QP_DEPLOYMENT", "QP_API_URL", "QP_API_TOKEN", "QP_DASHBOARD_PASSWORD_HASH"} | {
        k for k in env if k.startswith("STREAMLIT_")
    }
    assert env["QP_DEPLOYMENT"] == "cloud" and env["STREAMLIT_CLIENT_SHOW_ERROR_DETAILS"] == "false"


def test_every_service_restarts_by_itself_and_logs_rotate():
    for name, svc in compose()["services"].items():
        assert svc["restart"] == "unless-stopped", name
        assert svc["logging"]["options"]["max-size"], name


def test_the_cloud_env_example_holds_no_secret_and_stays_paper():
    text = (DEPLOY / "cloud.env.example").read_text()
    values = dict(re.findall(r"^([A-Z0-9_]+)=(.*)$", text, re.MULTILINE))
    for name in ("POSTGRES_PASSWORD", "QP_API_TOKEN", "QP_DASHBOARD_PASSWORD_HASH", "QP_ALPACA_API_KEY_ID",
                 "QP_ALPACA_API_SECRET_KEY", "QP_ALERT_NTFY_URL", "QP_ALERT_WEBHOOK_URL", "QP_HEARTBEAT_URL"):  # fmt: skip
        assert values[name] == "", name
    assert values["QP_ALPACA_PAPER"] == "true"
    assert not re.search(r"(?<![\w.-])(api|broker-api)\.alpaca\.markets", text)


def test_the_image_never_contains_secrets_or_the_deploy_folder():
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    assert {".env", ".env.*", "deploy", "data", "*.db", "backups"} <= set(ignored)
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert not re.search(r"^\s*(COPY|ADD)\s+[^\n]*(\.env|deploy)", dockerfile, re.MULTILINE)
    assert "USER quantpulse" in dockerfile


def test_the_helper_scripts_are_strict_and_executable():
    for script in ("qp", "bootstrap.sh"):
        path = DEPLOY / script
        assert os.access(path, os.X_OK), script
        assert "set -euo pipefail" in path.read_text(), script
    qp = (DEPLOY / "qp").read_text()
    # secrets are read without echo, and travel in the environment, never on a command line
    assert 'read -r -s -p "Alpaca PAPER secret key' in qp and 'QP_VALUE="$2"' in qp
    assert "echo $QP_API_TOKEN" not in qp and 'echo "$secret"' not in qp


# --------------------------------------------------------------------------- Oracle Cloud Always Free (ARM)
def test_the_oracle_bootstrap_keeps_the_server_private_and_self_maintaining():
    path = DEPLOY / "bootstrap-oracle.sh"
    assert os.access(path, os.X_OK) and os.access(DEPLOY / "qpops.py", os.X_OK)
    text = path.read_text()
    assert "set -euo pipefail" in text
    # Oracle's own iptables rules are kept (ufw would fight them and Docker)
    assert "ufw --force enable" not in text and "ufw allow" not in text
    assert "--dport (8000|8501)" in text  # refuses a rule that would expose the API or the dashboard
    assert "PasswordAuthentication no" in text and "PermitRootLogin no" in text
    # security updates install themselves; a reboot they need happens outside US market hours
    assert 'Automatic-Reboot "true"' in text and 'Automatic-Reboot-Time "07:40"' in text
    assert "systemctl enable --now containerd docker" in text and '"shutdown-timeout": 180' in text
    assert "/swapfile" in text and "vm.swappiness = 10" in text
    assert "docker-compose-plugin" in text and "dpkg --print-architecture" in text


def test_the_backup_container_holds_the_database_password_and_nothing_else():
    backup = compose()["services"]["backup"]
    assert "env_file" not in backup and "ports" not in backup
    env = env_of(backup)
    assert set(env) == {"PGHOST", "PGUSER", "PGDATABASE", "PGPASSWORD"}
    assert "pg_dump" not in " ".join(backup["command"])  # idle: the host's timer runs the backups through it


def test_the_api_stops_gracefully_and_runs_the_deployed_image():
    services = compose()["services"]
    api = services["api"]
    assert api["image"] == services["dashboard"]["image"] == "quantpulse:${QP_IMAGE_TAG:-current}"
    seconds = int(str(api["stop_grace_period"]).rstrip("s"))
    assert seconds >= 75 + 15 + 30  # the drain, the final reconciliation, uvicorn's own graceful shutdown
    db = " ".join(services["db"]["command"])
    assert "shared_buffers=" in db and "effective_cache_size=" in db
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert 'ARG QP_GIT_COMMIT=""' in dockerfile and "ENV QP_GIT_COMMIT=${QP_GIT_COMMIT}" in dockerfile


def test_the_server_settings_hold_no_secret_and_reach_no_container():
    text = (DEPLOY / "ops.env.example").read_text()
    values = dict(re.findall(r"^([A-Z0-9_]+)=(.*)$", text, re.MULTILINE))
    for name in ("GITHUB_TOKEN", "QP_BACKUP_PAR_URL", "QP_BACKUP_HEARTBEAT_URL", "QP_OPS_NTFY_URL"):
        assert values[name] == "", name
    assert values["QP_AUTO_UPDATE_WINDOW"] == "closed" and values["QP_WATCHDOG"] == "true"
    assert values["QP_DASHBOARD_ACCESS"] == "tailscale"  # the dashboard is reached through Tailscale only
    assert "ops.env" not in (DEPLOY / "compose.yaml").read_text()
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert {"deploy/*.env", "!deploy/*.env.example", "deploy/state/", "backups/"} <= set(ignored)


def test_the_timers_run_the_helper_as_the_owner_never_as_root():
    units = {p.name: p.read_text() for p in (DEPLOY / "systemd").iterdir()}
    commands = {"watchdog": "watchdog", "update": "auto-update", "backup": "backup nightly",
                "restore-test": "restore-test"}  # fmt: skip
    for name, command in commands.items():
        service, timer = units[f"quantpulse-{name}.service"], units[f"quantpulse-{name}.timer"]
        assert f"ExecStart=@DEPLOY@/qp {command}\n" in service and "User=@USER@" in service
        assert "Requires=docker.service" in service and "WantedBy=timers.target" in timer
    assert "OnUnitActiveSec=1min" in units["quantpulse-watchdog.timer"]
    assert "OnCalendar=*:0/10" in units["quantpulse-update.timer"]
    assert "Persistent=true" in units["quantpulse-backup.timer"]
    assert len(units) == 2 * len(commands)


FAKE_DOCKER = r'''#!/usr/bin/env python3
"""A stand-in for `docker compose` that interpolates the compose file like Docker does: every ${VAR:?} must
have a value, from the environment first, then from the .env next to the compose file."""
import os, re, sys
args = sys.argv[1:]
if args[:1] != ["compose"]:
    sys.exit(0)
compose_file = args[args.index("-f") + 1]
env_file = os.path.join(os.path.dirname(compose_file), ".env")
values = {}
if os.path.exists(env_file):
    for line in open(env_file):
        if "=" in line and not line.startswith("#"):
            k, v = line.rstrip("\n").split("=", 1)
            values[k] = v
for name in re.findall(r"\$\{(\w+):\?", open(compose_file).read()):
    if not os.environ.get(name, values.get(name, "")):
        sys.stderr.write(f"error while interpolating: required variable {name} is missing a value\n")
        sys.exit(1)
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(" ".join(args[3:]) + "\n")
if "quantpulse-hash-password" in args:
    password = sys.stdin.readline().rstrip("\n")
    assert len(password) >= 12, "the password reached the hasher"
    print("pbkdf2_sha256:600000:c2FsdHNhbHRzYWx0:ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGk")
'''


def test_qp_setup_works_on_a_fresh_server(tmp_path):
    """The first ./qp setup on a new server: the image is built before the dashboard password exists, and the
    compose file requires that hash, so the build must not need it (a placeholder for that command only)."""
    import shutil
    import subprocess

    deploy = tmp_path / "repo" / "deploy"
    deploy.mkdir(parents=True)
    for name in ("qp", "compose.yaml", "cloud.env.example", "ops.env.example"):
        shutil.copy2(DEPLOY / name, deploy / name)
    git = ["git", "-C", str(tmp_path / "repo"), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "claude/test-branch"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "x"], check=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    log = tmp_path / "docker.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith("QP_")}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", FAKE_DOCKER_LOG=str(log))
    typed = "PKTESTKEY123\nsecret-not-shown\nlong enough password\nlong enough password\n"
    run = subprocess.run(["bash", str(deploy / "qp"), "setup"], input=typed, text=True, capture_output=True,
                         env=env, timeout=60)  # fmt: skip
    assert run.returncode == 0, run.stderr
    written = (deploy / ".env").read_text()
    assert re.search(r"^QP_DASHBOARD_PASSWORD_HASH=pbkdf2_sha256:600000:", written, re.M)
    assert re.search(r"^QP_ALPACA_API_KEY_ID=PKTESTKEY123$", written, re.M)
    # neither the placeholder nor the password itself ends up in the file
    assert "not-set-yet" not in written and "long enough password" not in written
    assert re.search(r"^POSTGRES_PASSWORD=[0-9a-f]{48}$", written, re.M)
    assert re.search(r"^QP_API_TOKEN=[0-9a-f]{64}$", written, re.M)
    assert re.search(r"^QP_DEPLOY_BRANCH=claude/test-branch$", (deploy / "ops.env").read_text(), re.M)
    assert oct((deploy / ".env").stat().st_mode & 0o777) == "0o600"
    calls = log.read_text().splitlines()
    assert calls[0] == "build api" and "quantpulse-hash-password" in calls[1]
    assert "secret-not-shown" not in run.stdout + run.stderr and "PKTESTKEY123" not in run.stdout + run.stderr
    # run again: everything is already set, nothing is asked, and it still works
    again = subprocess.run(["bash", str(deploy / "qp"), "setup"], input="", text=True, capture_output=True,
                           env=env, timeout=60)  # fmt: skip
    assert again.returncode == 0, again.stderr
    assert (deploy / ".env").read_text() == written
