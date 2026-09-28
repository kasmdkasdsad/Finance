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
