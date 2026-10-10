"""render.yaml keeps its safety properties: paper only, the Brain owns the account, one API instance, secrets
never in the file, migrations and the preflight before every deploy, a private database, deploys only after CI."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1].parent
BRANCH = "claude/keen-tesla-y7rion"


def blueprint() -> dict:
    return yaml.safe_load((ROOT / "render.yaml").read_text())


def service(name: str) -> dict:
    return next(s for s in blueprint()["services"] if s["name"] == name)


def env(svc: dict) -> dict[str, dict]:
    return {e["key"]: e for e in svc["envVars"]}


def test_the_api_runs_the_brain_on_paper_only():
    e = env(service("quantpulse-api"))
    fixed = {k: v.get("value") for k, v in e.items() if "value" in v}
    assert fixed["QP_DEPLOYMENT"] == "cloud" and fixed["QP_ALPACA_PAPER"] == "true"
    assert fixed["QP_BRAIN_MODE"] == "paper_execution"
    assert fixed["QP_ALPACA_TRADING_ENABLED"] == "true" and fixed["QP_TRADING_DRY_RUN"] == "false"
    assert fixed["QP_BRAIN_KILL_SWITCH"] == "false" and fixed["QP_TRADING_SCHEDULER_ENABLED"] == "false"
    assert fixed["QP_TRADING_SCHEDULER_REQUIRES_ARMING"] == "true" and fixed["QP_AUTO_MIGRATE"] == "false"
    text = (ROOT / "render.yaml").read_text()
    assert not re.search(r"(?<![\w.-])(api|broker-api)\.alpaca\.markets", text)  # no live endpoint anywhere
    for risky in ("QP_TRADING_MAX_", "QP_TRADING_REQUIRE_LIVE_DATA", "QP_TRADING_MIN_"):
        assert risky not in text  # the protected controls keep their shipped values


def test_secrets_are_never_in_the_file():
    e = env(service("quantpulse-api"))
    for key in ("QP_ALPACA_API_KEY_ID", "QP_ALPACA_API_SECRET_KEY"):
        assert e[key] == {"key": key, "sync": False}  # asked for once in the Render dashboard
    assert e["QP_API_TOKEN"] == {"key": "QP_API_TOKEN", "generateValue": True}
    assert e["QP_DATABASE_URL"]["fromDatabase"] == {"name": "quantpulse-db", "property": "connectionString"}
    d = env(service("quantpulse-dashboard"))
    assert d["QP_DASHBOARD_PASSWORD"] == {"key": "QP_DASHBOARD_PASSWORD", "sync": False}
    assert d["QP_API_TOKEN"]["fromService"]["envVarKey"] == "QP_API_TOKEN"


def test_one_api_instance_with_room_for_the_stock_model_and_a_graceful_handover():
    api = service("quantpulse-api")
    assert api["numInstances"] == 1 and api["plan"] == "1c-2g"  # the S&P 500 model peaks near 1.5 GB
    assert api["healthCheckPath"] == "/health"
    assert api["maxShutdownDelaySeconds"] >= 100  # the drain (75 s) plus uvicorn's own 20 s fit
    assert "--port $PORT" in api["startCommand"] and "--timeout-graceful-shutdown 20" in api["startCommand"]
    assert api["preDeployCommand"] == "quantpulse-preflight && quantpulse-migrate"


def test_deploys_follow_the_branch_only_after_ci_passes():
    for s in blueprint()["services"]:
        assert s["branch"] == BRANCH and s["autoDeployTrigger"] == "checksPass", s["name"]
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    on = ci.get("on", ci.get(True))
    assert "claude/**" in on["push"]["branches"]  # so Render sees checks on this branch
    assert set(on) == {"push"}  # one run per push: a pull_request trigger would run the PR's pushes twice
    assert {"test", "postgres"} <= set(ci["jobs"])


def test_the_dashboard_gets_no_broker_key_and_talks_to_the_api_privately():
    d = service("quantpulse-dashboard")
    keys = set(env(d))
    assert not {k for k in keys if "ALPACA" in k or "DATABASE" in k}
    assert env(d)["QP_API_URL"]["fromService"] == {
        "type": "web",
        "name": "quantpulse-api",
        "property": "hostport",
    }
    assert env(d)["STREAMLIT_CLIENT_SHOW_ERROR_DETAILS"]["value"] == "false"
    assert d["plan"] == "free" and "$PORT" in d["startCommand"]


def test_the_database_is_private_postgres_16_in_the_same_region():
    db = blueprint()["databases"][0]
    assert db["postgresMajorVersion"] == "16" and db["ipAllowList"] == []  # no internet access at all
    regions = {s["region"] for s in blueprint()["services"]} | {db["region"]}
    assert len(regions) == 1  # the private network needs one region


def test_render_pins_the_python_the_suite_runs_on():
    assert (ROOT / ".python-version").read_text().strip() == "3.11"
