"""The cloud preflight: paper only, secured, PostgreSQL, protected controls intact — and it never prints a secret."""

import os
import subprocess
import sys

import pytest

from quantpulse.config import Settings
from quantpulse.core.passwords import (
    ITERATIONS,
    PasswordHashError,
    hash_password,
    parse,
    verify_password,
)
from quantpulse.services import preflight

KEY = "PKtest-cloud-key-0001"
SECRET = "cloud-test-secret-value-000000000000001"
TOKEN = "t" * 20 + "0123456789abcdefghij"  # 40 characters
DB_PASSWORD = "db-password-never-shown"
DB = f"postgresql+asyncpg://quantpulse:{DB_PASSWORD}@db:5432/quantpulse"
HASH = hash_password("correct horse battery staple", iterations=200_000)
ENV = {"QP_ALPACA_PAPER": "true"}


def cloud(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        deployment="cloud",
        alpaca_api_key_id=KEY,
        alpaca_api_secret_key=SECRET,
        api_token=TOKEN,
        dashboard_password_hash=HASH,
        database_url=DB,
    )
    base.update(overrides)
    return Settings(**base)


def failed(report: preflight.Report) -> set[str]:
    return {c.name for c in report.failures}


def assert_no_secret(text: str) -> None:
    for secret in (SECRET, TOKEN, DB_PASSWORD, HASH, KEY):
        assert secret not in text


# --------------------------------------------------------------------------- passing
def test_a_complete_cloud_configuration_passes_and_shows_no_secret():
    report = preflight.run(cloud(), ENV)
    assert report.ok, report.lines()
    names = {c.name for c in report.checks}
    assert {"paper_setting", "no_live_endpoint", "paper_endpoint_variables", "paper_key", "paper_client",
            "data_endpoint", "api_token", "dashboard_password", "database", "protected_controls"} <= names  # fmt: skip
    text = "\n".join(report.lines()) + repr(report.as_dict())
    assert_no_secret(text)
    assert "quantpulse:***@db:5432" in text and "https://paper-api.alpaca.markets" in text
    enforced = preflight.enforce(cloud(), ENV)
    assert enforced is not None and enforced.ok


def test_local_deployments_are_not_enforced():
    assert preflight.enforce(Settings(_env_file=None), {}) is None


# --------------------------------------------------------------------------- paper only
def test_the_paper_setting_must_be_explicit():
    assert "paper_setting" in failed(preflight.run(cloud(), {}))
    assert "paper_setting" in failed(preflight.run(cloud(), {"QP_ALPACA_PAPER": "false"}))
    assert "paper_setting" not in failed(preflight.run(cloud(), {"QP_ALPACA_PAPER": "TRUE"}))
    with pytest.raises(ValueError, match="PAPER"):
        cloud(alpaca_paper=False)  # a live setting does not even load


@pytest.mark.parametrize(
    "env",
    [
        {"APCA_API_BASE_URL": "https://api.alpaca.markets"},
        {"ALPACA_BASE_URL": "https://api.alpaca.markets/v2"},
        {"SOMETHING_ELSE": "https://API.alpaca.markets"},
        {"QP_ALPACA_TRADING_URL": "https://broker-api.alpaca.markets"},
    ],
)
def test_any_live_alpaca_endpoint_anywhere_refuses_to_start(env):
    report = preflight.run(cloud(), {**ENV, **env})
    assert not report.ok and "no_live_endpoint" in failed(report)
    with pytest.raises(preflight.PreflightFailed, match="live-money"):
        preflight.enforce(cloud(), {**ENV, **env})


@pytest.mark.parametrize(
    ("env", "ok"),
    [
        ({"APCA_API_BASE_URL": "https://paper-api.alpaca.markets"}, True),
        ({"APCA_API_BASE_URL": "https://paper-api.alpaca.markets/v2/"}, True),
        ({"APCA_API_BASE_URL": "https://example.com/alpaca"}, False),
        ({"ALPACA_PAPER": "true"}, True),
        ({"ALPACA_PAPER": "false"}, False),
    ],
)
def test_endpoint_variables_other_tools_read_must_name_paper(env, ok):
    report = preflight.run(cloud(), {**ENV, **env})
    assert ("paper_endpoint_variables" not in failed(report)) is ok


def test_a_live_key_or_no_key_refuses():
    live = preflight.run(cloud(alpaca_api_key_id="AKtest-live-key-0001"), ENV)
    assert "paper_key" in failed(live) and "PAPER key" in next(c.detail for c in live.failures)
    none = preflight.run(cloud(alpaca_api_key_id=None, alpaca_api_secret_key=None), ENV)
    assert {"paper_key", "paper_client"} <= failed(none)


def test_the_keys_go_only_to_alpaca_market_data():
    assert "data_endpoint" in failed(preflight.run(cloud(alpaca_data_url="https://data.example.com"), ENV))


# --------------------------------------------------------------------------- access
@pytest.mark.parametrize("token", [None, "short-token", "change-me-" + "x" * 30, SECRET])
def test_a_strong_api_token_is_required(token):
    report = preflight.run(cloud(api_token=token), ENV)
    assert "api_token" in failed(report)
    assert_no_secret("\n".join(report.lines()))


@pytest.mark.parametrize("hashed", ["plain-password", "pbkdf2_sha256:1000:c2FsdHNhbHQ:" + "A" * 43])
def test_a_dashboard_password_hash_given_to_the_api_must_be_valid(hashed):
    assert "dashboard_password" in failed(preflight.run(cloud(dashboard_password_hash=hashed), ENV))


def test_the_api_service_does_not_need_the_dashboard_password():
    """On Render the dashboard is its own service and refuses to open without a password (tests/frontend)."""
    assert preflight.run(cloud(dashboard_password_hash=None), ENV).ok


# --------------------------------------------------------------------------- ownership
@pytest.mark.parametrize("mode", ["research_only", "dry_run", "paper_recommendation"])
def test_the_cloud_runs_the_brain_as_the_account_owner(mode):
    assert failed(preflight.run(cloud(brain_mode=mode), ENV)) == {"brain_mode"}


def test_one_scheduler_the_brain_supervisor():
    assert failed(preflight.run(cloud(trading_scheduler_enabled=True), ENV)) == {"one_scheduler"}


def test_a_credential_given_twice_with_different_values_is_ambiguous():
    same = preflight.run(cloud(), {**ENV, "APCA_API_KEY_ID": KEY})
    assert same.ok
    different = preflight.run(
        cloud(), {**ENV, "QP_ALPACA_API_KEY_ID": KEY, "APCA_API_KEY_ID": "PKsomething-else-0001"}
    )
    assert failed(different) == {"unambiguous_credentials"}
    assert_no_secret("\n".join(different.lines()))
    assert "PKsomething-else-0001" not in "\n".join(different.lines())


def test_postgres_is_required_in_the_cloud():
    report = preflight.run(cloud(database_url="sqlite+aiosqlite:///./data/quantpulse.db"), ENV)
    assert "database" in failed(report)


# --------------------------------------------------------------------------- protected controls
@pytest.mark.parametrize(
    "loose",
    [
        {"trading_require_live_data": False},
        {"trading_max_quote_age_seconds": 900},
        {"trading_max_spread_bps": 60},
        {"trading_max_daily_loss_pct": 0.1},
        {"trading_max_positions": 20},
        {"trading_max_order_notional": 50_000},
        {"trading_min_dollar_volume": 1_000_000},
        {"trading_scheduler_requires_arming": False},
    ],
)
def test_protected_controls_may_not_be_loosened_in_the_cloud(loose):
    report = preflight.run(cloud(**loose), ENV)
    assert failed(report) == {"protected_controls"}, report.lines()


def test_protected_controls_may_be_tightened():
    tight = dict(trading_max_quote_age_seconds=120, trading_max_spread_bps=15, trading_max_daily_loss_pct=0.02,
                 trading_max_positions=5, trading_min_dollar_volume=50_000_000)  # fmt: skip
    assert preflight.run(cloud(**tight), ENV).ok


def test_mask_url():
    assert preflight.mask_url(DB) == "postgresql+asyncpg://quantpulse:***@db:5432/quantpulse"
    assert preflight.mask_url("sqlite+aiosqlite:///x.db") == "sqlite+aiosqlite:///x.db"


# --------------------------------------------------------------------------- the command
def run_cli(tmp_path, args, env, stdin=None):
    empty = tmp_path / "empty.env"
    empty.write_text("")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("QP_", "APCA_", "ALPACA_"))}
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.argv[0] = 'qp'; from quantpulse.cli import {args[0]}; {args[0]}()",
         *args[1:]],
        env={**clean, "QP_ENV_FILE": str(empty), **env},
        input=stdin, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip


def cloud_env(**extra) -> dict[str, str]:
    return {
        "QP_DEPLOYMENT": "cloud",
        "QP_ALPACA_PAPER": "true",
        "QP_ALPACA_API_KEY_ID": KEY,
        "QP_ALPACA_API_SECRET_KEY": SECRET,
        "QP_API_TOKEN": TOKEN,
        "QP_DASHBOARD_PASSWORD_HASH": HASH,
        "QP_DATABASE_URL": DB,
        **extra,
    }


def test_preflight_command_exit_codes_and_masking(tmp_path):
    good = run_cli(tmp_path, ["run_preflight"], cloud_env())
    assert good.returncode == 0, good.stdout + good.stderr
    assert "PASS" in good.stdout
    assert_no_secret(good.stdout + good.stderr)
    bad = run_cli(tmp_path, ["run_preflight"], cloud_env(APCA_API_BASE_URL="https://api.alpaca.markets"))
    assert bad.returncode == 2 and "FAIL" in bad.stdout and "Nothing was started" in bad.stdout
    assert_no_secret(bad.stdout + bad.stderr)
    live = run_cli(tmp_path, ["run_preflight"], cloud_env(QP_ALPACA_PAPER="false"))
    assert live.returncode == 2 and "alpaca_paper" in live.stdout
    assert_no_secret(live.stdout + live.stderr)


def test_hash_password_command_reads_stdin_and_prints_only_the_hash(tmp_path):
    out = run_cli(tmp_path, ["run_hash_password"], {}, stdin="a long dashboard password\n")
    assert out.returncode == 0, out.stderr
    hashed = out.stdout.strip()
    assert "a long dashboard password" not in out.stdout + out.stderr
    assert verify_password("a long dashboard password", hashed)
    short = run_cli(tmp_path, ["run_hash_password"], {}, stdin="short\n")
    assert short.returncode != 0 and "at least" in short.stderr


# --------------------------------------------------------------------------- passwords
def test_password_hashes():
    hashed = hash_password("correct horse battery staple")
    assert hashed.startswith(f"pbkdf2_sha256:{ITERATIONS}:") and "correct" not in hashed
    assert "$" not in hashed and all(ch.isalnum() or ch in "-_:" for ch in hashed)  # safe unquoted in .env
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("correct horse battery stapl", hashed)
    assert hash_password("correct horse battery staple") != hashed  # a new salt each time
    assert not verify_password("anything", hashed[:-2] + "AA")
    assert not verify_password("anything", "garbage")
    with pytest.raises(PasswordHashError):
        hash_password("short")
    with pytest.raises(PasswordHashError, match="too weak"):
        parse(hash_password("correct horse battery staple", iterations=1000))
