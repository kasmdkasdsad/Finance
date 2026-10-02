"""Which .env the settings come from, where each trading switch came from, and .env edits after startup."""

import pytest

from quantpulse import config as cfg
from quantpulse.config import Settings, env_file_drift, resolve_env_file, setting_sources

TRADING_ON = "QP_ALPACA_TRADING_ENABLED=true\nQP_TRADING_DRY_RUN=false\n"


def test_an_explicit_env_file_wins(tmp_path):
    f = tmp_path / "custom.env"
    f.write_text(TRADING_ON)
    assert resolve_env_file({"QP_ENV_FILE": str(f)}, cwd=tmp_path / "elsewhere") == f.resolve()


def test_the_project_root_env_is_found_from_any_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "Finance"
    root.mkdir()
    (root / ".env").write_text(TRADING_ON)
    monkeypatch.setattr(cfg, "project_root", lambda: root)
    other = tmp_path / "somewhere-else"
    other.mkdir()
    assert resolve_env_file({}, cwd=other) == root / ".env"


def test_the_working_directory_is_the_last_resort(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "project_root", lambda: None)
    assert resolve_env_file({}, cwd=tmp_path) is None
    (tmp_path / ".env").write_text(TRADING_ON)
    assert resolve_env_file({}, cwd=tmp_path) == (tmp_path / ".env").resolve()


def test_a_file_saved_with_a_byte_order_mark_still_loads(tmp_path):
    f = tmp_path / ".env"
    f.write_bytes(b"\xef\xbb\xbf" + TRADING_ON.encode())  # Windows Notepad's "UTF-8 with BOM"
    s = Settings(_env_file=f)
    assert s.alpaca_trading_enabled and not s.trading_dry_run


def test_sources_show_environment_overrides_and_never_secrets(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text(TRADING_ON + "QP_ALPACA_API_KEY_ID=PKSECRETKEY\nQP_ALPACA_API_SECRET_KEY=topsecret\n")
    monkeypatch.setenv("QP_TRADING_DRY_RUN", "true")  # e.g. left in the Windows user environment
    s = Settings(_env_file=f)
    s._source_file = f
    got = {x.variable: x for x in setting_sources(s)}
    assert got["QP_TRADING_DRY_RUN"].source == "environment" and got["QP_TRADING_DRY_RUN"].value == "true"
    assert got["QP_ALPACA_TRADING_ENABLED"].source == "env_file"
    assert got["QP_TRADING_KILL_SWITCH"].source == "default"
    assert got["QP_ALPACA_API_KEY_ID"].value == "set" and got["QP_ALPACA_API_SECRET_KEY"].value == "set"
    assert "PKSECRETKEY" not in repr(got) and "topsecret" not in repr(got)
    assert not s.trading_can_submit  # the environment wins over .env


def test_editing_env_after_startup_asks_for_a_restart(tmp_path):
    f = tmp_path / ".env"
    f.write_text(TRADING_ON + "QP_ALPACA_API_SECRET_KEY=old-secret\n")
    s = Settings(_env_file=f)
    s._source_file = f
    assert env_file_drift(s) == []
    f.write_text(
        "QP_ALPACA_TRADING_ENABLED=true\nQP_TRADING_DRY_RUN=true\nQP_ALPACA_API_SECRET_KEY=new-secret\n"
    )
    drift = env_file_drift(s)
    assert any(d.startswith("QP_TRADING_DRY_RUN: running false, file now says true") for d in drift)
    assert any(d.startswith("QP_ALPACA_API_SECRET_KEY: changed in the file") for d in drift)
    assert not any("secret" in d.split(":", 1)[1] for d in drift if "SECRET" in d.split(":", 1)[0])
    f.write_text("QP_ALPACA_PAPER=false\n")
    assert "no longer loads (alpaca_paper invalid)" in env_file_drift(s)[0]


def test_get_settings_reads_the_resolved_file(tmp_path, monkeypatch):
    f = tmp_path / "picked.env"
    f.write_text(TRADING_ON)
    monkeypatch.setenv("QP_ENV_FILE", str(f))
    cfg.get_settings.cache_clear()
    try:
        s = cfg.get_settings()
        assert s._source_file == f.resolve() and s._loaded_at is not None
        assert s.alpaca_trading_enabled and not s.trading_dry_run
    finally:
        cfg.get_settings.cache_clear()


@pytest.mark.parametrize("value", ["false", "False", "0"])
def test_paper_can_never_be_switched_off(value, monkeypatch):
    monkeypatch.setenv("QP_ALPACA_PAPER", value)
    with pytest.raises(ValueError, match="PAPER"):
        Settings(_env_file=None)
