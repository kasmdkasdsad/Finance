"""Paper-trading settings: safe defaults, the paper-only and long-only guards, and credential handling."""

import pytest
from pydantic import ValidationError

from quantpulse.config import DEFAULT_SIGNAL_WEIGHTS, Settings
from quantpulse.domain.trading_regime import LABELS
from quantpulse.domain.trading_signals import COMPONENTS


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_defaults_are_a_dry_run_that_never_submits():
    s = settings(alpaca_api_key_id="k", alpaca_api_secret_key="s")
    assert s.alpaca_paper is True and s.alpaca_trading_enabled is False and s.trading_dry_run is True
    assert s.trading_kill_switch is False and s.trading_allow_shorts is False
    assert s.trading_can_submit is False
    assert (s.trading_max_position_pct, s.trading_max_total_exposure_pct, s.trading_max_positions) == (
        0.30,
        0.95,
        8,
    )
    assert (s.trading_max_order_notional, s.trading_min_order_notional) == (15_000.0, 100.0)
    assert (s.trading_max_daily_loss_pct, s.trading_max_position_loss_pct) == (0.04, 0.08)
    assert s.trading_time == "10:00" and s.trading_rebalance_interval_minutes == 30
    assert s.trading_require_live_data is True


def test_submission_needs_every_switch():
    base = dict(
        alpaca_api_key_id="k", alpaca_api_secret_key="s", alpaca_trading_enabled=True, trading_dry_run=False
    )
    assert settings(**base).trading_can_submit
    assert not settings(**{**base, "trading_dry_run": True}).trading_can_submit
    assert not settings(**{**base, "alpaca_trading_enabled": False}).trading_can_submit
    assert not settings(**{**base, "trading_kill_switch": True}).trading_can_submit
    assert not settings(**{**base, "alpaca_api_key_id": None}).trading_can_submit


def test_live_money_and_shorting_cannot_be_configured():
    with pytest.raises(ValidationError, match="PAPER"):
        settings(alpaca_paper=False)
    with pytest.raises(ValidationError, match="long-only"):
        settings(trading_allow_shorts=True)


def test_blank_credentials_mean_not_set(monkeypatch):
    monkeypatch.setenv("QP_ALPACA_API_KEY_ID", "")
    monkeypatch.setenv("QP_API_TOKEN", "  ")
    s = settings()
    assert s.alpaca_api_key_id is None and s.api_token is None and not s.has_credentials("alpaca")


def test_alpaca_sdk_variable_names_are_accepted(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "PKFROMSDKENV")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "sdk-secret")
    s = settings()
    assert s.has_credentials("alpaca")
    assert "PKFROMSDKENV" not in repr(s) and "sdk-secret" not in str(s.model_dump())


def test_signal_weights_and_regime_maps():
    assert set(DEFAULT_SIGNAL_WEIGHTS) == set(COMPONENTS)
    s = settings(trading_signal_weights="momentum=0.5,model=0.5", trading_regime_exposure='{"risk_off": 0}')
    assert s.trading_signal_weights == {"momentum": 0.5, "model": 0.5}
    assert s.trading_regime_exposure["risk_off"] == 0 and set(s.trading_regime_exposure) == set(LABELS)
    with pytest.raises(ValidationError, match="unknown signal components"):
        settings(trading_signal_weights="luck=1")
    with pytest.raises(ValidationError, match="positive"):
        settings(trading_signal_weights="momentum=0")
    with pytest.raises(ValidationError, match="between 0 and 1"):
        settings(trading_regime_exposure="bullish=1.5")


def test_universe_and_time_validation():
    assert settings(trading_universe=" aapl, msft ,AAPL").trading_universe == "AAPL,MSFT"
    assert settings(trading_etfs="spy,qqq").trading_etfs == ["SPY", "QQQ"]
    with pytest.raises(ValidationError):
        settings(trading_time="25:00")
    with pytest.raises(ValidationError):
        settings(trading_max_position_pct=1.5)
