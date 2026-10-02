"""Application settings.

All settings are read from environment variables prefixed with ``QP_`` (or a ``.env`` file).
Secrets are held as :class:`pydantic.SecretStr` so they never leak into logs or API responses.

Which ``.env`` file (deterministic, whatever the working directory)
    1. the file named by ``QP_ENV_FILE``, if that variable is set;
    2. otherwise ``.env`` at the project root (the folder holding ``pyproject.toml``), when running from
       a source checkout;
    3. otherwise ``.env`` in the current working directory.

Process environment variables always win over the file (a ``QP_TRADING_DRY_RUN`` left in the shell
overrides the ``.env``), and settings are read once at startup: after editing ``.env`` the API must be
restarted. :func:`setting_sources` and :func:`env_file_drift` make both visible (``/trading/status``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, PrivateAttr, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MarketProviderName = Literal["polygon", "alpaca", "yahoo"]
TradingOrderType = Literal["marketable_limit", "limit", "market"]

# Liquid ETFs the paper-trading strategy ranks next to the stock universe (index, size and sector exposure).
DEFAULT_TRADING_ETFS = ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "SMH")
# Underlyings whose option chains the Brain reads (deep, liquid chains; a short list keeps data calls few).
DEFAULT_OPTIONS_UNIVERSE = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META")
# Option structures that may reach the paper account without a person enabling them: defined risk only.
DEFAULT_OPTIONS_STRUCTURES = (
    "long_call",
    "long_put",
    "bull_call_spread",
    "bear_put_spread",
    "bull_put_spread",
    "bear_call_spread",
    "covered_call",
)
# Weights of the opportunity-score components (each component is a cross-sectional z-score).
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "momentum": 0.25,
    "trend": 0.20,
    "volume": 0.10,
    "volatility": 0.10,
    "fundamental": 0.10,
    "model": 0.20,
    "regime": 0.05,
}
# Share of the maximum long exposure the strategy may deploy in each market regime.
DEFAULT_REGIME_EXPOSURE: dict[str, float] = {
    "bullish": 1.0,
    "neutral": 0.8,
    "high_volatility": 0.6,
    "bearish": 0.4,
    "risk_off": 0.15,
}
# Extra opportunity score (z) a new position needs in each regime ("stop opening weak positions").
DEFAULT_REGIME_ENTRY_PENALTY: dict[str, float] = {
    "bullish": 0.0,
    "neutral": 0.15,
    "high_volatility": 0.3,
    "bearish": 0.5,
    "risk_off": 1.0,
}

DEFAULT_SEC_USER_AGENT = "QuantPulse Terminal (set QP_SEC_USER_AGENT to 'Your Name your@email.com')"
DEFAULT_PICKS_UNIVERSE = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "TSLA",
    "BRK-B",
    "JPM",
    "LLY",
    "V",
    "UNH",
    "XOM",
    "MA",
    "COST",
    "HD",
    "PG",
    "JNJ",
    "ABBV",
    "WMT",
    "NFLX",
    "CRM",
    "BAC",
    "KO",
    "AMD",
    "ORCL",
    "PEP",
    "MRK",
    "ADBE",
)


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _parse_mapping(value: object) -> object:
    """``{"a": 1}`` (JSON) or ``a=1,b=2`` from an environment variable; dicts pass through."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return json.loads(text)
    out: dict[str, str] = {}
    for part in text.split(","):
        key, sep, raw = part.partition("=")
        if not sep:
            raise ValueError(f"expected name=value pairs, got {part!r}")
        out[key.strip()] = raw.strip()
    return out


def _unit_mapping(value: dict[str, float], allowed: tuple[str, ...], what: str) -> dict[str, float]:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; expected some of {list(allowed)}")
    if any(v < 0 for v in value.values()):
        raise ValueError(f"{what} values must be non-negative")
    return {k: float(value[k]) for k in allowed if k in value}


class Settings(BaseSettings):
    """Runtime configuration for the API, providers, caching, and domain defaults."""

    model_config = SettingsConfigDict(
        env_prefix="QP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
        populate_by_name=True,
    )

    # Where these settings came from (set by get_settings; None when built directly, e.g. in tests).
    _source_file: Path | None = PrivateAttr(default=None)
    _loaded_at: datetime | None = PrivateAttr(default=None)

    # --- Runtime -----------------------------------------------------------------------------
    environment: Literal["development", "production", "test"] = "development"
    deployment: Literal["local", "cloud"] = Field(
        default="local",
        description="cloud: the 24/7 server. Start-up refuses to run unless the preflight passes (Alpaca PAPER "
        "endpoint and key, API token, dashboard password, PostgreSQL, protected risk controls), and every "
        "request needs the API token (no exemption for this machine).",
    )
    log_level: str = "INFO"
    log_json: bool = False
    log_dir: str | None = Field(
        default=None,
        description="Also write logs to <dir>/quantpulse.log (rotated at 10 MB, ten old files kept; secrets masked).",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/quantpulse.db",
        description="SQLite locally; PostgreSQL in the cloud. postgres:// and postgresql:// URLs (as Render and "
        "other hosts hand them out) are accepted and use the asyncpg driver.",
    )
    auto_migrate: bool = Field(
        default=True,
        description="Apply migrations at start-up. In the cloud they run once per deploy instead (the pre-deploy "
        "command), and the API refuses to start on a schema that is not at the head.",
    )
    db_pool_size: int = Field(default=5, ge=1, le=50, description="PostgreSQL connections kept open.")
    db_max_overflow: int = Field(
        default=5, ge=0, le=50, description="Extra PostgreSQL connections under load."
    )
    db_connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    db_command_timeout_seconds: float = Field(
        default=120.0, gt=0, le=3600, description="A PostgreSQL statement that takes longer is cancelled."
    )
    db_startup_wait_seconds: float = Field(
        default=90.0,
        ge=0,
        le=600,
        description="At start-up, wait this long (retrying with back-off) for the database to answer before "
        "giving up: a database restarting for maintenance does not crash-loop the API. Nothing starts until it "
        "answers.",
    )
    shutdown_drain_seconds: float = Field(
        default=75.0,
        ge=0,
        le=280,
        description="On shutdown (a deploy, a restart) a Brain or trading cycle already running may finish for "
        "this long before it is cancelled; no new one starts meanwhile.",
    )
    api_token: SecretStr | None = Field(
        default=None, description="If set, every /api request must send `X-API-Key: <token>`."
    )
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    dashboard_password_hash: SecretStr | None = Field(
        default=None,
        description="PBKDF2 hash of the dashboard password (make it with quantpulse-hash-password); required "
        "in the cloud.",
    )

    # --- Live data switches ------------------------------------------------------------------
    enable_live_data: bool = Field(
        default=True, description="Master switch. When false every request is served from synthetic data."
    )
    market_providers: Annotated[list[MarketProviderName], NoDecode] = ["polygon", "alpaca", "yahoo"]

    # --- Provider credentials (all optional) ---------------------------------------------------
    polygon_api_key: SecretStr | None = None
    polygon_base_url: str = "https://api.polygon.io"
    # Alpaca paper keys serve both market data and the paper-trading API. The SDK's own variable names
    # (APCA_API_KEY_ID / APCA_API_SECRET_KEY) are accepted too.
    alpaca_api_key_id: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("QP_ALPACA_API_KEY_ID", "APCA_API_KEY_ID", "ALPACA_API_KEY_ID"),
    )
    alpaca_api_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "QP_ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY"
        ),
    )
    alpaca_data_url: str = "https://data.alpaca.markets"
    alpaca_stock_feed: Literal["iex", "sip", "delayed_sip"] = "iex"
    alpaca_options_feed: Literal["indicative", "opra"] = "indicative"
    fmp_api_key: SecretStr | None = None
    eia_api_key: SecretStr | None = None
    odds_api_key: SecretStr | None = None
    sec_user_agent: str = DEFAULT_SEC_USER_AGENT

    # --- HTTP / resilience ---------------------------------------------------------------------
    http_timeout_seconds: float = Field(default=12.0, gt=0)
    http_max_retries: int = Field(default=1, ge=0, le=5)
    circuit_failure_threshold: int = Field(default=3, ge=1)
    circuit_cooldown_seconds: float = Field(default=60.0, gt=0)
    rate_limit_max_wait_seconds: float = Field(
        default=2.0,
        ge=0,
        description="Longest a request will queue for a rate-limit token before failing over.",
    )

    # --- Cache TTLs (seconds) -----------------------------------------------------------------
    cache_max_entries: int = Field(default=4096, ge=16)
    ttl_quote: float = 15.0
    ttl_bars_intraday: float = 60.0
    ttl_bars_daily: float = 900.0
    ttl_options_chain: float = 120.0
    ttl_yield_curve: float = 3600.0
    ttl_fundamentals: float = 21600.0
    ttl_estimates: float = 21600.0
    ttl_fuel_prices: float = 21600.0
    ttl_vehicle_specs: float = 604800.0
    ttl_scoreboard_live: float = 20.0
    ttl_scoreboard_idle: float = 600.0
    ttl_season_results: float = 1800.0
    ttl_odds: float = 900.0
    ttl_model: float = Field(default=21600.0, gt=0, description="How long a stock-model run is reused.")
    ttl_reference: float = Field(
        default=604800.0,
        gt=0,
        description="S&P membership, SEC profiles and earnings dates are refreshed weekly.",
    )
    ttl_fundamentals_frames: float = Field(
        default=604800.0,
        gt=0,
        description="How long SEC XBRL frames (cross-company fundamentals) are reused.",
    )
    stale_grace_seconds: float = Field(
        default=86400.0, ge=0, description="How long an expired cache entry may still be served as STALE."
    )

    # --- Background polling ------------------------------------------------------------------
    polling_enabled: bool = True
    watchlist: Annotated[list[str], NoDecode] = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    poll_quotes_market_hours_seconds: float = Field(default=15.0, ge=1)
    poll_quotes_off_hours_seconds: float = Field(default=300.0, ge=1)
    poll_rates_seconds: float = Field(default=3600.0, ge=10)
    poll_sports_seconds: float = Field(default=60.0, ge=5)

    # --- Finance defaults --------------------------------------------------------------------
    equity_risk_premium: float = Field(default=0.05, ge=0, le=0.2)
    default_credit_spread: float = Field(default=0.015, ge=0, le=0.2)
    default_tax_rate: float = Field(default=0.21, ge=0, le=0.6)
    benchmark_symbol: str = "SPY"

    # --- Vehicle / fuel ----------------------------------------------------------------------
    fuel_region: str = "NUS"
    fuel_grade: Literal["regular", "midgrade", "premium", "diesel"] = "regular"

    # --- Sports ------------------------------------------------------------------------------
    odds_bookmaker_regions: str = "us"
    sports_include_prior_season: bool = Field(
        default=True, description="Seed Elo ratings with last season's results (regressed to the mean)."
    )

    # --- Daily picks & email digest -------------------------------------------------------------
    picks_universe: Annotated[list[str], NoDecode] = list(DEFAULT_PICKS_UNIVERSE)
    picks_top_n: int = Field(default=10, ge=1, le=50)
    picks_email_enabled: bool = Field(default=False, description="Send the digest every trading morning.")
    picks_recipients: Annotated[list[str], NoDecode] = []
    picks_send_time: str = Field(default="08:45", description="HH:MM, America/New_York, trading days only.")
    picks_allow_synthetic_email: bool = False
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_security: Literal["starttls", "ssl", "none"] = "starttls"
    email_from: str | None = None

    # --- Paper-trading sandbox -------------------------------------------------------------------
    sandbox_scheduler_enabled: bool = Field(
        default=True, description="Let the poller run auto-trading agents and record daily equity marks."
    )
    sandbox_trade_time: str = Field(
        default="10:00",
        description="HH:MM, America/New_York: when auto-trading agents rebalance each trading day.",
    )
    sandbox_mark_time: str = Field(
        default="16:05", description="HH:MM, America/New_York: when every account is marked to market."
    )

    # --- Stock model -----------------------------------------------------------------------------
    model_universe: str = Field(
        default="auto",
        description=(
            "'sp500' (every stock that was in the S&P 500 during the window, point-in-time), 'picks' "
            "(QP_PICKS_UNIVERSE), a comma-separated list of tickers, or 'auto': the S&P 500 when a "
            "bulk price source (Alpaca) is configured, otherwise the picks list."
        ),
    )
    model_type: Literal["ensemble", "ridge", "gbm"] = Field(
        default="ensemble", description="Which walk-forward model produces the live rankings."
    )
    model_sector_neutral: bool = Field(
        default=True, description="Compare each stock with its industry peers (sector-neutral features)."
    )
    model_sync_wait_seconds: float = Field(
        default=25.0,
        ge=0,
        description="How long a request waits for a model run before it continues in the background.",
    )
    model_warmup: bool = Field(
        default=True, description="Let the poller keep today's model run warm in the background."
    )

    # --- Forecasts ---------------------------------------------------------------------------------
    forecast_iv_weight: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Weight of options-implied volatility in forecast volatility (0 = GARCH only).",
    )
    forecast_variance_premium: float = Field(
        default=1.1,
        ge=1,
        le=2,
        description="Implied variance is divided by this before blending (options usually overprice risk).",
    )
    forecast_earnings_jumps: bool = Field(
        default=True, description="Add earnings-day jumps (from past reactions) to price forecasts."
    )

    # --- Alpaca paper trading (Alpaca's real *paper* API: simulated money only) --------------------
    # Nothing is ever sent to Alpaca unless QP_ALPACA_TRADING_ENABLED=true AND QP_TRADING_DRY_RUN=false.
    # There is deliberately no setting that points QuantPulse at a live-money account.
    alpaca_trading_enabled: bool = Field(
        default=False,
        description="Allow orders to be sent to the Alpaca paper account (also needs QP_TRADING_DRY_RUN=false).",
    )
    alpaca_paper: bool = Field(
        default=True, description="Must stay true: QuantPulse only ever trades the Alpaca paper account."
    )
    trading_dry_run: bool = Field(
        default=True, description="Compute signals, targets, trades and risk checks, but submit nothing."
    )
    trading_kill_switch: bool = Field(
        default=False, description="Refuse every new order (the dashboard has a runtime kill switch too)."
    )
    trading_scheduler_enabled: bool = Field(
        default=False,
        description=(
            "Let the poller run strategy cycles during market hours (dry runs included). Off by default: "
            "cycles run only when you start one, until you turn this on."
        ),
    )
    trading_scheduler_requires_arming: bool = Field(
        default=True,
        description=(
            "Scheduled cycles stay dry runs until paper execution has been exercised once by hand (a manual "
            "paper cycle or the confirmed test order), so enabling paper trading never fires a batch on its own."
        ),
    )
    trading_time: str = Field(default="10:00", description="HH:MM New York: the first cycle of each day.")
    trading_rebalance_interval_minutes: int = Field(default=30, ge=5, le=390)
    trading_stop_minutes_before_close: int = Field(
        default=15, ge=0, le=120, description="No scheduled cycle starts this close to the closing bell."
    )
    # universe
    trading_universe: str = Field(
        default="auto",
        description=(
            "'auto' (the stock model's universe — the S&P 500 with Alpaca data — plus QP_TRADING_ETFS, "
            "narrowed to the most liquid names) or a comma-separated list of tickers."
        ),
    )
    trading_universe_size: int = Field(default=120, ge=10, le=600)
    trading_etfs: Annotated[list[str], NoDecode] = list(DEFAULT_TRADING_ETFS)
    # position and portfolio limits
    trading_max_position_pct: float = Field(default=0.30, gt=0, le=1)
    trading_max_total_exposure_pct: float = Field(default=0.95, gt=0, le=1)
    trading_max_order_notional: float = Field(default=15_000.0, gt=0)
    trading_min_order_notional: float = Field(default=100.0, ge=1)
    trading_max_positions: int = Field(default=8, ge=1, le=50)
    trading_min_position_pct: float = Field(
        default=0.03, ge=0, lt=1, description="Target weights below this are dropped (no tiny positions)."
    )
    trading_cash_buffer_pct: float = Field(default=0.02, ge=0, lt=1)
    trading_max_daily_loss_pct: float = Field(default=0.04, gt=0, lt=1)
    trading_daily_loss_action: Literal["halt", "flatten"] = Field(
        default="halt",
        description="When the daily loss limit is hit: 'halt' stops new positions; 'flatten' also sells everything.",
    )
    trading_max_position_loss_pct: float = Field(
        default=0.08, gt=0, lt=1, description="Stop-loss: a position down this much is closed."
    )
    trading_take_profit_pct: float = Field(
        default=0.25, gt=0, description="A position up this much has part of the gain locked in."
    )
    trading_take_profit_fraction: float = Field(default=0.5, gt=0, le=1)
    trading_allow_shorts: bool = Field(
        default=False, description="Must stay false: the strategy is long-only."
    )
    trading_require_live_data: bool = Field(
        default=True, description="Refuse orders unless the quote is live (never synthetic, never stale)."
    )
    trading_max_quote_age_seconds: float = Field(default=600.0, gt=0)
    # liquidity
    trading_min_price: float = Field(default=5.0, ge=0)
    trading_min_dollar_volume: float = Field(
        default=25_000_000.0, ge=0, description="Minimum 20-day average daily dollar volume."
    )
    trading_max_spread_bps: float = Field(default=30.0, gt=0)
    trading_max_adv_pct: float = Field(
        default=0.01, gt=0, le=1, description="A position may not exceed this share of average dollar volume."
    )
    # execution
    trading_order_type: TradingOrderType = "marketable_limit"
    trading_limit_offset_bps: float = Field(
        default=10.0,
        ge=0,
        le=500,
        description="How far through the quote a marketable limit order is priced.",
    )
    trading_order_timeout_minutes: float = Field(
        default=20.0, gt=0, description="Unfilled strategy orders older than this are canceled next cycle."
    )
    trading_fill_wait_seconds: float = Field(
        default=15.0, ge=0, le=120, description="How long a cycle waits for sells to fill before buying."
    )
    # signal and turnover controls
    trading_entry_threshold: float = Field(
        default=0.75, description="Opportunity score (cross-sectional z) a new position needs."
    )
    trading_exit_threshold: float = Field(
        default=0.0, description="A held position whose score falls below this is sold."
    )
    trading_incumbent_bonus: float = Field(
        default=0.35, ge=0, description="Score head start for holdings when ranking (limits churn)."
    )
    trading_min_weight_change: float = Field(
        default=0.02, ge=0, lt=1, description="Smallest target-weight change worth trading."
    )
    trading_cooldown_minutes: float = Field(
        default=120.0,
        ge=0,
        description="No discretionary trade reversing a symbol's last trade (buy after sell, sell after buy) this soon.",
    )
    trading_max_cycle_turnover_pct: float = Field(
        default=0.6, gt=0, le=2, description="Discretionary trading per cycle, as a share of equity."
    )
    trading_position_vol_budget: float = Field(
        default=0.12, gt=0, le=1, description="Cap on weight × annual volatility for any one position."
    )
    trading_vol_floor: float = Field(
        default=0.12, gt=0, le=1, description="Volatility assumed at least this high when sizing."
    )
    trading_earnings_blackout_days: int = Field(
        default=2, ge=0, le=10, description="No new position this many days before an earnings release."
    )
    trading_use_implied_vol: bool = Field(
        default=True, description="Use options-implied volatility for the leading candidates when available."
    )
    trading_model_wait_seconds: float = Field(
        default=5.0, ge=0, description="How long a cycle waits for the stock model (it uses the last run)."
    )
    trading_signal_weights: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: dict(DEFAULT_SIGNAL_WEIGHTS),
        description="Opportunity-score weights, JSON or name=value pairs.",
    )
    trading_regime_exposure: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: dict(DEFAULT_REGIME_EXPOSURE),
        description="Share of maximum exposure deployed per regime.",
    )
    trading_regime_entry_penalty: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: dict(DEFAULT_REGIME_ENTRY_PENALTY),
        description="Extra entry score required per regime.",
    )

    # --- Brain (multi-agent portfolio manager of the Alpaca PAPER account) ---------------------------
    brain_mode: Literal["research_only", "dry_run", "paper_recommendation", "paper_execution"] = Field(
        default="paper_execution",
        description=(
            "paper_execution: the Brain owns the Alpaca PAPER account — its approved decisions go through the "
            "trading service (the same risk engine, order manager and switches as every order; nothing is sent "
            "while QP_ALPACA_TRADING_ENABLED=false or QP_TRADING_DRY_RUN=true) and the strategy only runs as a "
            "dry run. paper_recommendation / dry_run: proposals only, managed in the Brain's simulated paper "
            "book; research_only: analysis and memory only."
        ),
    )
    brain_strategy_shadow: bool = Field(
        default=True,
        description="While the Brain owns the account, run the strategy it replaced against a hypothetical "
        "portfolio of its own (same risk limits, modelled fills; never an order) for the comparison.",
    )
    brain_kill_switch: bool = Field(
        default=False,
        description="Refuse every new Brain-originated order (the dashboard has a runtime Brain kill switch too). "
        "Positions stay where they are; the trading kill switch and close-all still work.",
    )
    brain_focus_candidates: int = Field(
        default=16,
        ge=0,
        le=50,
        description="Non-held symbols studied closely per cycle (best of a pre-screen).",
    )
    brain_agent_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    brain_min_confidence: float = Field(
        default=0.45,
        ge=0,
        le=1,
        description="Consensus confidence (after the devil's advocate) needed before a buy, increase, reduce or "
        "close is proposed. A starting value: the learning system's calibration shows whether it is borne out.",
    )
    brain_fresh_quote_seconds: float = Field(
        default=60.0, gt=0, description="A live quote younger than this is FRESH (older but allowed: LIVE)."
    )
    brain_min_reliability_observations: int = Field(
        default=30,
        ge=5,
        description="Evaluated predictions an agent needs before its measured reliability affects its weight.",
    )
    brain_max_new_positions_per_cycle: int = Field(default=2, ge=0, le=20)
    brain_max_opportunities: int = Field(
        default=10,
        ge=0,
        le=30,
        description="Detected opportunities added to a cycle's focus (strongest first).",
    )
    brain_max_opportunities_recorded: int = Field(
        default=40, ge=1, le=500, description="Detections recorded per cycle (the rest are dropped)."
    )
    brain_earnings_caution_days: int = Field(
        default=3,
        ge=0,
        le=30,
        description="No new position or increase this many days before an earnings release (at least the "
        "trading blackout).",
    )
    brain_supervisor_enabled: bool = Field(
        default=True,
        description="Run the brain by market session and by event while the server runs (in paper_execution "
        "its approved decisions go to the trading service). Can be paused at runtime.",
    )
    brain_cycle_minutes: int = Field(
        default=5,
        ge=5,
        le=390,
        description="A full cycle this often in the session; Brain orders are deduplicated per slot of this length. "
        "The risk limits, the per-cycle caps and the no-reversal cooldown apply as always.",
    )
    brain_lease_seconds: int = Field(
        default=180,
        ge=30,
        le=3600,
        description="The single-supervisor lease: at most one process supervises the Brain and sends orders; "
        "if it stops renewing (a crash), another may take over only after this many seconds.",
    )
    brain_monitor_minutes: int = Field(
        default=5, ge=1, le=120, description="Quote monitor for holdings and the last focus in the session."
    )
    brain_max_event_cycles_per_hour: int = Field(
        default=4, ge=0, le=60, description="Focused cycles that events may trigger per hour."
    )
    brain_lab_history_days: int = Field(
        default=1500, ge=400, le=7300, description="Calendar days of daily history the strategy lab tests on."
    )
    brain_lab_universe_size: int = Field(
        default=150, ge=10, le=600, description="Most liquid names the strategy lab backtests."
    )
    brain_lab_min_oos_sessions: int = Field(
        default=500, ge=50, le=5000, description="Out-of-sample sessions a strategy needs before a verdict."
    )
    brain_lab_paper_days: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Paper-tracked sessions required before a version can be promoted.",
    )
    brain_use_stock_model: bool = Field(
        default=True,
        description="Give the fundamental, valuation and factor agents the stock model's latest completed run "
        "(never waits for training).",
    )
    brain_options_analysis: bool = Field(
        default=True,
        description="Read option chains for the focus symbols (live data only; never synthetic).",
    )
    brain_catalyst_analysis: bool = Field(
        default=True, description="Read earnings calendars and past earnings reactions for the focus symbols."
    )
    brain_research_timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        le=120,
        description="Time budget for each extra evidence fetch (options, earnings).",
    )
    brain_book_capital: float = Field(
        default=100_000.0,
        gt=0,
        description="Starting capital of the Brain's hypothetical paper book (simulated; never an Alpaca account).",
    )
    brain_book_slippage_bps: float = Field(
        default=5.0,
        ge=0,
        le=200,
        description="Simulated slippage beyond half the quoted spread on every paper-book fill.",
    )
    brain_book_cost_bps: float = Field(
        default=1.0, ge=0, le=100, description="Simulated fees on every paper-book fill (of notional)."
    )
    brain_book_default_half_spread_bps: float = Field(
        default=5.0,
        ge=0,
        le=200,
        description="Half-spread assumed when no believable bid/ask exists (an assumption, labelled on each fill).",
    )
    brain_llm_provider: str = Field(
        default="none",
        description="Language-model provider for the optional model-backed agents ('none': every analysis is "
        "deterministic and those agents skip themselves). Calculations never go to a model.",
    )
    brain_llm_fast_model: str = Field(
        default="", description="Model for fast tasks (summaries, extraction, classification); empty: none."
    )
    brain_llm_strong_model: str = Field(
        default="", description="Model for strong tasks (research, debate, synthesis); empty: none."
    )
    brain_llm_daily_token_budget: int = Field(
        default=0,
        ge=0,
        description="Input + output tokens the brain may spend on language models per day (0: no calls).",
    )
    brain_llm_max_output_tokens: int = Field(default=600, ge=16, le=8000)
    brain_llm_timeout_seconds: float = Field(default=45.0, gt=0, le=600)
    brain_llm_max_concurrency: int = Field(default=2, ge=1, le=16)
    brain_llm_cache_minutes: int = Field(
        default=240, ge=0, le=10080, description="Identical requests are answered from the cache."
    )
    brain_llm_max_briefings: int = Field(
        default=3, ge=0, le=20, description="Focus symbols summarised by the briefing agent per cycle."
    )

    # --- Options (Alpaca PAPER only; defined risk; the same risk engine, order manager and kill switches) -
    options_enabled: bool = Field(
        default=True,
        description="Research options, shadow-trade them on live quotes and let the Brain weigh option "
        "expressions of its views. Every option order goes through the trading service, the risk engine and "
        "the order manager exactly like a stock order (paper only).",
    )
    options_execution: bool = Field(
        default=True,
        description="Send approved option orders to the Alpaca paper account. False: options are researched "
        "and shadow-traded only (every decision is still recorded and graded).",
    )
    options_priority_weight: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="How much an option expression is favoured over the equivalent stock trade when both clear "
        "every check (added to its risk-adjusted score). It never forces an option: one that fails any check, "
        "or whose expected value per dollar at risk is not positive, is never chosen.",
    )
    options_universe: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_OPTIONS_UNIVERSE),
        description="Underlyings whose chains are read each cycle (liquid names keep spreads tight).",
    )
    options_allowed_structures: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_OPTIONS_STRUCTURES),
        description="Structure families that may be sent to the paper account. Only defined-risk families can "
        "be listed; naked short options are never executable. Adding a family needs a person (protected).",
    )
    options_max_loss_per_trade: float = Field(
        default=500.0, gt=0, description="Largest possible loss of one new option position, in dollars."
    )
    options_max_loss_pct_per_trade: float = Field(
        default=0.01, gt=0, le=0.1, description="…and as a share of equity (the smaller limit applies)."
    )
    options_max_total_risk_pct: float = Field(
        default=0.06, gt=0, le=0.5, description="The sum of the maximum losses of every option position."
    )
    options_max_underlying_risk_pct: float = Field(
        default=0.02, gt=0, le=0.2, description="The maximum loss of all option positions on one underlying."
    )
    options_max_positions: int = Field(default=6, ge=1, le=50, description="Open option structures at most.")
    options_max_contracts: int = Field(default=10, ge=1, le=200, description="Contracts per leg per order.")
    options_min_dte: int = Field(
        default=7, ge=1, le=365, description="No new position closer to expiration (0DTE is research only)."
    )
    options_max_dte: int = Field(default=60, ge=1, le=730)
    options_close_dte: int = Field(
        default=2,
        ge=0,
        le=30,
        description="Positions are closed (or rolled) at this many days to expiration: never carried into "
        "expiration, never exercised by QuantPulse.",
    )
    options_max_spread_pct: float = Field(
        default=0.15, gt=0, le=1, description="Widest bid-ask spread of a leg, as a share of its mid."
    )
    options_max_quote_age_seconds: float = Field(
        default=120.0, gt=0, le=3600, description="An option quote older than this is not traded on."
    )
    options_min_open_interest: float = Field(default=100.0, ge=0, description="Per leg, for new positions.")
    options_max_delta_pct: float = Field(
        default=0.30,
        gt=0,
        le=2,
        description="Net option delta (in dollars of underlying) across the book, as a share of equity.",
    )
    options_max_vega_pct: float = Field(
        default=0.005,
        gt=0,
        le=0.1,
        description="Net option vega (dollars per volatility point) across the book, as a share of equity.",
    )
    options_take_profit_pct: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Close a position once it has made this share of its maximum profit.",
    )
    options_stop_loss_pct: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Close a position once it has lost this share of its maximum loss.",
    )
    options_fee_per_contract: float = Field(
        default=0.05,
        ge=0,
        le=5,
        description="Assumed fees per contract per side in research and shadow trades.",
    )
    options_min_shadow_trades: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Closed shadow trades (on live quotes) a strategy needs before paper execution at full size.",
    )
    options_exploration: bool = Field(
        default=True,
        description="A strategy that passed walk-forward and stress validation may trade one contract on the "
        "paper account (maximum loss capped below) while its shadow record builds, to measure real paper fills. "
        "Labelled as exploration everywhere; false: only fully validated strategies trade.",
    )
    options_exploration_max_loss: float = Field(
        default=250.0, gt=0, description="Largest possible loss of one exploration trade, in dollars."
    )
    options_research_budget_seconds: float = Field(
        default=180.0, ge=10, le=3600, description="Time the research lab may spend per run (background)."
    )
    options_research_time: str = Field(
        default="16:40", description="HH:MM New York: the daily research run (after the close)."
    )
    evolution_enabled: bool = Field(
        default=True,
        description="Measure the market daily (volatility, microstructure, options, correlations, liquidity, "
        "execution, strategies), detect changes, test competing explanations and re-validate what depends on "
        "what changed.",
    )
    evolution_time: str = Field(default="16:50", description="HH:MM New York: the daily evolution scan.")
    evolution_reference_days: int = Field(
        default=120, ge=30, le=1000, description="Days of history a recent window is compared against."
    )
    evolution_recent_days: int = Field(default=20, ge=5, le=250, description="The recent window, in days.")

    # --- 24/7: the closed-market research loop (analysis only; it never changes trading) -------------
    research_enabled: bool = Field(
        default=True,
        description="While the market is closed the Brain works through its research queue (grading, analysis, "
        "backtests, preparation). Research never sends an order or changes a setting, limit or strategy.",
    )
    research_max_concurrent: int = Field(
        default=1, ge=1, le=2, description="Research jobs running at once (a small server: keep 1)."
    )
    research_max_memory_pct: float = Field(
        default=70.0,
        ge=30,
        le=90,
        description="No new research job starts while memory use (the container's limit, or the machine's) is "
        "above this percentage.",
    )
    research_abort_memory_pct: float = Field(
        default=85.0,
        ge=40,
        le=95,
        description="A running research job is stopped (and queued again) above this memory percentage: "
        "execution and safety always come first.",
    )
    research_max_load: float = Field(
        default=0.85, gt=0, le=4, description="No new research job starts above this load average per CPU."
    )
    research_job_timeout_minutes: float = Field(
        default=30.0, gt=0, le=240, description="A research job running longer than this is stopped."
    )

    # --- Cloud monitoring and alerts (all optional and free; nothing is sent unless a URL is set) ------
    alert_ntfy_url: SecretStr | None = Field(
        default=None,
        description="Push alerts to a phone through ntfy (https://ntfy.sh/<a long random topic>); the topic "
        "acts as a password: keep it secret.",
    )
    alert_webhook_url: SecretStr | None = Field(
        default=None, description="Also POST alerts as JSON to this webhook (Slack- and Discord-compatible)."
    )
    heartbeat_url: SecretStr | None = Field(
        default=None,
        description="A dead-man's switch (e.g. healthchecks.io): pinged every few minutes while QuantPulse is "
        "healthy, '<url>/fail' when it is not; the service alerts you when the pings stop (server down).",
    )
    alert_cooldown_minutes: int = Field(
        default=30, ge=1, le=1440, description="The same alert is not repeated within this many minutes."
    )
    brain_anomaly_orders_per_hour: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Brain orders rejected, failed or left unknown within an hour that turn the Brain kill "
        "switch on by themselves (fail closed; a person releases it).",
    )
    health_data_quality_alert_cycles: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Consecutive in-session cycles halted by market data before an alert.",
    )

    # --- Prediction ledger ------------------------------------------------------------------------
    predictions_enabled: bool = Field(
        default=True, description="Log forecasts and model predictions after each close and grade them later."
    )
    predictions_log_time: str = Field(
        default="16:20", description="HH:MM, America/New_York, trading days only."
    )
    predictions_allow_synthetic: bool = Field(
        default=False, description="Log predictions made from synthetic prices (never recommended)."
    )

    # --- Provider quotas ------------------------------------------------------------------------
    polygon_requests_per_minute: float = Field(default=5.0, gt=0, description="Free tier: 5/min.")

    @field_validator(
        "cors_origins",
        "watchlist",
        "market_providers",
        "picks_universe",
        "picks_recipients",
        "trading_etfs",
        "options_universe",
        "options_allowed_structures",
        mode="before",
    )
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator(
        "api_token",
        "dashboard_password_hash",
        "polygon_api_key",
        "alpaca_api_key_id",
        "alpaca_api_secret_key",
        "fmp_api_key",
        "eia_api_key",
        "odds_api_key",
        "smtp_password",
        "alert_ntfy_url",
        "alert_webhook_url",
        "heartbeat_url",
        mode="before",
    )
    @classmethod
    def _blank_secret(cls, value: object) -> object:
        """``QP_X=`` (as copied from .env.example) means "not set", not an empty credential."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def _async_database_url(cls, value: object) -> object:
        """``postgres://`` / ``postgresql://`` (Render, Heroku, …) → the asyncpg driver QuantPulse uses."""
        if isinstance(value, str):
            v = value.strip()
            for prefix in ("postgres://", "postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://"):
                if v.startswith(prefix):
                    return "postgresql+asyncpg://" + v[len(prefix) :]
            return v
        return value

    @field_validator("log_dir", mode="before")
    @classmethod
    def _blank_dir(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("alpaca_paper")
    @classmethod
    def _paper_only(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "QuantPulse only supports Alpaca PAPER trading; live-money trading is intentionally not "
                "implemented. Set QP_ALPACA_PAPER=true."
            )
        return value

    @field_validator("trading_allow_shorts")
    @classmethod
    def _long_only(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "short selling is not implemented: the strategy is long-only (QP_TRADING_ALLOW_SHORTS=false)"
            )
        return value

    @field_validator(
        "trading_signal_weights", "trading_regime_exposure", "trading_regime_entry_penalty", mode="before"
    )
    @classmethod
    def _parse_mappings(cls, value: object) -> object:
        return _parse_mapping(value)

    @field_validator("trading_signal_weights")
    @classmethod
    def _validate_signal_weights(cls, value: dict[str, float]) -> dict[str, float]:
        from quantpulse.domain.trading_signals import COMPONENTS

        weights = _unit_mapping(value, COMPONENTS, "signal components")
        if sum(weights.values()) <= 0:
            raise ValueError("at least one signal weight must be positive")
        return weights

    @field_validator("trading_regime_exposure", "trading_regime_entry_penalty")
    @classmethod
    def _validate_regime_mapping(cls, value: dict[str, float]) -> dict[str, float]:
        from quantpulse.domain.trading_regime import LABELS

        return _unit_mapping(value, LABELS, "market regimes")

    @field_validator("trading_regime_exposure")
    @classmethod
    def _exposure_share(cls, value: dict[str, float]) -> dict[str, float]:
        if any(v > 1 for v in value.values()):
            raise ValueError("regime exposure shares must be between 0 and 1")
        return {**DEFAULT_REGIME_EXPOSURE, **value}

    @field_validator("trading_regime_entry_penalty")
    @classmethod
    def _entry_defaults(cls, value: dict[str, float]) -> dict[str, float]:
        return {**DEFAULT_REGIME_ENTRY_PENALTY, **value}

    @field_validator("trading_universe")
    @classmethod
    def _validate_trading_universe(cls, value: str) -> str:
        v = value.strip()
        if v.lower() == "auto":
            return "auto"
        symbols = [s.strip().upper() for s in v.split(",") if s.strip()]
        if not symbols:
            raise ValueError("trading_universe must be 'auto' or a comma-separated list of tickers")
        return ",".join(dict.fromkeys(symbols))

    @field_validator("model_universe")
    @classmethod
    def _validate_universe(cls, value: str) -> str:
        v = value.strip()
        if v.lower() in ("auto", "sp500", "picks"):
            return v.lower()
        symbols = [s.strip().upper() for s in v.split(",") if s.strip()]
        if len(symbols) < 3:
            raise ValueError(
                "model_universe must be auto, sp500, picks or at least 3 comma-separated tickers"
            )
        return ",".join(dict.fromkeys(symbols))

    @field_validator("options_allowed_structures")
    @classmethod
    def _defined_risk_structures(cls, value: list[str]) -> list[str]:
        from quantpulse.options.structures import FAMILIES

        out: list[str] = []
        for name in (v.strip().lower() for v in value):
            if not name or name in out:
                continue
            family = FAMILIES.get(name)
            if family is None or name == "stock":
                raise ValueError(f"unknown option structure {name!r}")
            if not family.defined_risk:
                raise ValueError(f"{name} has unlimited risk: it can never be executed")
            out.append(name)
        return out

    @field_validator("watchlist", "picks_universe", "trading_etfs", "options_universe")
    @classmethod
    def _normalise_symbols(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for symbol in value:
            s = symbol.strip().upper()
            if s and s not in out:
                out.append(s)
        return out

    @field_validator("picks_recipients")
    @classmethod
    def _validate_emails(cls, value: list[str]) -> list[str]:
        from pydantic import EmailStr, TypeAdapter

        adapter = TypeAdapter(EmailStr)
        return [adapter.validate_python(v) for v in value]

    @field_validator(
        "picks_send_time",
        "sandbox_trade_time",
        "sandbox_mark_time",
        "predictions_log_time",
        "trading_time",
        "options_research_time",
        "evolution_time",
    )
    @classmethod
    def _validate_time(cls, value: str) -> str:
        from datetime import datetime as _dt

        _dt.strptime(value, "%H:%M")
        return value

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.email_from)

    @property
    def trading_can_submit(self) -> bool:
        """Whether settings allow real paper orders (the runtime kill switch is checked separately)."""
        return (
            self.alpaca_trading_enabled
            and self.alpaca_paper
            and not self.trading_dry_run
            and not self.trading_kill_switch
            and self.has_credentials("alpaca")
        )

    @property
    def brain_owns_account(self) -> bool:
        """Whether the Brain manages the Alpaca paper account (the strategy is then a dry run only)."""
        return self.brain_mode == "paper_execution"

    @property
    def sqlite_path(self) -> Path | None:
        """Filesystem path of the SQLite database, or ``None`` for in-memory / non-SQLite URLs."""
        prefix = "sqlite+aiosqlite:///"
        if self.database_url.startswith(prefix):
            raw = self.database_url[len(prefix) :]
            if raw in ("", ":memory:"):
                return None
            return Path(raw)
        return None

    def has_credentials(self, provider: str) -> bool:
        """Whether credentials required by ``provider`` are configured."""
        checks: dict[str, bool] = {
            "polygon": self.polygon_api_key is not None,
            "alpaca": self.alpaca_api_key_id is not None and self.alpaca_api_secret_key is not None,
            "fmp": self.fmp_api_key is not None,
            "eia": self.eia_api_key is not None,
            "odds_api": self.odds_api_key is not None,
        }
        return checks.get(provider, True)


ENV_FILE_VAR = "QP_ENV_FILE"
# Settings that decide whether (and how) orders reach the Alpaca paper account, with the variable names
# that can set each one. Reported with their source, never with secret values.
TRADING_SWITCHES: dict[str, tuple[str, ...]] = {
    "alpaca_trading_enabled": ("QP_ALPACA_TRADING_ENABLED",),
    "trading_dry_run": ("QP_TRADING_DRY_RUN",),
    "alpaca_paper": ("QP_ALPACA_PAPER",),
    "trading_kill_switch": ("QP_TRADING_KILL_SWITCH",),
    "trading_scheduler_enabled": ("QP_TRADING_SCHEDULER_ENABLED",),
    "trading_scheduler_requires_arming": ("QP_TRADING_SCHEDULER_REQUIRES_ARMING",),
    "brain_mode": ("QP_BRAIN_MODE",),
    "brain_kill_switch": ("QP_BRAIN_KILL_SWITCH",),
    "trading_require_live_data": ("QP_TRADING_REQUIRE_LIVE_DATA",),
    "trading_max_spread_bps": ("QP_TRADING_MAX_SPREAD_BPS",),
    "enable_live_data": ("QP_ENABLE_LIVE_DATA",),
    "alpaca_stock_feed": ("QP_ALPACA_STOCK_FEED",),
    "alpaca_api_key_id": ("QP_ALPACA_API_KEY_ID", "APCA_API_KEY_ID", "ALPACA_API_KEY_ID"),
    "alpaca_api_secret_key": ("QP_ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY"),
    "api_token": ("QP_API_TOKEN",),
}
SECRET_SWITCHES = frozenset({"alpaca_api_key_id", "alpaca_api_secret_key", "api_token"})


def project_root() -> Path | None:
    """The source checkout this package runs from (the folder with ``pyproject.toml``), if any."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "quantpulse").is_dir():
            return parent
    return None


def resolve_env_file(environ: Mapping[str, str] | None = None, cwd: Path | None = None) -> Path | None:
    """The ``.env`` file settings are read from (see the module docstring for the order)."""
    env = os.environ if environ is None else environ
    explicit = env.get(ENV_FILE_VAR, "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = project_root()
    if root is not None and (root / ".env").is_file():
        return root / ".env"
    local = (cwd or Path.cwd()) / ".env"
    return local.resolve() if local.is_file() else None


def _read_env_file(path: Path | None) -> dict[str, str | None]:
    if path is None or not path.is_file():
        return {}
    from dotenv import dotenv_values

    return {k.upper(): v for k, v in dotenv_values(path, encoding="utf-8-sig").items()}


def _render(field: str, value: Any) -> str:
    if field in SECRET_SWITCHES:
        return "set" if value is not None else "not set"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True, slots=True)
class SettingSource:
    field: str
    variable: str  # the variable that set it (or the primary name when defaulted)
    source: Literal["environment", "env_file", "default"]
    value: str  # rendered; secrets are only "set" / "not set"


def setting_sources(
    settings: Settings,
    environ: Mapping[str, str] | None = None,
    fields: Mapping[str, Sequence[str]] = TRADING_SWITCHES,
) -> list[SettingSource]:
    """Where each trading switch's running value came from: the process environment (which overrides
    ``.env``), the ``.env`` file, or the built-in default."""
    env = {k.upper(): v for k, v in (os.environ if environ is None else environ).items()}
    file_values = _read_env_file(settings._source_file)
    out: list[SettingSource] = []
    for field, names in fields.items():
        value = _render(field, getattr(settings, field))
        in_env = next((n for n in names if n.upper() in env), None)
        in_file = next((n for n in names if n.upper() in file_values), None)
        if in_env is not None:
            out.append(SettingSource(field, in_env, "environment", value))
        elif in_file is not None:
            out.append(SettingSource(field, in_file, "env_file", value))
        else:
            out.append(SettingSource(field, names[0], "default", value))
    return out


def env_file_drift(settings: Settings, fields: Sequence[str] = tuple(TRADING_SWITCHES)) -> list[str]:
    """Trading switches whose value in the ``.env`` file no longer matches the running settings: the file
    was edited after startup, so the API must be restarted for the change to apply."""
    path = settings._source_file
    if path is None or not path.is_file():
        return []
    try:
        fresh = Settings(_env_file=path)
    except ValidationError as exc:
        bad = sorted({str(e["loc"][0]) for e in exc.errors() if e.get("loc")})
        return [f"{path.name} no longer loads ({', '.join(bad)} invalid): the running API keeps its settings"]
    out: list[str] = []
    for field in fields:
        running, on_disk = getattr(settings, field), getattr(fresh, field)
        if field in SECRET_SWITCHES:
            same = (running is None) == (on_disk is None) and (
                running is None or running.get_secret_value() == on_disk.get_secret_value()
            )
            detail = "changed in the file" if not same else ""
        else:
            same = running == on_disk
            detail = f"running {_render(field, running)}, file now says {_render(field, on_disk)}"
        if not same:
            name = TRADING_SWITCHES.get(field, (field.upper(),))[0]
            out.append(f"{name}: {detail} — restart the API to apply")
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance (read once: restart after editing ``.env``)."""
    env_file = resolve_env_file()
    settings = Settings(_env_file=env_file)
    settings._source_file = env_file
    settings._loaded_at = datetime.now(UTC)
    return settings
