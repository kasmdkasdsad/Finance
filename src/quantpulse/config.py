"""Application settings.

All settings are read from environment variables prefixed with ``QP_`` (or a ``.env`` file).
Secrets are held as :class:`pydantic.SecretStr` so they never leak into logs or API responses.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MarketProviderName = Literal["polygon", "alpaca", "yahoo"]

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


class Settings(BaseSettings):
    """Runtime configuration for the API, providers, caching, and domain defaults."""

    model_config = SettingsConfigDict(
        env_prefix="QP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    # --- Runtime -----------------------------------------------------------------------------
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_json: bool = False
    database_url: str = "sqlite+aiosqlite:///./data/quantpulse.db"
    auto_migrate: bool = True
    api_token: SecretStr | None = Field(
        default=None, description="If set, every /api request must send `X-API-Key: <token>`."
    )
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    # --- Live data switches ------------------------------------------------------------------
    enable_live_data: bool = Field(
        default=True, description="Master switch. When false every request is served from synthetic data."
    )
    market_providers: Annotated[list[MarketProviderName], NoDecode] = ["polygon", "alpaca", "yahoo"]

    # --- Provider credentials (all optional) ---------------------------------------------------
    polygon_api_key: SecretStr | None = None
    polygon_base_url: str = "https://api.polygon.io"
    alpaca_api_key_id: SecretStr | None = None
    alpaca_api_secret_key: SecretStr | None = None
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

    # --- Provider quotas ------------------------------------------------------------------------
    polygon_requests_per_minute: float = Field(default=5.0, gt=0, description="Free tier: 5/min.")

    @field_validator(
        "cors_origins", "watchlist", "market_providers", "picks_universe", "picks_recipients", mode="before"
    )
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("watchlist", "picks_universe")
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

    @field_validator("picks_send_time")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        from datetime import datetime as _dt

        _dt.strptime(value, "%H:%M")
        return value

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.email_from)

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()
