"""Cloud preflight: the checks a 24/7 deployment must pass before anything starts.

With ``QP_DEPLOYMENT=cloud`` the API refuses to start — no database connection, no supervisor, no poller,
no order — unless every check below passes. ``quantpulse-preflight`` runs the same checks by hand (exit
code 0 when they pass, 2 when not) and prints only masked values: never a key, a secret, a token, a
password or a database password.

Paper only
    ``QP_ALPACA_PAPER=true`` must be set explicitly; no variable anywhere in the environment (or the
    ``.env`` file) may name Alpaca's live-money trading or broker API; every variable that names an Alpaca
    trading endpoint must name the paper one; the Alpaca SDK client is built (no network call) and must
    point at ``https://paper-api.alpaca.markets``; the key must pass the existing paper-key check (paper key
    ids start with ``PK``) and the market-data URL must be Alpaca's own, so the keys go nowhere else.
Access
    ``QP_API_TOKEN`` of at least 32 characters (every API request needs it; nothing is exempt for this
    machine in the cloud). The dashboard service enforces its own password.
Ownership
    ``QP_BRAIN_MODE=paper_execution`` (the Brain owns the paper account), ``QP_TRADING_SCHEDULER_ENABLED=false``
    (one scheduler: the Brain supervisor), and no credential given twice with different values.
Storage
    PostgreSQL (``postgresql+asyncpg://``): the Brain's history lives in a database with its own volume and
    backups, not a file inside a container.
Protected controls
    The risk controls may be tightened in the cloud, never loosened past the limits QuantPulse ships with:
    live data required, quote age, spread, daily loss, position, exposure and order limits, liquidity floors,
    long-only, arming before scheduled paper orders.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from quantpulse.config import Settings, _read_env_file
from quantpulse.core.passwords import PasswordHashError, parse
from quantpulse.providers.alpaca_trading import PAPER_URL, AlpacaPaperBroker, BrokerError

# Alpaca's live-money trading API and its broker API: named anywhere, the deployment does not start.
LIVE_ENDPOINT = re.compile(r"(?<![\w.-])(api|broker-api)\.alpaca\.markets", re.IGNORECASE)
# Variables other tools use for the trading endpoint: when set, they must name the paper API.
ENDPOINT_VARIABLES = (
    "APCA_API_BASE_URL",
    "APCA_ENDPOINT",
    "ALPACA_BASE_URL",
    "ALPACA_API_BASE_URL",
    "ALPACA_ENDPOINT",
    "ALPACA_TRADING_URL",
    "QP_ALPACA_BASE_URL",
    "QP_ALPACA_TRADING_URL",
    "QP_ALPACA_ENDPOINT",
)
# The same credential can be given under several names (QuantPulse's and the Alpaca SDK's): if two of them
# disagree, which one is used is ambiguous — refused.
CREDENTIAL_ALIASES = {
    "the Alpaca key id": ("QP_ALPACA_API_KEY_ID", "APCA_API_KEY_ID", "ALPACA_API_KEY_ID"),
    "the Alpaca secret": ("QP_ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY"),
}
PAPER_FLAGS = ("ALPACA_PAPER", "APCA_PAPER", "QP_ALPACA_ENV", "ALPACA_ENV")
DATA_URL = "https://data.alpaca.markets"
MIN_TOKEN = 32
PLACEHOLDERS = ("change-me", "changeme", "replace", "your-", "your_", "example", "xxxxxxxx", "secret")
# Protected risk controls: (setting, "max" = may not rise above the shipped default, "min" = may not fall
# below it, "equal" = must keep it).
PROTECTED: tuple[tuple[str, str], ...] = (
    ("trading_require_live_data", "equal"),
    ("trading_scheduler_requires_arming", "equal"),
    ("trading_allow_shorts", "equal"),
    ("alpaca_paper", "equal"),
    ("trading_max_quote_age_seconds", "max"),
    ("trading_max_spread_bps", "max"),
    ("trading_max_daily_loss_pct", "max"),
    ("trading_max_position_pct", "max"),
    ("trading_max_total_exposure_pct", "max"),
    ("trading_max_order_notional", "max"),
    ("trading_max_positions", "max"),
    ("trading_max_position_loss_pct", "max"),
    ("trading_max_adv_pct", "max"),
    ("trading_max_cycle_turnover_pct", "max"),
    ("brain_max_new_positions_per_cycle", "max"),
    ("trading_min_price", "min"),
    ("trading_min_dollar_volume", "min"),
    ("trading_cash_buffer_pct", "min"),
)


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str  # never a secret: masked values only


@dataclass(frozen=True, slots=True)
class Report:
    deployment: str
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict[str, Any]:
        return {"deployment": self.deployment, "ok": self.ok, "checks": [asdict(c) for c in self.checks]}

    def lines(self) -> list[str]:
        head = "PASS" if self.ok else "FAIL"
        out = [f"QuantPulse preflight ({self.deployment}): {head}"]
        out += [f"  [{'ok' if c.ok else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks]
        if not self.ok:
            out.append("Nothing was started. Fix the FAIL lines in the environment file and start again.")
        return out


def mask_url(url: str) -> str:
    """A URL with any password replaced by ``***`` (``postgresql+asyncpg://qp:***@db:5432/quantpulse``)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable>"
    if parts.password is None:
        return url
    user = parts.username or ""
    host = parts.hostname or ""
    netloc = f"{user}:***@{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on", "paper")


def _secret(value: Any) -> str:
    return value.get_secret_value() if value is not None else ""


def _defaults() -> dict[str, Any]:
    return {name: field.default for name, field in Settings.model_fields.items()}


def run(settings: Settings, environ: Mapping[str, str] | None = None) -> Report:
    """Every check (whatever the deployment); the API enforces them only when ``QP_DEPLOYMENT=cloud``."""
    env = {k.upper(): v for k, v in (os.environ if environ is None else environ).items()}
    file_values = {k: v or "" for k, v in _read_env_file(settings._source_file).items()}
    everything = {**file_values, **env}
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(Check(name, bool(ok), detail))

    # --- paper only ---------------------------------------------------------------------------
    explicit = everything.get("QP_ALPACA_PAPER")
    add(
        "paper_setting",
        settings.alpaca_paper and _truthy(explicit),
        "QP_ALPACA_PAPER=true (set explicitly)"
        if _truthy(explicit)
        else "QP_ALPACA_PAPER must be set to true explicitly in the cloud environment",
    )
    live = sorted(k for k, v in everything.items() if v and LIVE_ENDPOINT.search(v))
    add(
        "no_live_endpoint",
        not live,
        "no variable names Alpaca's live-money or broker API"
        if not live
        else f"{', '.join(live)} name(s) Alpaca's live-money API: remove them (QuantPulse trades paper only)",
    )
    wrong = sorted(
        k
        for k in ENDPOINT_VARIABLES
        if (everything.get(k) or "").strip()
        and everything[k].strip().rstrip("/").removesuffix("/v2").rstrip("/") != PAPER_URL
    )
    wrong += sorted(
        k for k in PAPER_FLAGS if (everything.get(k) or "").strip() and not _truthy(everything[k])
    )
    add(
        "paper_endpoint_variables",
        not wrong,
        f"every Alpaca endpoint variable that is set names {PAPER_URL}"
        if not wrong
        else f"{', '.join(wrong)} do(es) not name the paper API ({PAPER_URL}): fix or remove",
    )
    key, secret = _secret(settings.alpaca_api_key_id), _secret(settings.alpaca_api_secret_key)
    add(
        "paper_key",
        key.startswith("PK") and len(secret) >= 16,
        "paper key id (starts with PK) and a secret are set"
        if key.startswith("PK") and len(secret) >= 16
        else (
            "Alpaca paper keys are missing (QP_ALPACA_API_KEY_ID / QP_ALPACA_API_SECRET_KEY)"
            if not key or not secret
            else "the key id does not look like an Alpaca PAPER key (paper key ids start with 'PK')"
            if not key.startswith("PK")
            else "the Alpaca secret looks incomplete"
        ),
    )
    try:
        endpoint = AlpacaPaperBroker(key or None, secret or None).verify_paper_client()
        add("paper_client", endpoint == PAPER_URL, f"the trading client points at {endpoint}")
    except BrokerError as exc:
        add("paper_client", False, f"the paper trading client could not be built: {type(exc).__name__}")
    data_url = settings.alpaca_data_url.rstrip("/")
    add(
        "data_endpoint",
        data_url == DATA_URL,
        f"market data from {DATA_URL}"
        if data_url == DATA_URL
        else f"QP_ALPACA_DATA_URL must be {DATA_URL} (the keys are sent there)",
    )

    # --- access -------------------------------------------------------------------------------
    token = _secret(settings.api_token)
    weak = (
        "QP_API_TOKEN is not set"
        if not token
        else f"QP_API_TOKEN is shorter than {MIN_TOKEN} characters"
        if len(token) < MIN_TOKEN
        else "QP_API_TOKEN looks like a placeholder"
        if any(p in token.lower() for p in PLACEHOLDERS)
        else "QP_API_TOKEN must differ from the Alpaca credentials"
        if token in (key, secret)
        else ""
    )
    add("api_token", not weak, weak or f"set ({len(token)} characters); required on every API request")
    # The dashboard enforces its own password (it stays locked without one); a hash given to this service too
    # must at least be a valid one.
    hashed = _secret(settings.dashboard_password_hash)
    problem = ""
    if hashed:
        try:
            parse(hashed)
        except PasswordHashError as exc:
            problem = f"QP_DASHBOARD_PASSWORD_HASH: {exc}"
    add(
        "dashboard_password",
        not problem,
        problem
        or (
            "a PBKDF2 password hash is set"
            if hashed
            else "enforced by the dashboard service (locked without one)"
        ),
    )

    # --- ownership: the Brain alone trades the paper account in the cloud ------------------------
    add(
        "brain_mode",
        settings.brain_mode == "paper_execution",
        "QP_BRAIN_MODE=paper_execution: the Brain owns the paper account (every order through the trading service)"
        if settings.brain_mode == "paper_execution"
        else f"QP_BRAIN_MODE={settings.brain_mode}: the cloud deployment runs the Brain as the account owner "
        "(paper_execution); to observe only, keep it and set QP_ALPACA_TRADING_ENABLED=false or the kill switch",
    )
    add(
        "one_scheduler",
        not settings.trading_scheduler_enabled,
        "QP_TRADING_SCHEDULER_ENABLED=false: the Brain supervisor alone schedules (the old strategy is a shadow)"
        if not settings.trading_scheduler_enabled
        else "QP_TRADING_SCHEDULER_ENABLED=true: two schedulers for one account is ambiguous; the Brain supervisor "
        "schedules in the cloud — set it to false",
    )
    ambiguous = [
        f"{field} ({', '.join(names)})"
        for field, names in CREDENTIAL_ALIASES.items()
        if len({(everything.get(n) or "").strip() for n in names if (everything.get(n) or "").strip()}) > 1
    ]
    add(
        "unambiguous_credentials",
        not ambiguous,
        "each Alpaca credential is set once (or its aliases agree)"
        if not ambiguous
        else "set to different values under different names: " + "; ".join(ambiguous) + " — keep one",
    )

    # --- storage ------------------------------------------------------------------------------
    url = settings.database_url
    add(
        "database",
        url.startswith("postgresql+asyncpg://"),
        f"PostgreSQL ({mask_url(url)})"
        if url.startswith("postgresql+asyncpg://")
        else "QP_DATABASE_URL must be PostgreSQL (postgresql+asyncpg://…) in the cloud",
    )

    # --- protected controls -------------------------------------------------------------------
    defaults = _defaults()
    looser: list[str] = []
    for name, rule in PROTECTED:
        value, shipped = getattr(settings, name), defaults[name]
        if (
            (rule == "equal" and value != shipped)
            or (rule == "max" and value > shipped)
            or (rule == "min" and value < shipped)
        ):
            looser.append(f"QP_{name.upper()}={value} (protected: {shipped})")
    add(
        "protected_controls",
        not looser,
        "every protected risk control is at or stricter than the shipped limit"
        if not looser
        else "looser than the protected limits: " + "; ".join(looser),
    )
    return Report(settings.deployment, tuple(checks))


def enforce(settings: Settings, environ: Mapping[str, str] | None = None) -> Report | None:
    """In the cloud: run the preflight and raise :class:`PreflightFailed` unless it passes."""
    if settings.deployment != "cloud":
        return None
    report = run(settings, environ)
    if not report.ok:
        raise PreflightFailed(report)
    return report


class PreflightFailed(RuntimeError):
    def __init__(self, report: Report) -> None:
        super().__init__("; ".join(f"{c.name}: {c.detail}" for c in report.failures))
        self.report = report
