"""Alpaca **paper** trading through Alpaca's official Python SDK (``alpaca-py``).

Paper only, by construction
    * the SDK client is always built as ``TradingClient(key, secret, paper=True)`` with no URL override;
    * once built, its base URL is compared with Alpaca's paper endpoint and the broker refuses to work if
      they differ;
    * no argument, setting or code path selects Alpaca's live-money endpoint.

The account, positions, orders and fills reported here are authoritative: QuantPulse's own database is
only a record, reconciled against this broker.

The SDK is synchronous (``requests``); every call runs in a worker thread with a hard HTTP timeout.
Credentials are held by the SDK client only; they never appear in errors, logs or ``repr``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeVar

import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from quantpulse.core.errors import QuantPulseError

logger = logging.getLogger(__name__)

NAME = "alpaca_paper"
PAPER_URL = "https://paper-api.alpaca.markets"
DEFAULT_TIMEOUT = 10.0
# Terminal order states: nothing more will happen to the order.
TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"})

T = TypeVar("T")
Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


# ----------------------------------------------------------------------------- errors
class BrokerError(QuantPulseError):
    """An Alpaca paper-trading call failed.

    ``ambiguous`` is true when the request may or may not have reached Alpaca (a timeout or a dropped
    connection): an order submitted that way must be looked up by its client order id, never resent."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.ambiguous = ambiguous


class BrokerNotConfigured(BrokerError):
    """Alpaca paper keys are missing."""


class BrokerNotFound(BrokerError):
    """The order or position does not exist."""


class OrderRejected(BrokerError):
    """Alpaca refused the order (buying power, invalid quantity, market closed, …)."""


class DuplicateClientOrderId(OrderRejected):
    """An order with this client order id already exists: the earlier submission got through."""


class NotPaperTrading(BrokerError):
    """The SDK client does not point at Alpaca's paper endpoint (should be impossible)."""


# ----------------------------------------------------------------------------- data
@dataclass(frozen=True, slots=True)
class BrokerAccount:
    account_number: str  # masked: only the last four characters are kept
    status: str
    currency: str
    equity: float
    last_equity: float  # equity at the previous close (Alpaca's day P/L reference)
    cash: float
    buying_power: float
    long_market_value: float
    short_market_value: float
    portfolio_value: float
    trading_blocked: bool
    account_blocked: bool
    trade_suspended_by_user: bool
    pattern_day_trader: bool
    daytrade_count: int
    multiplier: float

    @property
    def day_pl(self) -> float:
        return self.equity - self.last_equity

    @property
    def day_pl_pct(self) -> float:
        return self.equity / self.last_equity - 1.0 if self.last_equity > 0 else 0.0

    @property
    def blocked(self) -> bool:
        return self.trading_blocked or self.account_blocked or self.trade_suspended_by_user


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    symbol: str
    qty: float
    qty_available: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    unrealized_intraday_pl: float
    lastday_price: float | None


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    status: str
    qty: float | None
    notional: float | None
    filled_qty: float
    filled_avg_price: float | None
    limit_price: float | None
    created_at: datetime | None
    submitted_at: datetime | None
    updated_at: datetime | None
    filled_at: datetime | None
    canceled_at: datetime | None
    expired_at: datetime | None
    failed_at: datetime | None

    @property
    def is_open(self) -> bool:
        return self.status not in TERMINAL_STATUSES

    @property
    def remaining_qty(self) -> float:
        return max((self.qty or 0.0) - self.filled_qty, 0.0)


@dataclass(frozen=True, slots=True)
class MarketClock:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True, slots=True)
class OrderSpec:
    """One order as QuantPulse wants it placed (always a regular-hours DAY order)."""

    symbol: str
    side: Side
    qty: float
    order_type: OrderType
    client_order_id: str
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("order quantity must be positive")
        if self.order_type == "limit" and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("a limit order needs a positive limit price")
        if not 1 <= len(self.client_order_id) <= 128:
            raise ValueError("client order ids are 1-128 characters")


# ----------------------------------------------------------------------------- helpers
def _f(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _opt(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _mask(account_number: str | None) -> str:
    s = account_number or ""
    return f"…{s[-4:]}" if len(s) > 4 else s


def account_from_sdk(a: Any) -> BrokerAccount:
    return BrokerAccount(
        account_number=_mask(a.account_number),
        status=_enum(a.status),
        currency=a.currency or "USD",
        equity=_f(a.equity),
        last_equity=_f(a.last_equity),
        cash=_f(a.cash),
        buying_power=_f(a.buying_power),
        long_market_value=_f(a.long_market_value),
        short_market_value=_f(a.short_market_value),
        portfolio_value=_f(a.portfolio_value, _f(a.equity)),
        trading_blocked=bool(a.trading_blocked),
        account_blocked=bool(a.account_blocked),
        trade_suspended_by_user=bool(a.trade_suspended_by_user),
        pattern_day_trader=bool(a.pattern_day_trader),
        daytrade_count=int(a.daytrade_count or 0),
        multiplier=_f(a.multiplier, 1.0),
    )


def position_from_sdk(p: Any) -> BrokerPosition:
    return BrokerPosition(
        symbol=p.symbol,
        qty=_f(p.qty),
        qty_available=_f(p.qty_available, _f(p.qty)),
        side=_enum(p.side),
        avg_entry_price=_f(p.avg_entry_price),
        current_price=_f(p.current_price),
        market_value=_f(p.market_value),
        cost_basis=_f(p.cost_basis),
        unrealized_pl=_f(p.unrealized_pl),
        unrealized_plpc=_f(p.unrealized_plpc),
        unrealized_intraday_pl=_f(p.unrealized_intraday_pl),
        lastday_price=_opt(p.lastday_price),
    )


def order_from_sdk(o: Any) -> BrokerOrder:
    return BrokerOrder(
        id=str(o.id),
        client_order_id=o.client_order_id,
        symbol=o.symbol or "",
        side=_enum(o.side),
        order_type=_enum(o.order_type or o.type),
        time_in_force=_enum(o.time_in_force),
        status=_enum(o.status),
        qty=_opt(o.qty),
        notional=_opt(o.notional),
        filled_qty=_f(o.filled_qty),
        filled_avg_price=_opt(o.filled_avg_price),
        limit_price=_opt(o.limit_price),
        created_at=o.created_at,
        submitted_at=o.submitted_at,
        updated_at=o.updated_at,
        filled_at=o.filled_at,
        canceled_at=o.canceled_at,
        expired_at=o.expired_at,
        failed_at=o.failed_at,
    )


def _api_message(exc: APIError) -> tuple[str, int | None]:
    try:
        return str(exc.message), int(exc.code)
    except Exception:  # the body was not Alpaca's JSON error shape
        return str(exc)[:300], None


def _translate(exc: Exception, what: str) -> BrokerError:
    """Map SDK / transport failures onto QuantPulse's broker errors (no credentials in any message)."""
    if isinstance(exc, BrokerError):
        return exc
    if isinstance(exc, APIError):
        message, code = _api_message(exc)
        status = exc.status_code
        text = f"Alpaca paper {what} failed (HTTP {status}): {message}"
        if status == 404:
            return BrokerNotFound(text, status_code=status, code=code)
        if status == 401:
            return BrokerNotConfigured(
                f"Alpaca paper {what} failed: the API keys were refused (HTTP 401). Check that "
                "QP_ALPACA_API_KEY_ID / QP_ALPACA_API_SECRET_KEY hold *paper* keys.",
                status_code=status,
                code=code,
            )
        lowered = message.lower()
        if status == 422 and "client_order_id" in lowered and "unique" in lowered:
            return DuplicateClientOrderId(text, status_code=status, code=code)
        if status in (403, 422):
            return OrderRejected(text, status_code=status, code=code)
        return BrokerError(text, status_code=status, code=code, ambiguous=status is None or status >= 500)
    if isinstance(exc, requests.Timeout | requests.ConnectionError):
        return BrokerError(
            f"Alpaca paper {what}: no answer ({type(exc).__name__}); the outcome is unknown",
            ambiguous=True,
        )
    return BrokerError(f"Alpaca paper {what} failed: {type(exc).__name__}", ambiguous=True)


class _TimeoutSession(requests.Session):
    """``requests`` has no default timeout; the SDK sets none, so every call gets ours."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, method: Any, url: Any, *args: Any, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        return super().request(method, url, *args, **kwargs)


# ----------------------------------------------------------------------------- broker
class AlpacaPaperBroker:
    """The Alpaca paper account. Every method is async; SDK calls run in a worker thread."""

    name = NAME

    def __init__(
        self,
        key_id: str | None,
        secret: str | None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: requests.adapters.BaseAdapter | None = None,
    ) -> None:
        self._key_id = key_id or None
        self._secret = secret or None
        self._timeout = timeout
        self._transport = transport  # tests mount a fake Alpaca here; production uses the network
        self._client: TradingClient | None = None

    def __repr__(self) -> str:  # never show credentials
        return f"AlpacaPaperBroker(configured={self.configured()}, url={PAPER_URL!r})"

    def configured(self) -> bool:
        return bool(self._key_id and self._secret)

    @property
    def base_url(self) -> str:
        return PAPER_URL

    def _sdk(self) -> TradingClient:
        if self._client is not None:
            return self._client
        if not self.configured():
            raise BrokerNotConfigured(
                "Alpaca paper trading is not configured: set QP_ALPACA_API_KEY_ID and "
                "QP_ALPACA_API_SECRET_KEY (paper keys) in .env"
            )
        client = TradingClient(self._key_id, self._secret, paper=True)
        raw = getattr(client, "_base_url", "")
        actual = str(getattr(raw, "value", raw))  # the SDK stores its BaseURL enum
        if actual.rstrip("/") != PAPER_URL:
            raise NotPaperTrading(
                f"refusing to trade: the Alpaca client points at {actual!r}, not the paper API"
            )
        session = _TimeoutSession(self._timeout)
        if self._transport is not None:
            session.mount("https://", self._transport)
        client._session = session
        # A 504 on POST /orders is ambiguous: never let the SDK resend it. Only 429 (not processed) is retried;
        # anything else is resolved by looking the order up by its client order id.
        client._retry_codes = [429]
        self._client = client
        return client

    async def _call(self, what: str, fn: Callable[[TradingClient], T]) -> T:
        def run() -> T:
            return fn(self._sdk())

        try:
            return await asyncio.to_thread(run)
        except BrokerError:
            raise
        except Exception as exc:
            raise _translate(exc, what) from None

    # -- account -----------------------------------------------------------------------------
    async def account(self) -> BrokerAccount:
        return account_from_sdk(await self._call("account", lambda c: c.get_account()))

    async def clock(self) -> MarketClock:
        raw: Any = await self._call("clock", lambda c: c.get_clock())
        return MarketClock(
            timestamp=raw.timestamp,
            is_open=bool(raw.is_open),
            next_open=raw.next_open,
            next_close=raw.next_close,
        )

    async def positions(self) -> list[BrokerPosition]:
        raw = await self._call("positions", lambda c: c.get_all_positions())
        return sorted((position_from_sdk(p) for p in raw), key=lambda p: p.symbol)

    # -- orders ------------------------------------------------------------------------------
    async def orders(
        self,
        status: Literal["open", "closed", "all"] = "all",
        *,
        limit: int = 200,
        after: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> list[BrokerOrder]:
        request = GetOrdersRequest(
            status=QueryOrderStatus(status),
            limit=min(max(limit, 1), 500),
            after=after,
            symbols=symbols,
            nested=False,
        )
        raw = await self._call("orders", lambda c: c.get_orders(filter=request))
        return [order_from_sdk(o) for o in raw]

    async def open_orders(self) -> list[BrokerOrder]:
        return await self.orders("open", limit=500)

    async def order(self, order_id: str) -> BrokerOrder:
        return order_from_sdk(await self._call("order lookup", lambda c: c.get_order_by_id(order_id)))

    async def order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """The order placed under ``client_order_id``, or ``None`` if Alpaca never received it."""
        try:
            raw = await self._call("order lookup", lambda c: c.get_order_by_client_id(client_order_id))
        except BrokerNotFound:
            return None
        return order_from_sdk(raw)

    async def submit(self, spec: OrderSpec) -> BrokerOrder:
        side = OrderSide.BUY if spec.side == "buy" else OrderSide.SELL
        request: MarketOrderRequest | LimitOrderRequest
        if spec.order_type == "limit":
            request = LimitOrderRequest(
                symbol=spec.symbol,
                qty=spec.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(float(spec.limit_price or 0.0), 2),
                client_order_id=spec.client_order_id,
            )
        else:
            request = MarketOrderRequest(
                symbol=spec.symbol,
                qty=spec.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=spec.client_order_id,
            )
        return order_from_sdk(await self._call("order submission", lambda c: c.submit_order(request)))

    async def cancel(self, order_id: str) -> None:
        await self._call("order cancel", lambda c: c.cancel_order_by_id(order_id))

    async def cancel_all(self) -> int:
        """Cancel every open order on the paper account; the number of orders Alpaca accepted to cancel."""
        responses = await self._call("cancel all", lambda c: c.cancel_orders())
        return sum(1 for r in responses or [] if 200 <= int(getattr(r, "status", 200)) < 300)

    async def close_position(self, symbol: str, qty: float | None = None) -> BrokerOrder:
        """Alpaca's own market order closing ``symbol`` (all of it, or ``qty`` shares)."""
        opts = ClosePositionRequest(qty=str(qty)) if qty is not None else None
        raw = await self._call("close position", lambda c: c.close_position(symbol, close_options=opts))
        return order_from_sdk(raw)

    async def close_all_positions(self, cancel_orders: bool = True) -> int:
        """Alpaca's own close-everything call; the number of positions it started closing."""
        responses = await self._call(
            "close all positions", lambda c: c.close_all_positions(cancel_orders=cancel_orders)
        )
        return sum(1 for r in responses or [] if 200 <= int(getattr(r, "status", 200)) < 300)
