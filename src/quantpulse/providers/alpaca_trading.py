"""Alpaca **paper** trading through Alpaca's official Python SDK (``alpaca-py``).

Paper only, by construction
    * the SDK client is always built as ``TradingClient(key, secret, paper=True)`` with no URL override;
    * once built, its base URL is compared with Alpaca's paper endpoint and the broker refuses to work if
      they differ;
    * no argument, setting or code path selects Alpaca's live-money endpoint.

The account, positions, orders and fills reported here are authoritative: QuantPulse's own database is
only a record, reconciled against this broker.

Options go through the same client and the same checks: a single-leg option order or a multi-leg (``mleg``)
order of two to four option legs, always a DAY *limit* order in whole contracts, each leg with its position
intent (buy/sell to open/close). A multi-leg order may never sell more contracts of a kind (calls, puts)
than it buys: no naked short leg can be expressed here (the risk engine checks coverage by shares and cash
as well).

The SDK is synchronous (``requests``); every call runs in a worker thread with a hard HTTP timeout.
Credentials are held by the SDK client only; they never appear in errors, logs or ``repr``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeVar

import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OptionLegRequest,
)

from quantpulse.core.errors import QuantPulseError
from quantpulse.options.contracts import is_option_symbol, parse_occ

logger = logging.getLogger(__name__)

NAME = "alpaca_paper"
PAPER_URL = "https://paper-api.alpaca.markets"
DEFAULT_TIMEOUT = 10.0
# Terminal order states: nothing more will happen to the order.
TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"})

# Alpaca accepts fractional quantities to 9 decimal places and notional amounts to the cent.
QTY_DECIMALS = 9
T = TypeVar("T")
Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
AssetClass = Literal["us_equity", "us_option"]
Intent = Literal["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"]
INTENTS: tuple[Intent, ...] = ("buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close")
MAX_LEGS = 4
MLEG_MARK = ":MLEG"  # a multi-leg order's record symbol: its underlying and this mark (AAPL:MLEG)


def mleg_symbol(leg_symbols: list[str] | tuple[str, ...]) -> str:
    """The symbol QuantPulse records a multi-leg order under (Alpaca reports none for it)."""
    roots = sorted({parse_occ(x).underlying for x in leg_symbols if is_option_symbol(x)})
    return (roots[0] if len(roots) == 1 else "MULTI") + MLEG_MARK


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


class InvalidOrder(BrokerError):
    """The order could not even be expressed as an Alpaca request (it was never sent)."""


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
    # the account's effective options level (0 none, 1 covered calls / cash-secured puts, 2 long calls and
    # puts, 3 spreads) and options buying power; None when Alpaca does not report them
    options_trading_level: int | None = None
    options_buying_power: float | None = None

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
    asset_class: str = "us_equity"

    @property
    def is_option(self) -> bool:
        return self.asset_class == "us_option" or is_option_symbol(self.symbol)


@dataclass(frozen=True, slots=True)
class BrokerOrderLeg:
    """One leg of a multi-leg option order, as Alpaca reports it."""

    symbol: str
    side: str
    ratio_qty: float
    position_intent: str | None
    status: str
    qty: float | None
    filled_qty: float
    filled_avg_price: float | None


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
    asset_class: str = "us_equity"
    order_class: str = "simple"
    position_intent: str | None = None
    legs: tuple[BrokerOrderLeg, ...] = ()

    @property
    def is_option(self) -> bool:
        return self.asset_class == "us_option" or bool(self.legs) or is_option_symbol(self.symbol)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every symbol the order trades: its legs' for a multi-leg order, else its own."""
        return tuple(leg.symbol for leg in self.legs) if self.legs else (self.symbol,)

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
class OrderLeg:
    """One leg of a multi-leg option order: ``ratio_qty`` contracts per unit of the order's quantity."""

    symbol: str
    side: Side
    ratio_qty: int
    position_intent: Intent

    def __post_init__(self) -> None:
        if not is_option_symbol(self.symbol):
            raise ValueError(f"a leg must be an option contract (OCC symbol), not {self.symbol!r}")
        if self.side not in ("buy", "sell"):
            raise ValueError("a leg's side is buy or sell")
        if self.position_intent not in INTENTS:
            raise ValueError(f"unknown position intent {self.position_intent!r}")
        if not self.position_intent.startswith(self.side):
            raise ValueError(f"a {self.side} leg cannot {self.position_intent.replace('_', ' ')}")
        if isinstance(self.ratio_qty, bool) or int(self.ratio_qty) != self.ratio_qty or self.ratio_qty < 1:
            raise ValueError("a leg's ratio is a whole number of contracts, at least 1")


def naked_short_legs(legs: tuple[OrderLeg, ...] | list[OrderLeg]) -> list[str]:
    """Kinds (call, put) of which a multi-leg order *opens* more short contracts than long ones."""
    out = []
    for kind in ("call", "put"):
        short = sum(
            x.ratio_qty
            for x in legs
            if x.position_intent == "sell_to_open" and parse_occ(x.symbol).kind == kind
        )
        long = sum(
            x.ratio_qty
            for x in legs
            if x.position_intent == "buy_to_open" and parse_occ(x.symbol).kind == kind
        )
        if short > long:
            out.append(kind)
    return out


@dataclass(frozen=True, slots=True)
class OrderSpec:
    """One order as QuantPulse wants it placed (always a regular-hours DAY order).

    Stocks: exactly one of ``qty`` (shares, fractional allowed) and ``notional`` (dollars, market orders
    only) is set. Quantities are rounded to Alpaca's 9 decimal places, so a float such as
    0.30000000000000004 is sent as 0.3.

    Options (``asset_class="us_option"``): a limit order for a whole number of contracts, with a position
    intent. A multi-leg order carries two to four ``legs`` (unique contracts); its ``symbol`` is the record
    symbol (``AAPL:MLEG``) and its ``limit_price`` the net price per unit — positive a debit paid,
    negative a credit received, as Alpaca expects."""

    symbol: str
    side: Side
    qty: float | None
    order_type: OrderType
    client_order_id: str
    limit_price: float | None = None
    notional: float | None = None
    asset_class: AssetClass = "us_equity"
    position_intent: Intent | None = None
    legs: tuple[OrderLeg, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= len(self.client_order_id) <= 128:
            raise ValueError("client order ids are 1-128 characters")
        if self.asset_class == "us_option":
            self._check_option()
            return
        if self.asset_class != "us_equity":
            raise ValueError(f"unknown asset class {self.asset_class!r}")
        if self.legs or self.position_intent is not None:
            raise ValueError("legs and position intents are for option orders only")
        if is_option_symbol(self.symbol):
            raise ValueError(f"{self.symbol} is an option contract: send it as an option order")
        if (self.qty is None) == (self.notional is None):
            raise ValueError("an order needs exactly one of a quantity or a notional amount")
        if self.qty is not None:
            object.__setattr__(self, "qty", round(float(self.qty), QTY_DECIMALS))
            if not self.qty > 0:
                raise ValueError("order quantity must be positive")
        if self.notional is not None:
            object.__setattr__(self, "notional", round(float(self.notional), 2))
            if not self.notional > 0:
                raise ValueError("order notional must be positive")
            if self.order_type != "market":
                raise ValueError("a notional (dollar-amount) order must be a market order")
        if self.order_type == "limit" and (self.limit_price is None or not self.limit_price > 0):
            raise ValueError("a limit order needs a positive limit price")

    def _check_option(self) -> None:
        if self.order_type != "limit":
            raise ValueError("option orders are limit orders only")
        if self.notional is not None:
            raise ValueError("option orders are sized in contracts, never in dollars")
        if self.qty is None or int(self.qty) != self.qty or not self.qty >= 1:
            raise ValueError("an option order is for a whole number of contracts, at least 1")
        object.__setattr__(self, "qty", float(int(self.qty)))
        price = self.limit_price
        if price is None or not math.isfinite(price):
            raise ValueError("an option order needs a limit price")
        if self.legs:
            if not 2 <= len(self.legs) <= MAX_LEGS:
                raise ValueError(f"a multi-leg order has 2 to {MAX_LEGS} legs")
            if len({x.symbol for x in self.legs}) != len(self.legs):
                raise ValueError("a multi-leg order's legs must be different contracts")
            if self.position_intent is not None:
                raise ValueError("a multi-leg order's intents are on its legs")
            naked = naked_short_legs(self.legs)
            if naked:
                raise ValueError(f"the order would open naked short {' and '.join(naked)}s: refused")
            if round(abs(price), 2) == 0 and price != 0:
                raise ValueError("the net limit price rounds to zero")
            return
        if not is_option_symbol(self.symbol):
            raise ValueError(f"{self.symbol!r} is not an option contract (OCC symbol)")
        if self.position_intent not in INTENTS:
            raise ValueError("a single-leg option order needs its position intent (buy/sell to open/close)")
        if not self.position_intent.startswith(self.side):
            raise ValueError(f"a {self.side} order cannot {self.position_intent.replace('_', ' ')}")
        if not price > 0:
            raise ValueError("a single-leg option limit price is positive")


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


def _int(value: Any) -> int | None:
    try:
        return None if value is None or value == "" else int(value)
    except (TypeError, ValueError):
        return None


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _mask(account_number: str | None) -> str:
    s = account_number or ""
    return f"…{s[-4:]}" if len(s) > 4 else s


def _option_request(spec: OrderSpec, side: OrderSide) -> LimitOrderRequest:
    price = round(float(spec.limit_price or 0.0), 2)
    if spec.legs:
        return LimitOrderRequest(
            qty=spec.qty,
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            limit_price=price,
            client_order_id=spec.client_order_id,
            legs=[
                OptionLegRequest(
                    symbol=x.symbol,
                    ratio_qty=x.ratio_qty,
                    side=OrderSide.BUY if x.side == "buy" else OrderSide.SELL,
                    position_intent=PositionIntent(x.position_intent),
                )
                for x in spec.legs
            ],
        )
    return LimitOrderRequest(
        symbol=spec.symbol,
        qty=spec.qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        limit_price=price,
        client_order_id=spec.client_order_id,
        position_intent=PositionIntent(spec.position_intent or ""),
    )


def order_request(spec: OrderSpec) -> MarketOrderRequest | LimitOrderRequest:
    """The SDK request for ``spec``; ``InvalidOrder`` (never sent) if the SDK refuses to build it."""
    side = OrderSide.BUY if spec.side == "buy" else OrderSide.SELL
    try:
        if spec.asset_class == "us_option":
            return _option_request(spec, side)
        if spec.order_type == "limit":
            return LimitOrderRequest(
                symbol=spec.symbol,
                qty=spec.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(float(spec.limit_price or 0.0), 2 if (spec.limit_price or 0) >= 1 else 4),
                client_order_id=spec.client_order_id,
            )
        return MarketOrderRequest(
            symbol=spec.symbol,
            qty=spec.qty,
            notional=spec.notional,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=spec.client_order_id,
        )
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        first = str(exc).strip().splitlines()
        detail = first[-1] if first else type(exc).__name__
        raise InvalidOrder(
            f"Alpaca paper order for {spec.symbol} could not be built (never sent): {detail[:200]}"
        ) from None


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
        options_trading_level=_int(getattr(a, "options_trading_level", None)),
        options_buying_power=_opt(getattr(a, "options_buying_power", None)),
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
        asset_class=_enum(getattr(p, "asset_class", None)) or "us_equity",
    )


def _leg_from_sdk(o: Any) -> BrokerOrderLeg:
    return BrokerOrderLeg(
        symbol=o.symbol or "",
        side=_enum(o.side),
        ratio_qty=_f(getattr(o, "ratio_qty", None), 1.0),
        position_intent=_enum(getattr(o, "position_intent", None)) or None,
        status=_enum(o.status),
        qty=_opt(o.qty),
        filled_qty=_f(o.filled_qty),
        filled_avg_price=_opt(o.filled_avg_price),
    )


def order_from_sdk(o: Any) -> BrokerOrder:
    legs = tuple(_leg_from_sdk(x) for x in (getattr(o, "legs", None) or []))
    order_class = _enum(getattr(o, "order_class", None)) or "simple"
    mleg = order_class == "mleg"
    symbol = o.symbol or ""
    if mleg and legs:
        symbol = mleg_symbol([x.symbol for x in legs])
    asset_class = _enum(getattr(o, "asset_class", None)) or ("us_option" if mleg else "us_equity")
    if is_option_symbol(symbol):
        asset_class = "us_option"
    side = _enum(o.side)
    if mleg and not side:
        # a multi-leg order has no side of its own: a debit is paid (buy), a credit received (sell)
        side = "sell" if (_opt(o.limit_price) or 0.0) < 0 else "buy"
    return BrokerOrder(
        id=str(o.id),
        client_order_id=o.client_order_id,
        symbol=symbol,
        side=side,
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
        asset_class=asset_class,
        order_class=order_class,
        position_intent=_enum(getattr(o, "position_intent", None)) or None,
        legs=legs,
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

    def verify_paper_client(self) -> str:
        """Build the SDK client (no network call) and return the endpoint it points at — always the paper
        API, or ``NotPaperTrading`` / ``BrokerNotConfigured`` is raised."""
        client = self._sdk()
        raw = getattr(client, "_base_url", "")
        return str(getattr(raw, "value", raw)).rstrip("/")

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
            # a multi-leg order's legs come nested under it (never as orders of their own)
            nested=True,
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
        """Send one order (``POST /v2/orders``). An ``InvalidOrder`` was never sent; an ambiguous
        ``BrokerError`` may have been: look it up by its client order id, never resend it."""
        request = order_request(spec)
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
