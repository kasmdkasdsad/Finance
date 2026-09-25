"""Risk engine for Alpaca paper trading. Every order passes through :meth:`RiskBook.evaluate` before it can
reach the broker, and every decision records each named check so the dashboard can show why.

Checks (an order is approved only if all pass)
    ``kill_switch``      no order while the kill switch is on (except explicit flatten orders);
    ``account``          Alpaca has not blocked the account;
    ``market_open``      the regular session is open;
    ``live_data``        the symbol has a live quote no older than ``max_quote_age_seconds`` — never synthetic
                         (and never stale when ``require_live_data``);
    ``no_working_order`` no other order for the symbol is still working (no stacking, no duplicates);
    ``no_short``         a sell never exceeds the shares held (long-only);
    ``order_size``       ``min_order_notional`` ≤ notional ≤ ``max_order_notional`` (orders closing a whole
                         position are exempt from the cap so a stop-loss can always execute);
buys only:
    ``daily_loss``       no new exposure once the day's loss reaches ``max_daily_loss_pct``;
    ``position_limit``   the position stays within ``max_position_pct`` of equity;
    ``total_exposure``   long exposure stays within ``max_total_exposure_pct`` of equity;
    ``max_positions``    no more than ``max_positions`` names (held + pending);
    ``buying_power``     the order fits in buying power and in cash above the ``cash_buffer_pct`` reserve;
    ``liquidity``        price ≥ ``min_price``, 20-day dollar volume ≥ ``min_dollar_volume``, spread ≤
                         ``max_spread_bps``.

The book is *projected*: each approved order is committed so later orders in the same cycle see the
exposure, positions and cash it will use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from quantpulse.providers.alpaca_trading import BrokerAccount, BrokerOrder, BrokerPosition
from quantpulse.schemas.common import DataStatus

TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_position_pct: float = 0.30
    max_total_exposure_pct: float = 0.95
    max_order_notional: float = 15_000.0
    min_order_notional: float = 100.0
    max_positions: int = 8
    max_daily_loss_pct: float = 0.04
    max_position_loss_pct: float = 0.08
    cash_buffer_pct: float = 0.02
    min_price: float = 5.0
    min_dollar_volume: float = 25_000_000.0
    max_spread_bps: float = 30.0
    max_quote_age_seconds: float = 600.0
    require_live_data: bool = True
    allow_shorts: bool = False

    @classmethod
    def from_settings(cls, s: Any) -> RiskLimits:
        return cls(
            max_position_pct=s.trading_max_position_pct,
            max_total_exposure_pct=s.trading_max_total_exposure_pct,
            max_order_notional=s.trading_max_order_notional,
            min_order_notional=s.trading_min_order_notional,
            max_positions=s.trading_max_positions,
            max_daily_loss_pct=s.trading_max_daily_loss_pct,
            max_position_loss_pct=s.trading_max_position_loss_pct,
            cash_buffer_pct=s.trading_cash_buffer_pct,
            min_price=s.trading_min_price,
            min_dollar_volume=s.trading_min_dollar_volume,
            max_spread_bps=s.trading_max_spread_bps,
            max_quote_age_seconds=s.trading_max_quote_age_seconds,
            require_live_data=s.trading_require_live_data,
            allow_shorts=s.trading_allow_shorts,
        )


@dataclass(frozen=True, slots=True)
class QuoteCheck:
    """What the risk engine knows about a symbol's market data."""

    price: float
    status: DataStatus
    provider: str
    age_seconds: float | None = None
    spread_bps: float | None = None
    adv_dollar: float | None = None


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: str  # buy | sell
    qty: float
    est_price: float
    kind: str
    reason: str
    closes_position: bool = False
    score: float | None = None
    intent: str = "strategy"  # strategy | flatten (manual close-all or the daily-loss emergency policy)

    @property
    def notional(self) -> float:
        return self.qty * self.est_price


@dataclass(frozen=True, slots=True)
class RiskCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RiskDecision:
    intent: OrderIntent
    checks: tuple[RiskCheck, ...]

    @property
    def approved(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[RiskCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def summary(self) -> str:
        if self.approved:
            return "approved"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures)


@dataclass
class RiskBook:
    limits: RiskLimits
    account: BrokerAccount
    positions: Mapping[str, BrokerPosition]
    open_orders: Sequence[BrokerOrder]
    market_open: bool
    kill_switch: bool
    quotes: Mapping[str, QuoteCheck]
    daily_loss_hit: bool = False
    held: dict[str, float] = field(init=False)
    value: dict[str, float] = field(init=False)
    working: set[str] = field(init=False)
    pending_new: set[str] = field(init=False)
    exposure: float = field(init=False)
    cash_left: float = field(init=False)
    buying_power_left: float = field(init=False)

    def __post_init__(self) -> None:
        self.held = {s: p.qty for s, p in self.positions.items() if p.qty > 0}
        self.value = {s: p.market_value for s, p in self.positions.items()}
        self.working = {o.symbol for o in self.open_orders if o.is_open}
        self.pending_new = set()
        pending_buys = 0.0
        for o in self.open_orders:
            if not o.is_open:
                continue
            price = o.limit_price or (self.quotes[o.symbol].price if o.symbol in self.quotes else 0.0)
            if o.side == "buy":
                amount = o.remaining_qty * price
                pending_buys += amount
                self.value[o.symbol] = self.value.get(o.symbol, 0.0) + amount
                if o.symbol not in self.held:
                    self.pending_new.add(o.symbol)
            else:
                self.held[o.symbol] = self.held.get(o.symbol, 0.0) - o.remaining_qty
        self.exposure = self.account.long_market_value + pending_buys
        self.cash_left = self.account.cash - pending_buys
        self.buying_power_left = self.account.buying_power - pending_buys
        if self.account.last_equity > 0 and self.account.day_pl_pct <= -self.limits.max_daily_loss_pct:
            self.daily_loss_hit = True

    @property
    def equity(self) -> float:
        return self.account.equity

    def evaluate(self, o: OrderIntent) -> RiskDecision:
        L = self.limits
        checks: list[RiskCheck] = []

        def check(name: str, ok: bool, detail: str) -> None:
            checks.append(RiskCheck(name, bool(ok), detail))

        flatten = o.intent == "flatten"
        if self.kill_switch:
            check(
                "kill_switch",
                flatten,
                "kill switch is on; only explicit flatten orders may pass"
                if flatten
                else "kill switch is ON",
            )
        else:
            check("kill_switch", True, "off")
        check("account", not self.account.blocked, "blocked by Alpaca" if self.account.blocked else "active")
        check("market_open", self.market_open, "market open" if self.market_open else "market is closed")

        q = self.quotes.get(o.symbol)
        allowed = {DataStatus.LIVE, DataStatus.CACHED}
        if not L.require_live_data:
            allowed.add(DataStatus.STALE)
        if q is None:
            check("live_data", False, "no live quote for this symbol")
        elif q.status is DataStatus.SYNTHETIC:
            check("live_data", False, "only synthetic prices are available: never traded")
        elif q.status not in allowed:
            check("live_data", False, f"quote is {q.status.value} ({q.provider}); live data is required")
        elif q.age_seconds is not None and q.age_seconds > L.max_quote_age_seconds:
            check(
                "live_data",
                False,
                f"quote is {q.age_seconds:.0f}s old (limit {L.max_quote_age_seconds:.0f}s)",
            )
        else:
            age = f", {q.age_seconds:.0f}s old" if q.age_seconds is not None else ""
            check("live_data", True, f"{q.status.value} quote from {q.provider}{age}")

        check("quantity", o.qty > 0 and o.est_price > 0, f"{o.qty:g} shares at ~${o.est_price:,.2f}")
        check(
            "no_working_order",
            o.symbol not in self.working,
            "another order for this symbol is still working" if o.symbol in self.working else "none",
        )
        notional = o.notional
        if o.side == "sell":
            held = self.held.get(o.symbol, 0.0)
            check(
                "no_short",
                o.qty <= held + TOLERANCE,
                f"selling {o.qty:g} of {held:g} held"
                if o.qty <= held + TOLERANCE
                else f"would sell {o.qty:g} but only {held:g} are held (short selling is disabled)",
            )
            if o.closes_position:
                check("order_size", True, f"${notional:,.0f} closes the position (exits are never capped)")
            else:
                check(
                    "order_size",
                    L.min_order_notional <= notional <= L.max_order_notional + TOLERANCE,
                    f"${notional:,.0f} (allowed ${L.min_order_notional:,.0f}–${L.max_order_notional:,.0f})",
                )
        elif o.side == "buy":
            equity = self.equity
            check(
                "daily_loss",
                not self.daily_loss_hit,
                f"day P/L {self.account.day_pl_pct:+.2%} reached the −{L.max_daily_loss_pct:.0%} limit: "
                "no new positions today"
                if self.daily_loss_hit
                else f"day P/L {self.account.day_pl_pct:+.2%} (limit −{L.max_daily_loss_pct:.0%})",
            )
            check(
                "order_size",
                L.min_order_notional <= notional <= L.max_order_notional + TOLERANCE,
                f"${notional:,.0f} (allowed ${L.min_order_notional:,.0f}–${L.max_order_notional:,.0f})",
            )
            after = self.value.get(o.symbol, 0.0) + notional
            check(
                "position_limit",
                equity > 0 and after <= L.max_position_pct * equity + 0.01,
                f"{after / equity:.1%} of equity after the order (limit {L.max_position_pct:.0%})"
                if equity > 0
                else "equity is not positive",
            )
            exposure = self.exposure + notional
            check(
                "total_exposure",
                equity > 0 and exposure <= L.max_total_exposure_pct * equity + 0.01,
                f"{exposure / equity:.1%} long exposure after the order (limit {L.max_total_exposure_pct:.0%})"
                if equity > 0
                else "equity is not positive",
            )
            is_new = self.held.get(o.symbol, 0.0) <= 0 and o.symbol not in self.pending_new
            count = len([s for s, q in self.held.items() if q > 0]) + len(self.pending_new)
            check(
                "max_positions",
                not is_new or count + 1 <= L.max_positions,
                f"{count + int(is_new)} of {L.max_positions} positions",
            )
            reserve = L.cash_buffer_pct * equity
            spendable = min(self.buying_power_left, self.cash_left - reserve)
            check(
                "buying_power",
                notional <= spendable + 0.01,
                f"${notional:,.0f} vs ${max(spendable, 0.0):,.0f} available above the "
                f"{L.cash_buffer_pct:.0%} cash reserve",
            )
            problems: list[str] = []
            if o.est_price < L.min_price:
                problems.append(f"price ${o.est_price:,.2f} < ${L.min_price:,.2f}")
            adv = q.adv_dollar if q else None
            if adv is None:
                problems.append("unknown trading volume")
            elif adv < L.min_dollar_volume:
                problems.append(f"${adv / 1e6:,.1f}M/day < ${L.min_dollar_volume / 1e6:,.0f}M")
            if q is not None and q.spread_bps is not None and q.spread_bps > L.max_spread_bps:
                problems.append(f"spread {q.spread_bps:.0f}bp > {L.max_spread_bps:.0f}bp")
            check("liquidity", not problems, "; ".join(problems) or "liquid")
        else:
            check("side", False, f"unknown side {o.side!r}")
        return RiskDecision(o, tuple(checks))

    def commit(self, o: OrderIntent) -> None:
        """Book an approved order so the rest of the cycle sees it."""
        self.working.add(o.symbol)
        if o.side == "buy":
            if self.held.get(o.symbol, 0.0) <= 0:
                self.pending_new.add(o.symbol)
            self.value[o.symbol] = self.value.get(o.symbol, 0.0) + o.notional
            self.exposure += o.notional
            self.cash_left -= o.notional
            self.buying_power_left -= o.notional
        else:
            self.held[o.symbol] = self.held.get(o.symbol, 0.0) - o.qty


def losing_positions(positions: Mapping[str, BrokerPosition], limit: float) -> dict[str, float]:
    """Holdings whose unrealised loss has reached ``limit`` (the stop-loss level)."""
    return {s: p.unrealized_plpc for s, p in positions.items() if p.unrealized_plpc <= -limit}
