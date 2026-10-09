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
                         ``max_spread_bps`` — measured on a validated quote (the consolidated SIP quote when
                         available); a spread that cannot be measured fails when ``require_live_data``;
    ``quote_quality``    the price agrees with the price history (no bad tick, split or mis-mapped symbol).

Options (:meth:`RiskBook.evaluate_option`) are judged by the same book — the same kill switch, account,
market-hours, daily-loss, working-order and cash checks — plus their own, every number recomputed here from
the legs and fresh quotes (never taken from the caller):
    ``structure``        one to four distinct contracts on one underlying, intents that match the sides;
    ``no_naked_short``   no short option uncovered by a long option or by shares already held (never);
    ``defined_risk``     the maximum loss is finite, computed from the payoff at expiration;
    ``structure_allowed`` the family is one a person has allowed (defined risk only);
    ``options_level``    the account's Alpaca options level permits it;
    ``expiration``       between the minimum and maximum days to expiration — never 0DTE;
    ``max_loss``         the order's maximum loss (at its limit price) within the per-trade limits;
    ``total_risk`` / ``underlying_risk``  the book's maximum loss after the order stays within limits;
    ``option_positions`` open structures within the limit; ``contracts`` per leg within the limit;
    ``option_quotes``    every leg has a fresh, two-sided execution-grade quote (never model or recorded);
    ``option_liquidity`` spreads and open interest within limits;
    ``greeks``           the book's net delta and vega stay within limits (fail closed when unknown);
    ``limit_price``      the net limit is no worse than the natural price (asks paid, bids received);
    ``closes_held``      a closing leg never exceeds what is held (no accidental opening).

The book is *projected*: each approved order is committed so later orders in the same cycle see the
exposure, positions and cash it will use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quantpulse.options.contracts import ContractError, OptionContract, is_option_symbol, parse_occ
from quantpulse.options.structures import FAMILIES, Leg, Structure, StructureError
from quantpulse.providers.alpaca_trading import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    mleg_symbol,
)
from quantpulse.schemas.common import DataStatus

TOLERANCE = 1e-6
EXECUTION_FEEDS = ("opra", "indicative")
# The Alpaca options level each kind of opening order needs (1 covered calls / cash-secured puts, 2 long
# options, 3 spreads and other multi-leg orders).
LEVEL_FOR = {"covered_call": 1, "cash_secured_put": 1, "long_call": 2, "long_put": 2}


@dataclass(frozen=True, slots=True)
class OptionLimits:
    enabled: bool = True
    allowed_structures: tuple[str, ...] = (
        "long_call",
        "long_put",
        "bull_call_spread",
        "bear_put_spread",
        "bull_put_spread",
        "bear_call_spread",
        "covered_call",
    )
    max_loss_per_trade: float = 2000.0
    max_loss_pct_per_trade: float = 0.02
    max_total_risk_pct: float = 0.25
    max_underlying_risk_pct: float = 0.05
    max_positions: int = 12
    max_contracts: int = 20
    min_dte: int = 7
    max_dte: int = 60
    max_spread_pct: float = 0.15
    max_quote_age_seconds: float = 120.0
    min_open_interest: float = 100.0
    max_delta_pct: float = 1.50
    max_vega_pct: float = 0.01
    exploration_max_loss: float = 2000.0

    @classmethod
    def from_settings(cls, s: Any) -> OptionLimits:
        return cls(
            enabled=s.options_enabled and s.options_execution,
            allowed_structures=tuple(s.options_allowed_structures),
            max_loss_per_trade=s.options_max_loss_per_trade,
            max_loss_pct_per_trade=s.options_max_loss_pct_per_trade,
            max_total_risk_pct=s.options_max_total_risk_pct,
            max_underlying_risk_pct=s.options_max_underlying_risk_pct,
            max_positions=s.options_max_positions,
            max_contracts=s.options_max_contracts,
            min_dte=s.options_min_dte,
            max_dte=s.options_max_dte,
            max_spread_pct=s.options_max_spread_pct,
            max_quote_age_seconds=s.options_max_quote_age_seconds,
            min_open_interest=s.options_min_open_interest,
            max_delta_pct=s.options_max_delta_pct,
            max_vega_pct=s.options_max_vega_pct,
            exploration_max_loss=s.options_exploration_max_loss,
        )


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
    options: OptionLimits = field(default_factory=OptionLimits)

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
            options=OptionLimits.from_settings(s),
        )


@dataclass(frozen=True, slots=True)
class QuoteCheck:
    """What the risk engine knows about a symbol's market data."""

    price: float
    status: DataStatus
    provider: str
    age_seconds: float | None = None
    spread_bps: float | None = None  # validated (see trading_data.assess_quote); None: not measurable
    adv_dollar: float | None = None
    spread_source: str | None = None  # e.g. "SIP", "IEX only"
    quote_problems: tuple[str, ...] = ()  # why parts of the quote were not believed
    entry_blocks: tuple[str, ...] = ()  # price inconsistencies that forbid new buying
    source: str | None = None  # what the price and its age come from, e.g. "last IEX trade (alpaca)"


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
class OptionLegQuote:
    """Fresh market data for one contract, read by the trading service just before the order. The Greeks
    are per share, computed by the trading service from the quote (BSM on the mid's implied volatility)."""

    bid: float | None
    ask: float | None
    age_seconds: float | None
    feed: str
    open_interest: float | None = None
    iv: float | None = None
    delta: float | None = None
    vega: float | None = None

    @property
    def two_sided(self) -> bool:
        return self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > self.bid

    @property
    def mid(self) -> float | None:
        return 0.5 * (self.bid + self.ask) if self.two_sided else None  # type: ignore[operator]

    @property
    def spread_pct(self) -> float | None:
        mid = self.mid
        return (self.ask - self.bid) / mid if mid else None  # type: ignore[operator]


@dataclass(frozen=True, slots=True)
class OptionLegIntent:
    symbol: str
    side: str  # buy | sell
    ratio: int
    position_intent: str  # buy_to_open | buy_to_close | sell_to_open | sell_to_close


@dataclass(frozen=True, slots=True)
class OptionOrderIntent:
    """One option order: a single contract or a multi-leg structure, ``qty`` units of it, at a net limit
    price per share (positive: a debit paid; negative: a credit received)."""

    underlying: str
    family: str
    legs: tuple[OptionLegIntent, ...]
    qty: int
    limit_price: float
    kind: str  # entry | take_profit | stop_loss | expiration_close | thesis_exit | flatten | …
    reason: str
    opening: bool
    underlying_price: float
    exploration: bool = False
    score: float | None = None
    intent: str = "strategy"  # strategy | flatten
    strategy_key: str | None = None

    @property
    def net_debit(self) -> float:
        """The limit as a signed net price per share: positive a debit paid, negative a credit received. A
        multi-leg limit is already signed (as Alpaca takes it); a single-leg limit is a positive price, a
        credit when the leg is sold."""
        if len(self.legs) == 1 and self.legs[0].side == "sell":
            return -abs(self.limit_price)
        return self.limit_price if len(self.legs) > 1 else abs(self.limit_price)

    @property
    def symbol(self) -> str:
        """The record symbol: the contract of a single-leg order, ``AAPL:MLEG`` for a multi-leg one."""
        return self.legs[0].symbol if len(self.legs) == 1 else mleg_symbol([x.symbol for x in self.legs])

    @property
    def side(self) -> str:
        if len(self.legs) == 1:
            return self.legs[0].side
        return "buy" if self.net_debit >= 0 else "sell"

    @property
    def closes_position(self) -> bool:
        return not self.opening

    @property
    def notional(self) -> float:
        return abs(self.limit_price) * self.qty * 100

    @property
    def est_price(self) -> float:
        return abs(self.limit_price)


@dataclass(frozen=True, slots=True)
class RiskCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RiskDecision:
    intent: OrderIntent | OptionOrderIntent
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
    option_quotes: Mapping[str, OptionLegQuote] = field(default_factory=dict)
    now: datetime | None = None  # the moment of the decision (days to expiration are counted from it)
    held: dict[str, float] = field(init=False)
    value: dict[str, float] = field(init=False)
    working: set[str] = field(init=False)
    pending_new: set[str] = field(init=False)
    exposure: float = field(init=False)
    cash_left: float = field(init=False)
    buying_power_left: float = field(init=False)
    option_legs: dict[str, float] = field(init=False)  # contract -> signed contracts held (+ long, − short)
    option_pending: list[OptionOrderIntent] = field(init=False)  # opening option orders committed this cycle
    option_pending_capital: float = field(init=False)

    def __post_init__(self) -> None:
        stock = {s: p for s, p in self.positions.items() if not p.is_option}
        self.held = {s: p.qty for s, p in stock.items() if p.qty > 0}
        self.value = {s: p.market_value for s, p in stock.items()}
        self.option_legs = {s: p.qty for s, p in self.positions.items() if p.is_option}
        self.working = {sym for o in self.open_orders if o.is_open for sym in (o.symbol, *o.symbols)}
        self.pending_new = set()
        self.option_pending = []
        self.option_pending_capital = 0.0
        pending_buys = 0.0
        for o in self.open_orders:
            if not o.is_open:
                continue
            if o.is_option:
                self.option_pending_capital += _working_option_capital(o)
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
        self.cash_left = self.account.cash - pending_buys - self.option_pending_capital
        self.buying_power_left = self.account.buying_power - pending_buys - self.option_pending_capital
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
                f"quote is {q.age_seconds:.0f}s old (limit {L.max_quote_age_seconds:.0f}s"
                + (f"; {q.source})" if q.source else ")"),
            )
        else:
            age = f", {q.age_seconds:.0f}s old" if q.age_seconds is not None else ""
            check("live_data", True, f"{q.status.value} quote from {q.provider}{age}")

        if is_option_symbol(o.symbol):
            check("asset_class", False, "an option contract: it goes through the options checks")
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
            source = f" ({q.spread_source})" if q is not None and q.spread_source else ""
            if q is not None and q.spread_bps is not None and q.spread_bps > L.max_spread_bps:
                problems.append(f"spread {q.spread_bps:.0f}bp{source} > {L.max_spread_bps:.0f}bp")
            elif q is not None and q.spread_bps is None and L.require_live_data:
                why = "; ".join(p.split(": ", 1)[-1] for p in q.quote_problems) or "no bid/ask"
                problems.append(f"spread cannot be measured ({why})")
            spread = (
                f", spread {q.spread_bps:.0f}bp{source}" if q is not None and q.spread_bps is not None else ""
            )
            check("liquidity", not problems, "; ".join(problems) or f"liquid{spread}")
            blocks = list(q.entry_blocks) if q is not None else []
            check("quote_quality", not blocks, "; ".join(blocks) or "price consistent with its history")
        else:
            check("side", False, f"unknown side {o.side!r}")
        return RiskDecision(o, tuple(checks))

    def commit(self, o: OrderIntent | OptionOrderIntent) -> None:
        """Book an approved order so the rest of the cycle sees it."""
        if isinstance(o, OptionOrderIntent):
            self._commit_option(o)
            return
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

    # ------------------------------------------------------------------ options
    def _now(self) -> datetime:
        from quantpulse.core.clock import utcnow

        return self.now or utcnow()

    def _held_shares(self, underlying: str) -> float:
        return self.held.get(underlying, 0.0)

    def _option_price(self, symbol: str) -> float:
        q = self.option_quotes.get(symbol)
        if q is not None and q.mid is not None:
            return q.mid
        p = self.positions.get(symbol)
        if p is not None and p.current_price > 0:
            return p.current_price
        return max(p.avg_entry_price, 0.0) if p is not None else 0.0

    def _book_legs(self, underlying: str, extra: Sequence[tuple[str, float, float]] = ()) -> list[Leg]:
        """The option legs held on ``underlying`` (and those of orders committed this cycle), plus ``extra``
        (symbol, signed contracts, price per share), netted by contract."""
        net: dict[str, float] = {}
        price: dict[str, float] = {}
        for sym, n in self.option_legs.items():
            if _root(sym) == underlying:
                net[sym] = net.get(sym, 0.0) + n
                price[sym] = self._option_price(sym)
        for o in self.option_pending:
            if o.underlying != underlying:
                continue
            for leg in o.legs:
                sign = 1 if leg.side == "buy" else -1
                net[leg.symbol] = net.get(leg.symbol, 0.0) + sign * leg.ratio * o.qty
                price.setdefault(leg.symbol, self._option_price(leg.symbol))
        for sym, n, px in extra:
            net[sym] = net.get(sym, 0.0) + n
            price[sym] = px
        legs = []
        for sym, n in sorted(net.items()):
            if abs(n) < TOLERANCE:
                continue
            legs.append(
                Leg("long" if n > 0 else "short", round(abs(n)), max(price[sym], 0.0), parse_occ(sym))
            )
        return legs

    def _marginal_loss(self, underlying: str, legs: list[Leg], shares: float, spot: float) -> float:
        """The largest further loss the option legs add on ``underlying``: the whole book's maximum loss at
        expiration (shares included, so covered calls count as covered) less that of the shares alone."""
        if not legs:
            return 0.0
        whole = list(legs)
        n_shares = math.floor(max(shares, 0.0))
        if n_shares:
            whole.append(Leg("long", n_shares, spot))
        try:
            loss = Structure("book", underlying, tuple(whole)).max_loss()
        except StructureError:
            return math.inf
        return max(loss - n_shares * spot, 0.0) if math.isfinite(loss) else math.inf

    def option_risk(
        self, extra: Mapping[str, Sequence[tuple[str, float, float]]] | None = None
    ) -> dict[str, float]:
        """Maximum further loss of the option book per underlying (``inf`` when a short leg is uncovered)."""
        extra = extra or {}
        unders = (
            {_root(s) for s in self.option_legs} | {o.underlying for o in self.option_pending} | set(extra)
        )
        out: dict[str, float] = {}
        for u in sorted(unders):
            spot = self._spot(u)
            out[u] = self._marginal_loss(u, self._book_legs(u, extra.get(u, ())), self._held_shares(u), spot)
        return out

    def _spot(self, underlying: str) -> float:
        for o in self.option_pending:
            if o.underlying == underlying and o.underlying_price > 0:
                return o.underlying_price
        q = self.quotes.get(underlying)
        if q is not None and q.price > 0:
            return q.price
        p = self.positions.get(underlying)
        if p is not None and p.current_price > 0:
            return p.current_price
        strikes = [parse_occ(s).strike for s in self.option_legs if _root(s) == underlying]
        return float(sum(strikes) / len(strikes)) if strikes else 1.0

    def option_greeks(self) -> dict[str, float | None]:
        """Net option delta (dollars of underlying) and vega (dollars per volatility point) of the book and of
        the orders committed this cycle; ``None`` when any leg's Greek is unknown."""
        delta: float | None = 0.0
        vega: float | None = 0.0
        legs: list[tuple[str, float]] = list(self.option_legs.items())
        for o in self.option_pending:
            legs += [(x.symbol, (1 if x.side == "buy" else -1) * x.ratio * o.qty) for x in o.legs]
        for sym, n in legs:
            q = self.option_quotes.get(sym)
            spot = self._spot(_root(sym))
            if q is None or q.delta is None or delta is None:
                delta = None
            else:
                delta += n * 100 * q.delta * spot
            if q is None or q.vega is None or vega is None:
                vega = None
            else:
                vega += n * 100 * q.vega
        return {"delta_dollars": delta, "vega_dollars": vega}

    def option_structures(self) -> set[tuple[str, str]]:
        """Open option positions, one per (underlying, first expiration)."""
        out = {(_root(s), s[len(_root(s)) : len(_root(s)) + 6]) for s in self.option_legs}
        for o in self.option_pending:
            out.add((o.underlying, min(x.symbol[len(o.underlying) : len(o.underlying) + 6] for x in o.legs)))
        return out

    def evaluate_option(self, o: OptionOrderIntent, *, spot: float | None = None) -> RiskDecision:
        """The same book's verdict on an option order (see the module docstring for every check)."""
        L, OL = self.limits, self.limits.options
        checks: list[RiskCheck] = []

        def check(name: str, ok: bool, detail: str) -> None:
            checks.append(RiskCheck(name, bool(ok), detail))

        flatten = o.intent == "flatten"
        if self.kill_switch:
            check("kill_switch", flatten, "kill switch is on; only explicit flatten orders may pass"
                  if flatten else "kill switch is ON")  # fmt: skip
        else:
            check("kill_switch", True, "off")
        check("account", not self.account.blocked, "blocked by Alpaca" if self.account.blocked else "active")
        check("market_open", self.market_open, "market open" if self.market_open else "market is closed")
        if o.opening:
            check("options_enabled", OL.enabled, "option orders enabled" if OL.enabled
                  else "option execution is switched off (QP_OPTIONS_ENABLED / QP_OPTIONS_EXECUTION)")  # fmt: skip

        # ---- structure: what the order is, recomputed from its legs
        problems: list[str] = []
        contracts: list[OptionContract] = []
        if not 1 <= len(o.legs) <= 4:
            problems.append(f"{len(o.legs)} legs (1 to 4 allowed)")
        if len({x.symbol for x in o.legs}) != len(o.legs):
            problems.append("a contract appears twice")
        for x in o.legs:
            try:
                c = parse_occ(x.symbol)
            except ContractError:
                problems.append(f"{x.symbol} is not an option contract")
                continue
            contracts.append(c)
            if c.underlying != o.underlying:
                problems.append(f"{x.symbol} is not on {o.underlying}")
            if x.side not in ("buy", "sell") or not x.position_intent.startswith(x.side):
                problems.append(f"{x.symbol}: {x.side} cannot {x.position_intent.replace('_', ' ')}")
            if x.position_intent.endswith("_to_open") != o.opening:
                problems.append(
                    f"{x.symbol}: {x.position_intent} in an {'opening' if o.opening else 'closing'} order"
                )
            if x.ratio < 1:
                problems.append(f"{x.symbol}: ratio {x.ratio}")
        if o.qty < 1 or int(o.qty) != o.qty:
            problems.append(f"quantity {o.qty} (whole units, at least 1)")
        check("structure", not problems, "; ".join(problems) or f"{o.family}: {len(o.legs)} leg(s) × {o.qty}")
        if problems:
            return RiskDecision(o, tuple(checks))

        busy = [x.symbol for x in o.legs if x.symbol in self.working] + (
            [o.symbol] if o.symbol in self.working else []
        )
        check("no_working_order", not busy, "another order is still working for " + ", ".join(busy)
              if busy else "none")  # fmt: skip

        # ---- quotes: every leg fresh, two-sided, execution grade
        qproblems: list[str] = []
        for x in o.legs:
            q = self.option_quotes.get(x.symbol)
            if q is None:
                qproblems.append(f"{x.symbol}: no quote")
                continue
            if q.feed not in EXECUTION_FEEDS:
                qproblems.append(f"{x.symbol}: {q.feed} data is never an execution quote")
            if q.age_seconds is None or q.age_seconds > OL.max_quote_age_seconds:
                age = "unknown age" if q.age_seconds is None else f"{q.age_seconds:.0f}s old"
                qproblems.append(f"{x.symbol}: {age} (limit {OL.max_quote_age_seconds:.0f}s)")
            needs = "ask" if x.side == "buy" else "bid"
            px = q.ask if x.side == "buy" else q.bid
            if o.opening and not q.two_sided:
                qproblems.append(f"{x.symbol}: not a two-sided market")
            elif px is None or px <= 0:
                qproblems.append(f"{x.symbol}: no {needs} to trade against")
        check("option_quotes", not qproblems, "; ".join(qproblems) or "fresh two-sided quotes for every leg")

        # ---- the limit price: never worse than the natural price
        natural = 0.0
        known = True
        for x in o.legs:
            q = self.option_quotes.get(x.symbol)
            px = (q.ask if x.side == "buy" else q.bid) if q is not None else None
            if px is None or px <= 0:
                known = False
                break
            natural += (1 if x.side == "buy" else -1) * x.ratio * px
        tol = 0.01 * len(o.legs)
        if not known:
            check("limit_price", False, "the natural price cannot be computed (a leg has no quote)")
        else:
            ok = o.net_debit <= natural + tol
            check("limit_price", ok, f"net {o.net_debit:+.2f} vs natural {natural:+.2f} per share"
                  + ("" if ok else " (worse than crossing the spread)"))  # fmt: skip

        if not o.opening:
            over = []
            for x in o.legs:
                held = self.option_legs.get(x.symbol, 0.0)
                n = x.ratio * o.qty
                if x.position_intent == "sell_to_close" and n > held + TOLERANCE:
                    over.append(f"{x.symbol}: sells {n:g}, {max(held, 0):g} held long")
                if x.position_intent == "buy_to_close" and n > -held + TOLERANCE:
                    over.append(f"{x.symbol}: buys back {n:g}, {max(-held, 0):g} held short")
            check("closes_held", not over, "; ".join(over) or "closes what is held")
            return RiskDecision(o, tuple(checks))

        # ---- opening checks
        family = FAMILIES.get(o.family)
        allowed = o.family in OL.allowed_structures and family is not None and family.defined_risk
        check("structure_allowed", allowed, f"{o.family} is allowed" if allowed
              else f"{o.family} is not an allowed structure ({', '.join(OL.allowed_structures)})")  # fmt: skip

        u = o.underlying
        spot = spot if spot and spot > 0 else o.underlying_price
        mids = {x.symbol: (self.option_quotes[x.symbol].mid or 0.0) if x.symbol in self.option_quotes else 0.0
                for x in o.legs}  # fmt: skip
        legs = [Leg("long" if x.side == "buy" else "short", x.ratio, mids[x.symbol], c)
                for x, c in zip(o.legs, contracts, strict=True)]  # fmt: skip
        # shares held and not already covering a short call cover new short calls (a covered call)
        short_calls = sum(
            -n for s_, n in self.option_legs.items() if n < 0 and _root(s_) == u and s_[-9] == "C"
        )
        free_shares = max(self._held_shares(u) - 100 * short_calls, 0.0)
        with_shares = list(legs)
        if o.family == "covered_call" and free_shares >= 100:
            with_shares.append(Leg("long", int(free_shares // 100) * 100, spot))
        try:
            st = Structure(o.family, u, tuple(with_shares))
            naked = st.naked_legs()
        except StructureError as exc:
            check("no_naked_short", False, f"cannot be built: {exc}")
            return RiskDecision(o, tuple(checks))
        check("no_naked_short", not naked, "every short leg is covered" if not naked
              else "uncovered short: " + ", ".join(x.label() for x in naked))  # fmt: skip

        # the order's own maximum loss at its limit price (shares' own risk excluded)
        opt_only = Structure(o.family, u, tuple(legs))
        shift = o.net_debit * 100 - opt_only.debit()  # paid beyond the mids, per unit
        shares_n = sum(x.units for x in with_shares if x.contract is None)
        if shares_n:
            loss_unit = (
                max(st.max_loss() - shares_n * spot + shift, 0.0)
                if math.isfinite(st.max_loss())
                else math.inf
            )
        else:
            loss_unit = opt_only.max_loss() + shift if math.isfinite(opt_only.max_loss()) else math.inf
        order_loss = loss_unit * o.qty
        check("defined_risk", math.isfinite(order_loss), f"maximum loss ${order_loss:,.0f}"
              if math.isfinite(order_loss) else "unlimited loss: never executed")  # fmt: skip

        need = LEVEL_FOR.get(o.family, 3 if len(o.legs) > 1 else 2)
        level = self.account.options_trading_level
        check("options_level", level is not None and level >= need,
              f"account level {level}, needs {need}" if level is not None
              else "Alpaca did not report the account's options level")  # fmt: skip

        dtes = [c.dte(self._now()) for c in contracts]
        lo, hi = min(dtes), max(dtes)
        dte_ok = lo >= max(OL.min_dte, 1) and hi <= OL.max_dte
        check("expiration", dte_ok, f"{lo}–{hi} days to expiration (allowed {max(OL.min_dte, 1)}–{OL.max_dte};"
              " 0DTE is research only)")  # fmt: skip

        too_many = [x.symbol for x in o.legs if x.ratio * o.qty > OL.max_contracts]
        check("contracts", not too_many, f"{o.qty} unit(s), at most {OL.max_contracts} contracts per leg"
              if not too_many else f"more than {OL.max_contracts} contracts of {', '.join(too_many)}")  # fmt: skip

        equity = self.equity
        cap = min(OL.max_loss_per_trade, OL.max_loss_pct_per_trade * equity)
        label = "limit"
        if o.exploration:
            cap, label = min(cap, OL.exploration_max_loss), "exploration limit"
            check(
                "exploration_size",
                o.qty == 1,
                "one unit" if o.qty == 1 else "exploration trades one unit only",
            )
        check(
            "max_loss", order_loss <= cap + 0.01, f"${order_loss:,.0f} at most (the {label} is ${cap:,.0f})"
        )

        extra = {
            u: [(x.symbol, (1 if x.side == "buy" else -1) * x.ratio * o.qty, mids[x.symbol]) for x in o.legs]
        }
        before = self.option_risk()
        after = self.option_risk(extra)
        # what the order adds is what it pays beyond the mids too
        u_after = after.get(u, 0.0) + max(shift, 0.0) * o.qty
        total_after = sum(v for k, v in after.items() if k != u) + u_after
        check("underlying_risk", u_after <= OL.max_underlying_risk_pct * equity + 0.01,
              f"${u_after:,.0f} maximum loss on {u} after the order (limit {OL.max_underlying_risk_pct:.1%} of equity)"
              if math.isfinite(u_after) else f"{u}: unlimited loss")  # fmt: skip
        check("total_risk", total_after <= OL.max_total_risk_pct * equity + 0.01,
              f"${total_after:,.0f} maximum loss across options (limit {OL.max_total_risk_pct:.0%} of equity; "
              f"${sum(before.values()):,.0f} before)" if math.isfinite(total_after)
              else "the option book has unlimited loss")  # fmt: skip

        count = self.option_structures()
        new_key = (u, min(x.symbol[len(u) : len(u) + 6] for x in o.legs))
        n_after = len(count | {new_key})
        check(
            "option_positions",
            n_after <= OL.max_positions,
            f"{n_after} of {OL.max_positions} option positions",
        )

        check("daily_loss", not self.daily_loss_hit,
              f"day P/L {self.account.day_pl_pct:+.2%} reached the −{L.max_daily_loss_pct:.0%} limit: "
              "no new positions today" if self.daily_loss_hit
              else f"day P/L {self.account.day_pl_pct:+.2%} (limit −{L.max_daily_loss_pct:.0%})")  # fmt: skip

        capital = max(order_loss, o.net_debit * 100 * o.qty, 0.0) if math.isfinite(order_loss) else math.inf
        reserve = L.cash_buffer_pct * equity
        spendable = min(self.buying_power_left, self.cash_left - reserve)
        if self.account.options_buying_power is not None:
            spendable = min(spendable, self.account.options_buying_power - self.option_pending_capital)
        check("buying_power", capital <= spendable + 0.01,
              f"${capital:,.0f} needed vs ${max(spendable, 0.0):,.0f} available above the "
              f"{L.cash_buffer_pct:.0%} cash reserve")  # fmt: skip

        lproblems = []
        for x in o.legs:
            q = self.option_quotes.get(x.symbol)
            if q is None:
                continue
            sp = q.spread_pct
            if sp is None or sp > OL.max_spread_pct:
                lproblems.append(f"{x.symbol}: spread {'unknown' if sp is None else f'{sp:.0%}'} "
                                 f"(limit {OL.max_spread_pct:.0%})")  # fmt: skip
            if OL.min_open_interest > 0 and (
                q.open_interest is None or q.open_interest < OL.min_open_interest
            ):
                oi = "unknown" if q.open_interest is None else f"{q.open_interest:,.0f}"
                lproblems.append(f"{x.symbol}: open interest {oi} (minimum {OL.min_open_interest:,.0f})")
        check(
            "option_liquidity",
            not lproblems,
            "; ".join(lproblems) or "spreads and open interest within limits",
        )

        g_before = self.option_greeks()
        add_delta: float | None = 0.0
        add_vega: float | None = 0.0
        for x in o.legs:
            q = self.option_quotes.get(x.symbol)
            n = (1 if x.side == "buy" else -1) * x.ratio * o.qty
            add_delta = (
                None
                if q is None or q.delta is None or add_delta is None
                else add_delta + n * 100 * q.delta * spot
            )
            add_vega = (
                None if q is None or q.vega is None or add_vega is None else add_vega + n * 100 * q.vega
            )
        d0, v0 = g_before["delta_dollars"], g_before["vega_dollars"]
        if None in (d0, v0, add_delta, add_vega) or equity <= 0:
            check(
                "greeks",
                False,
                "the book's delta or vega cannot be measured (a leg has no Greeks): fail closed",
            )
        else:
            d1, v1 = d0 + add_delta, v0 + add_vega  # type: ignore[operator]
            ok = abs(d1) <= OL.max_delta_pct * equity + 0.01 and abs(v1) <= OL.max_vega_pct * equity + 0.01
            check("greeks", ok, f"net delta ${d1:,.0f} (limit ±{OL.max_delta_pct:.0%} of equity), net vega "
                  f"${v1:,.0f}/vol point (limit ±{OL.max_vega_pct:.1%})")  # fmt: skip
        return RiskDecision(o, tuple(checks))

    def _commit_option(self, o: OptionOrderIntent) -> None:
        self.working.add(o.symbol)
        self.working.update(x.symbol for x in o.legs)
        if o.opening:
            self.option_pending.append(o)
            mids = [self._option_price(x.symbol) for x in o.legs]
            legs = [Leg("long" if x.side == "buy" else "short", x.ratio, m, parse_occ(x.symbol))
                    for x, m in zip(o.legs, mids, strict=True)]  # fmt: skip
            try:
                st = Structure(o.family, o.underlying, tuple(legs))
                loss = st.max_loss() + o.net_debit * 100 - st.debit()
            except StructureError:
                loss = math.inf
            capital = max(loss if math.isfinite(loss) else 0.0, o.net_debit * 100, 0.0) * o.qty
            self.cash_left -= capital
            self.buying_power_left -= capital
            self.option_pending_capital += capital
        else:
            for x in o.legs:
                sign = 1 if x.side == "buy" else -1
                self.option_legs[x.symbol] = self.option_legs.get(x.symbol, 0.0) + sign * x.ratio * o.qty


def _root(symbol: str) -> str:
    try:
        return parse_occ(symbol).underlying
    except ContractError:
        return symbol.split(":")[0]


def _working_option_capital(o: BrokerOrder) -> float:
    """Cash an opening option order still working will tie up: its maximum loss at its limit price."""
    left = o.remaining_qty
    if left <= 0 or o.limit_price is None:
        return 0.0
    if o.legs:
        if any((x.position_intent or "").endswith("_to_close") for x in o.legs):
            return 0.0
        try:
            legs = tuple(Leg("long" if x.side == "buy" else "short", int(x.ratio_qty or 1), 0.0, parse_occ(x.symbol))
                         for x in o.legs)  # fmt: skip
            loss = Structure("working", legs[0].contract.underlying, legs).max_loss()  # type: ignore[union-attr]
        except (StructureError, ContractError):
            return math.inf
        return (loss + o.limit_price * 100) * left if math.isfinite(loss) else math.inf
    if (o.position_intent or "").endswith("_to_close"):
        return 0.0
    if o.side == "buy":
        return o.limit_price * 100 * left
    try:
        c = parse_occ(o.symbol)
    except ContractError:
        return 0.0
    return (c.strike * 100 - o.limit_price * 100) * left if c.kind == "put" else 0.0


def losing_positions(positions: Mapping[str, BrokerPosition], limit: float) -> dict[str, float]:
    """Holdings whose unrealised loss has reached ``limit`` (the stop-loss level)."""
    return {s: p.unrealized_plpc for s, p in positions.items() if p.unrealized_plpc <= -limit}
