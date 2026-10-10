"""Option structures: one position made of legs, with its payoff, risk and capital — all computed.

A spread is ONE position: a bull call spread is a long call and a short call held together, entered and
exited together, its P&L the sum of its legs'. Every structure answers, exactly:

* :meth:`Structure.pnl_at_expiry` — dollars per unit at a price (piecewise linear for one expiration);
* :meth:`Structure.max_profit`, :meth:`Structure.max_loss` — ``math.inf`` when unbounded;
* :meth:`Structure.breakevens` — every price where the P&L at expiration crosses zero;
* :meth:`Structure.capital_required` — the cash it ties up (the debit, or the maximum loss of a credit
  structure — what the broker holds as buying power);
* :meth:`Structure.greeks` — the net delta, gamma, theta, vega and rho in dollars, per unit;
* :meth:`Structure.naked_legs` — short options not covered by a long option or stock: never executed.

Dollars everywhere are for one unit of the structure; ``quantity`` units scale them linearly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Literal

import numpy as np

from quantpulse.options.contracts import OptionContract
from quantpulse.options.pricing import greeks as model_greeks

Side = Literal["long", "short"]
Direction = Literal["bullish", "bearish", "neutral", "volatility", "income"]
VolStance = Literal["long_vol", "short_vol", "neutral"]


class StructureError(ValueError):
    """A structure that is malformed (mixed underlyings, strikes in the wrong order, missing prices)."""


@dataclass(frozen=True, slots=True)
class Leg:
    """One leg. ``contract`` is ``None`` for shares of the underlying. ``ratio`` counts contracts (or shares
    for a stock leg) per unit of the structure; ``price`` is the per-share entry price."""

    side: Side
    ratio: int
    price: float
    contract: OptionContract | None = None

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise StructureError(f"side must be long or short, not {self.side!r}")
        if self.ratio <= 0:
            raise StructureError("ratio must be positive")
        if not (math.isfinite(self.price) and self.price >= 0):
            raise StructureError(f"leg price must be a non-negative number, not {self.price!r}")

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def is_stock(self) -> bool:
        return self.contract is None

    @property
    def units(self) -> int:
        """Shares represented: contracts × multiplier, or shares."""
        return self.ratio * (self.contract.multiplier if self.contract is not None else 1)

    def value_at_expiry(self, spot: float | np.ndarray) -> float | np.ndarray:
        if self.contract is None:
            return spot
        k = self.contract.strike
        return np.maximum(spot - k, 0.0) if self.contract.is_call else np.maximum(k - spot, 0.0)

    def label(self) -> str:
        if self.contract is None:
            return f"{self.side} {self.ratio} shares @ {self.price:.2f}"
        c = self.contract
        return f"{self.side} {self.ratio}× {c.expiration:%Y-%m-%d} {c.strike:g}{c.kind[0].upper()} @ {self.price:.2f}"


@dataclass(frozen=True, slots=True)
class Family:
    name: str
    direction: Direction
    vol: VolStance
    defined_risk: bool
    description: str
    # executable by default in paper mode (anything else waits for a person to enable it after validation)
    default_executable: bool


FAMILIES: dict[str, Family] = {f.name: f for f in (
    Family("long_call", "bullish", "long_vol", True, "buy a call: bullish, loss limited to the premium", True),
    Family("long_put", "bearish", "long_vol", True, "buy a put: bearish, loss limited to the premium", True),
    Family("bull_call_spread", "bullish", "neutral", True, "call debit spread: buy a call, sell a higher one", True),
    Family("bear_put_spread", "bearish", "neutral", True, "put debit spread: buy a put, sell a lower one", True),
    Family("bull_put_spread", "bullish", "short_vol", True, "put credit spread: sell a put, buy a lower one", True),
    Family("bear_call_spread", "bearish", "short_vol", True, "call credit spread: sell a call, buy a higher one", True),
    Family("covered_call", "income", "short_vol", True, "sell a call against 100 shares already held", True),
    Family("cash_secured_put", "income", "short_vol", True, "sell a put with the strike's cash set aside", False),
    Family("long_straddle", "volatility", "long_vol", True, "buy a call and a put at one strike", False),
    Family("long_strangle", "volatility", "long_vol", True, "buy an out-of-the-money call and put", False),
    Family("iron_condor", "neutral", "short_vol", True, "a put credit spread plus a call credit spread", False),
    Family("call_butterfly", "neutral", "short_vol", True, "long 1 low, short 2 middle, long 1 high call", False),
    Family("protective_put", "bullish", "long_vol", True, "a put bought against 100 shares held", False),
    Family("collar", "neutral", "neutral", True, "shares + a long put + a short call", False),
    Family("calendar", "neutral", "long_vol", True, "sell a near expiration, buy a later one (same strike)", False),
    Family("put_butterfly", "neutral", "short_vol", True, "long 1 high, short 2 middle, long 1 low put", False),
    Family("iron_butterfly", "neutral", "short_vol", True, "sell an at-the-money call and put, buy a wing on each side", False),
    Family("broken_wing_butterfly", "neutral", "short_vol", True, "a put butterfly with a wider lower wing: little or no debit, risk only below", False),
    Family("reverse_iron_condor", "volatility", "long_vol", True, "a put debit spread below and a call debit spread above: paid on a move either way", False),
    Family("naked_call", "bearish", "short_vol", False, "an uncovered short call: unlimited loss — never executed", False),
    Family("naked_put", "bullish", "short_vol", False, "an uncovered short put without cash: never executed", False),
    Family("stock", "bullish", "neutral", True, "shares only (the comparison every option trade must beat)", True),
)}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Structure:
    family: str
    underlying: str
    legs: tuple[Leg, ...]
    quantity: int = 1
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.legs:
            raise StructureError("a structure needs at least one leg")
        if self.quantity <= 0:
            raise StructureError("quantity must be positive")
        for leg in self.legs:
            if leg.contract is not None and leg.contract.underlying != self.underlying:
                raise StructureError(f"leg on {leg.contract.underlying} in a {self.underlying} structure")

    # ------------------------------------------------------------------ description
    @property
    def option_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in self.legs if leg.contract is not None)

    @property
    def expirations(self) -> list[date]:
        return sorted({leg.contract.expiration for leg in self.option_legs})  # type: ignore[union-attr]

    @property
    def first_expiration(self) -> date | None:
        exps = self.expirations
        return exps[0] if exps else None

    @property
    def single_expiry(self) -> bool:
        return len(self.expirations) <= 1

    @property
    def strikes(self) -> list[float]:
        return sorted({leg.contract.strike for leg in self.option_legs})  # type: ignore[union-attr]

    def describe(self) -> str:
        return f"{self.family} {self.underlying}: " + "; ".join(leg.label() for leg in self.legs)

    def key(self) -> str:
        """A stable identity for the same legs (duplicate candidates share it)."""
        parts = sorted(
            f"{leg.side}:{leg.ratio}:{leg.contract.symbol if leg.contract else 'STOCK'}" for leg in self.legs
        )
        return f"{self.family}|{self.underlying}|" + ",".join(parts)

    def scaled(self, quantity: int) -> Structure:
        return replace(self, quantity=quantity)

    # ------------------------------------------------------------------ money
    def debit(self) -> float:
        """Dollars paid to open one unit (negative: a credit received)."""
        return sum(leg.sign * leg.units * leg.price for leg in self.legs)

    def pnl_at_expiry(self, spot: float | np.ndarray) -> float | np.ndarray:
        """P&L per unit at the (first) expiration, holding every leg to it. Exact for one expiration."""
        if not self.single_expiry:
            raise StructureError("several expirations: use pnl_at with a model for the later legs")
        total: float | np.ndarray = 0.0
        for leg in self.legs:
            total = total + leg.sign * leg.units * (leg.value_at_expiry(spot) - leg.price)
        return total

    def pnl_at(
        self,
        spot: float,
        when: datetime,
        vol: float | Callable[[OptionContract], float],
        rate: float = 0.04,
        div: float = 0.0,
    ) -> float:
        """P&L per unit at ``when`` (any date before the last expiration) valuing unexpired legs with BSM."""
        total = 0.0
        for leg in self.legs:
            if leg.contract is None:
                value = spot
            elif leg.contract.expired(when):
                value = float(leg.value_at_expiry(spot))
            else:
                sigma = vol(leg.contract) if callable(vol) else vol
                value = model_greeks(leg.contract.kind, spot, leg.contract.strike, leg.contract.years(when), sigma,
                                     rate, div).price  # fmt: skip
            total += leg.sign * leg.units * (value - leg.price)
        return total

    def _slope_above(self) -> float:
        """d(P&L)/dS once the price is above every strike: calls and shares keep moving with it."""
        return float(
            sum(leg.sign * leg.units for leg in self.legs if leg.contract is None or leg.contract.is_call)
        )

    def _kinks(self) -> list[float]:
        return [0.0, *self.strikes]

    def max_profit(self) -> float:
        if not self.single_expiry:
            return self._grid_extreme(max)
        if self._slope_above() > 1e-9:
            return math.inf
        return float(np.max(self.pnl_at_expiry(np.array(self._kinks()))))

    def max_loss(self) -> float:
        """The largest loss per unit, as a positive number (``inf`` when unbounded)."""
        if not self.single_expiry:
            return -self._grid_extreme(min)
        if self._slope_above() < -1e-9:
            return math.inf
        return float(max(0.0, -np.min(self.pnl_at_expiry(np.array(self._kinks())))))

    def _grid_extreme(self, pick: Callable[[Iterable[float]], float]) -> float:
        """Several expirations: value the later legs with the model at the first expiration (approximate;
        labelled as such by callers) over a wide grid of prices."""
        first = self.first_expiration
        assert first is not None
        from quantpulse.options.contracts import expiration_time

        when = expiration_time(first)
        top = max(self.strikes) * 3
        grid = np.linspace(0.01, top, 600)
        vol = float(self.meta.get("model_vol", 0.3))
        values = [self.pnl_at(float(s), when, vol) for s in grid]
        if self._slope_above() < -1e-9 and pick is min:
            return -math.inf
        if self._slope_above() > 1e-9 and pick is max:
            return math.inf
        return float(pick(values))

    def breakevens(self) -> list[float]:
        """Prices at the (first) expiration where the P&L is zero, lowest first."""
        if not self.single_expiry:
            return []
        xs = self._kinks()
        ys = [float(self.pnl_at_expiry(x)) for x in xs]
        out: list[float] = []
        for (x0, y0), (x1, y1) in zip(
            zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True), strict=False
        ):
            if y0 == 0.0:
                out.append(x0)
            elif y0 * y1 < 0:
                out.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))
        slope = self._slope_above()
        if ys[-1] == 0.0:
            out.append(xs[-1])
        elif slope != 0 and ys[-1] * slope < 0:  # it crosses zero above the highest strike
            out.append(xs[-1] - ys[-1] / slope)
        return sorted({round(b, 6) for b in out if b > 0})

    def capital_required(self) -> float:
        """Cash tied up per unit: the maximum loss for defined-risk structures (a debit is its own maximum
        loss; a credit spread holds its width less the credit; a cash-secured put holds the strike less the
        premium; a covered call the shares less the premium). ``inf`` when the loss is unbounded."""
        loss = self.max_loss()
        return max(loss, self.debit(), 0.0)

    def reward_to_risk(self) -> float | None:
        loss, gain = self.max_loss(), self.max_profit()
        if not loss or math.isinf(loss):
            return None
        return math.inf if math.isinf(gain) else gain / loss

    # ------------------------------------------------------------------ risk character
    def naked_legs(self, *, cash_secured: bool = False) -> list[Leg]:
        """Short options not covered: a short call without a long call (same or later expiration) or shares;
        a short put without a long put (same or later expiration) — unless ``cash_secured`` (the strike's
        cash is set aside, which only a cash-secured put structure does)."""
        longs_by_kind: dict[str, list[OptionContract]] = {"call": [], "put": []}
        shares = 0
        for leg in self.legs:
            if leg.contract is None:
                shares += leg.sign * leg.units
            elif leg.side == "long":
                longs_by_kind[leg.contract.kind].extend([leg.contract] * leg.ratio)
        naked: list[Leg] = []
        for leg in self.legs:
            c = leg.contract
            if c is None or leg.side == "long":
                continue
            needed = leg.ratio
            covering = [lc for lc in longs_by_kind[c.kind] if lc.expiration >= c.expiration]
            used = min(needed, len(covering))
            for lc in covering[:used]:
                longs_by_kind[c.kind].remove(lc)
            needed -= used
            if needed and c.is_call and shares > 0:
                by_shares = min(needed, shares // c.multiplier)
                shares -= by_shares * c.multiplier
                needed -= by_shares
            if needed and not c.is_call and cash_secured:
                needed = 0
            if needed:
                naked.append(replace(leg, ratio=needed))
        return naked

    @property
    def defined_risk(self) -> bool:
        return math.isfinite(self.max_loss())

    # ------------------------------------------------------------------ Greeks
    def greeks(self, leg_greeks: Sequence[dict[str, float | None] | None]) -> dict[str, float | None]:
        """Net position Greeks per unit from each leg's per-share Greeks (``None`` for a stock leg): delta in
        shares, gamma in shares per $1, theta in dollars per day, vega in dollars per volatility point, rho in
        dollars per 1% rate. A Greek missing on any option leg makes the net value unknown (``None``) rather
        than wrong."""
        out: dict[str, float | None] = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for leg, g in zip(self.legs, leg_greeks, strict=True):
            if leg.contract is None:  # shares: delta 1 each, no other Greek
                if out["delta"] is not None:
                    out["delta"] = out["delta"] + leg.sign * leg.units
                continue
            for name in out:
                value = None if g is None else g.get(name)
                if value is None or out[name] is None:
                    out[name] = None
                else:
                    out[name] = out[name] + leg.sign * leg.units * value  # type: ignore[operator]
        return out

    def model_greeks(self, spot: float, now: datetime, vol: float | Callable[[OptionContract], float],
                     rate: float = 0.04, div: float = 0.0) -> dict[str, float | None]:  # fmt: skip
        per_leg: list[dict[str, float | None] | None] = []
        for leg in self.legs:
            if leg.contract is None:
                per_leg.append(None)
                continue
            sigma = vol(leg.contract) if callable(vol) else vol
            v = model_greeks(
                leg.contract.kind, spot, leg.contract.strike, leg.contract.years(now), sigma, rate, div
            )
            per_leg.append(
                {"delta": v.delta, "gamma": v.gamma, "theta": v.theta, "vega": v.vega, "rho": v.rho}
            )
        return self.greeks(per_leg)

    def summary(self) -> dict[str, Any]:
        loss, gain = self.max_loss(), self.max_profit()
        return {
            "family": self.family,
            "underlying": self.underlying,
            "quantity": self.quantity,
            "legs": [leg.label() for leg in self.legs],
            "debit": round(self.debit(), 2),
            "max_loss": None if math.isinf(loss) else round(loss, 2),
            "max_profit": None if math.isinf(gain) else round(gain, 2),
            "unbounded_loss": math.isinf(loss),
            "unbounded_profit": math.isinf(gain),
            "breakevens": [round(b, 2) for b in self.breakevens()],
            "capital_required": None
            if math.isinf(self.capital_required())
            else round(self.capital_required(), 2),
            "defined_risk": self.defined_risk,
            "expirations": [e.isoformat() for e in self.expirations],
        }


# --------------------------------------------------------------------------- constructors
def _need(contract: OptionContract, kind: str) -> None:
    if contract.kind != kind:
        raise StructureError(f"expected a {kind}, got a {contract.kind}")


def _same_expiry(*contracts: OptionContract) -> None:
    if len({c.expiration for c in contracts}) != 1:
        raise StructureError("the legs of this structure share one expiration")


def long_call(c: OptionContract, price: float) -> Structure:
    _need(c, "call")
    return Structure("long_call", c.underlying, (Leg("long", 1, price, c),))


def long_put(c: OptionContract, price: float) -> Structure:
    _need(c, "put")
    return Structure("long_put", c.underlying, (Leg("long", 1, price, c),))


def bull_call_spread(
    long: OptionContract, long_price: float, short: OptionContract, short_price: float
) -> Structure:
    _need(long, "call")
    _need(short, "call")
    _same_expiry(long, short)
    if not long.strike < short.strike:
        raise StructureError("a bull call spread buys the lower strike and sells the higher")
    return Structure(
        "bull_call_spread",
        long.underlying,
        (Leg("long", 1, long_price, long), Leg("short", 1, short_price, short)),
    )


def bear_put_spread(
    long: OptionContract, long_price: float, short: OptionContract, short_price: float
) -> Structure:
    _need(long, "put")
    _need(short, "put")
    _same_expiry(long, short)
    if not long.strike > short.strike:
        raise StructureError("a bear put spread buys the higher strike and sells the lower")
    return Structure(
        "bear_put_spread",
        long.underlying,
        (Leg("long", 1, long_price, long), Leg("short", 1, short_price, short)),
    )


def bull_put_spread(
    short: OptionContract, short_price: float, long: OptionContract, long_price: float
) -> Structure:
    _need(long, "put")
    _need(short, "put")
    _same_expiry(long, short)
    if not short.strike > long.strike:
        raise StructureError("a bull put spread sells the higher strike and buys the lower")
    return Structure(
        "bull_put_spread",
        short.underlying,
        (Leg("short", 1, short_price, short), Leg("long", 1, long_price, long)),
    )


def bear_call_spread(
    short: OptionContract, short_price: float, long: OptionContract, long_price: float
) -> Structure:
    _need(long, "call")
    _need(short, "call")
    _same_expiry(long, short)
    if not short.strike < long.strike:
        raise StructureError("a bear call spread sells the lower strike and buys the higher")
    return Structure(
        "bear_call_spread",
        short.underlying,
        (Leg("short", 1, short_price, short), Leg("long", 1, long_price, long)),
    )


def long_straddle(
    call: OptionContract, call_price: float, put: OptionContract, put_price: float
) -> Structure:
    _need(call, "call")
    _need(put, "put")
    _same_expiry(call, put)
    if call.strike != put.strike:
        raise StructureError("a straddle's call and put share a strike")
    return Structure(
        "long_straddle", call.underlying, (Leg("long", 1, call_price, call), Leg("long", 1, put_price, put))
    )


def long_strangle(
    call: OptionContract, call_price: float, put: OptionContract, put_price: float
) -> Structure:
    _need(call, "call")
    _need(put, "put")
    _same_expiry(call, put)
    if not put.strike < call.strike:
        raise StructureError("a strangle's put strike is below its call strike")
    return Structure(
        "long_strangle", call.underlying, (Leg("long", 1, call_price, call), Leg("long", 1, put_price, put))
    )


def iron_condor(
    long_put: OptionContract, long_put_price: float, short_put: OptionContract, short_put_price: float,
    short_call: OptionContract, short_call_price: float, long_call: OptionContract, long_call_price: float,
) -> Structure:  # fmt: skip
    _same_expiry(long_put, short_put, short_call, long_call)
    for c, k in ((long_put, "put"), (short_put, "put"), (short_call, "call"), (long_call, "call")):
        _need(c, k)
    if not (long_put.strike < short_put.strike <= short_call.strike < long_call.strike):
        raise StructureError("an iron condor's strikes run long put < short put ≤ short call < long call")
    return Structure("iron_condor", short_put.underlying, (
        Leg("long", 1, long_put_price, long_put), Leg("short", 1, short_put_price, short_put),
        Leg("short", 1, short_call_price, short_call), Leg("long", 1, long_call_price, long_call),
    ))  # fmt: skip


def call_butterfly(
    low: OptionContract,
    low_price: float,
    mid: OptionContract,
    mid_price: float,
    high: OptionContract,
    high_price: float,
) -> Structure:
    for c in (low, mid, high):
        _need(c, "call")
    _same_expiry(low, mid, high)
    if not (low.strike < mid.strike < high.strike):
        raise StructureError("a butterfly's strikes run low < middle < high")
    return Structure("call_butterfly", mid.underlying, (
        Leg("long", 1, low_price, low), Leg("short", 2, mid_price, mid), Leg("long", 1, high_price, high),
    ))  # fmt: skip


def put_butterfly(
    high: OptionContract,
    high_price: float,
    mid: OptionContract,
    mid_price: float,
    low: OptionContract,
    low_price: float,
) -> Structure:
    for c in (low, mid, high):
        _need(c, "put")
    _same_expiry(low, mid, high)
    if not (low.strike < mid.strike < high.strike):
        raise StructureError("a butterfly's strikes run low < middle < high")
    return Structure("put_butterfly", mid.underlying, (
        Leg("long", 1, high_price, high), Leg("short", 2, mid_price, mid), Leg("long", 1, low_price, low),
    ))  # fmt: skip


def iron_butterfly(
    long_put: OptionContract, long_put_price: float, short_put: OptionContract, short_put_price: float,
    short_call: OptionContract, short_call_price: float, long_call: OptionContract, long_call_price: float,
) -> Structure:  # fmt: skip
    _same_expiry(long_put, short_put, short_call, long_call)
    for c, k in ((long_put, "put"), (short_put, "put"), (short_call, "call"), (long_call, "call")):
        _need(c, k)
    if not (long_put.strike < short_put.strike == short_call.strike < long_call.strike):
        raise StructureError(
            "an iron butterfly sells one strike's call and put, with a long wing below and above"
        )
    return Structure("iron_butterfly", short_put.underlying, (
        Leg("long", 1, long_put_price, long_put), Leg("short", 1, short_put_price, short_put),
        Leg("short", 1, short_call_price, short_call), Leg("long", 1, long_call_price, long_call),
    ))  # fmt: skip


def broken_wing_butterfly(
    high: OptionContract,
    high_price: float,
    mid: OptionContract,
    mid_price: float,
    low: OptionContract,
    low_price: float,
) -> Structure:
    """A put butterfly whose lower wing is wider than its upper one: the extra width usually pays the debit, and
    the risk (still limited by the low put) sits only below the lower strike."""
    for c in (low, mid, high):
        _need(c, "put")
    _same_expiry(low, mid, high)
    if not (low.strike < mid.strike < high.strike) or not (
        mid.strike - low.strike > high.strike - mid.strike
    ):
        raise StructureError("a broken-wing butterfly's lower wing is wider than its upper wing")
    return Structure("broken_wing_butterfly", mid.underlying, (
        Leg("long", 1, high_price, high), Leg("short", 2, mid_price, mid), Leg("long", 1, low_price, low),
    ))  # fmt: skip


def reverse_iron_condor(
    short_put: OptionContract, short_put_price: float, long_put: OptionContract, long_put_price: float,
    long_call: OptionContract, long_call_price: float, short_call: OptionContract, short_call_price: float,
) -> Structure:  # fmt: skip
    _same_expiry(short_put, long_put, long_call, short_call)
    for c, k in ((short_put, "put"), (long_put, "put"), (long_call, "call"), (short_call, "call")):
        _need(c, k)
    if not (short_put.strike < long_put.strike <= long_call.strike < short_call.strike):
        raise StructureError(
            "a reverse iron condor's strikes run short put < long put ≤ long call < short call"
        )
    return Structure("reverse_iron_condor", long_put.underlying, (
        Leg("short", 1, short_put_price, short_put), Leg("long", 1, long_put_price, long_put),
        Leg("long", 1, long_call_price, long_call), Leg("short", 1, short_call_price, short_call),
    ))  # fmt: skip


def covered_call(stock_price: float, call: OptionContract, call_price: float) -> Structure:
    _need(call, "call")
    return Structure("covered_call", call.underlying,
                     (Leg("long", call.multiplier, stock_price, None), Leg("short", 1, call_price, call)))  # fmt: skip


def cash_secured_put(put: OptionContract, put_price: float) -> Structure:
    _need(put, "put")
    return Structure(
        "cash_secured_put", put.underlying, (Leg("short", 1, put_price, put),), meta={"cash_secured": True}
    )


def protective_put(stock_price: float, put: OptionContract, put_price: float) -> Structure:
    _need(put, "put")
    return Structure("protective_put", put.underlying,
                     (Leg("long", put.multiplier, stock_price, None), Leg("long", 1, put_price, put)))  # fmt: skip


def collar(
    stock_price: float, put: OptionContract, put_price: float, call: OptionContract, call_price: float
) -> Structure:
    _need(put, "put")
    _need(call, "call")
    if not put.strike < call.strike:
        raise StructureError("a collar's put strike is below its call strike")
    return Structure("collar", put.underlying, (
        Leg("long", put.multiplier, stock_price, None), Leg("long", 1, put_price, put), Leg("short", 1, call_price, call),
    ))  # fmt: skip


def calendar(
    near: OptionContract, near_price: float, far: OptionContract, far_price: float, *, model_vol: float = 0.3
) -> Structure:
    if near.kind != far.kind or near.strike != far.strike or not near.expiration < far.expiration:
        raise StructureError(
            "a calendar sells a nearer and buys a later expiration of the same strike and kind"
        )
    return Structure("calendar", near.underlying, (Leg("short", 1, near_price, near), Leg("long", 1, far_price, far)),
                     meta={"model_vol": model_vol})  # fmt: skip


def stock(underlying: str, price: float, shares: int = 100) -> Structure:
    return Structure("stock", underlying, (Leg("long", shares, price, None),))


def classify_risk(s: Structure) -> dict[str, Any]:
    """What kind of risk this is, for the gates: defined or not, naked legs, and whether the family may be
    executed at all (naked families never)."""
    naked = s.naked_legs(cash_secured=bool(s.meta.get("cash_secured")))
    fam = FAMILIES.get(s.family)
    return {
        "family_known": fam is not None,
        "defined_risk": s.defined_risk and not naked,
        "naked_legs": [leg.label() for leg in naked],
        "never_executable": fam is None or not fam.defined_risk or bool(naked),
    }
