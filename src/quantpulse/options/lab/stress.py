"""The tail-risk lab and the regime-transition lab: no strategy graduates on calm markets alone.

**Tail scenarios** revalue a structure (every leg, with the model at the shocked inputs) under:

==================  ========================================================================================
crash               the underlying gaps down 20% overnight, IV up 60%
melt_up             the underlying gaps up 15%, IV down 20%
vol_spike           IV doubles, the price unchanged
iv_crush            IV halves (after an event), the price unchanged
iv_expansion        IV up 50%
gap_through_strike  the price jumps to one width beyond the short strike (or 10% against a long premium)
liquidity_collapse  spreads four times wider: exit at the far side of a quadrupled spread
correlated_failure  every open position suffers its own worst scenario at once (the book, not a trade)
==================  ========================================================================================

Held to expiration, a defined-risk structure can never lose more than its maximum loss, whatever happens —
a result beyond it would be a bug, and is reported as a breach. Closing *early* in a crash can cost more than
that maximum, by the spreads paid to get out; that is real and reported separately
(``early_exit_beyond_max_loss``), never hidden.

**Regime transitions**: trades are grouped by whether their life spanned a change of regime (LOW_VOL→HIGH_VOL,
BULL→BEAR, TREND→RANGE, RANGE→TREND, NORMAL→EVENT) and the expectancy of each group is compared with the
trades that saw no change. Many strategies are fine inside a regime and fail across its edge.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from quantpulse.options.pricing import greeks
from quantpulse.options.structures import Structure


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    spot_move: float  # relative
    iv_mult: float
    spread_mult: float = 1.0


SCENARIOS = (
    Scenario("crash", -0.20, 1.6),
    Scenario("melt_up", 0.15, 0.8),
    Scenario("vol_spike", 0.0, 2.0),
    Scenario("iv_crush", 0.0, 0.5),
    Scenario("iv_expansion", 0.0, 1.5),
    Scenario("liquidity_collapse", 0.0, 1.0, 4.0),
)


def revalue(s: Structure, spot: float, now: datetime, iv: float, *, rate: float = 0.04, half_spread_pct: float = 0.03,
            spread_mult: float = 1.0) -> float:  # fmt: skip
    """What closing the position (one unit) would bring at these inputs, crossing the (widened) spread."""
    value = 0.0
    for leg in s.legs:
        if leg.contract is None:
            value += leg.sign * leg.units * spot
            continue
        c = leg.contract
        px = greeks(c.kind, spot, c.strike, c.years(now), max(iv, 0.01), rate).price
        half = max(0.025, half_spread_pct * px) * spread_mult
        exit_px = (
            px - half if leg.sign > 0 else px + half
        )  # sell longs at the bid, buy back shorts at the ask
        value += leg.sign * leg.units * max(exit_px, 0.0)
    return value


def tail(
    s: Structure, spot: float, now: datetime, iv: float, *, entry_value: float | None = None
) -> dict[str, Any]:
    """P&L per unit under every scenario, relative to the entry value (the debit paid)."""
    entry = s.debit() if entry_value is None else entry_value
    max_loss = s.max_loss()
    out: dict[str, Any] = {}
    for sc in SCENARIOS:
        v = revalue(s, spot * (1 + sc.spot_move), now, iv * sc.iv_mult, spread_mult=sc.spread_mult)
        out[sc.name] = round(v - entry, 2)
    strikes = s.strikes
    shorts = [
        leg.contract.strike for leg in s.option_legs if leg.side == "short" and leg.contract is not None
    ]
    if shorts:
        worst = None
        for k in shorts:
            for target in (k * 0.9, k * 1.1):
                v = revalue(s, target, now, iv * 1.3) - entry
                worst = v if worst is None else min(worst, v)
        out["gap_through_strike"] = round(worst or 0.0, 2)
    elif strikes:
        against = 0.9 if any(leg.contract.is_call for leg in s.option_legs if leg.contract) else 1.1
        out["gap_through_strike"] = round(revalue(s, spot * against, now, iv) - entry, 2)
    worst_name = min(out, key=lambda k: out[k])
    # the invariant: held to expiration, a defined-risk structure never loses more than its maximum loss
    settle: dict[str, float] = {}
    if s.single_expiry:
        for sc in SCENARIOS:
            settle[sc.name] = round(float(s.pnl_at_expiry(spot * (1 + sc.spot_move))), 2)
    breach = math.isfinite(max_loss) and any(-v > max_loss + 1e-6 for v in settle.values())
    exit_cost = max(0.0, -out[worst_name] - max_loss) if math.isfinite(max_loss) else None
    return {"scenarios": out, "worst": worst_name, "worst_pnl": out[worst_name],
            "held_to_expiration": settle,
            "max_loss": None if math.isinf(max_loss) else round(max_loss, 2),
            "early_exit_beyond_max_loss": None if exit_cost is None else round(exit_cost, 2),
            "breaches_max_loss": breach}  # fmt: skip


def correlated_failure(
    positions: Sequence[tuple[Structure, int, float, float]], now: datetime
) -> dict[str, Any]:
    """Every position (structure, quantity, spot, iv) takes its own worst scenario at the same time."""
    total, parts = 0.0, []
    for st, qty, spot, iv in positions:
        t = tail(st, spot, now, iv)
        total += t["worst_pnl"] * qty
        parts.append(
            {"position": st.describe(), "scenario": t["worst"], "pnl": round(t["worst_pnl"] * qty, 2)}
        )
    return {"total": round(total, 2), "positions": parts}


TRANSITIONS = {
    ("LOW_IV", "HIGH_IV"): "LOW_VOL→HIGH_VOL",
    ("NORMAL_IV", "HIGH_IV"): "LOW_VOL→HIGH_VOL",
    ("HIGH_IV", "LOW_IV"): "HIGH_VOL→LOW_VOL",
    ("TRENDING_UP", "TRENDING_DOWN"): "BULL→BEAR",
    ("TRENDING_DOWN", "TRENDING_UP"): "BEAR→BULL",
    ("TRENDING_UP", "MEAN_REVERTING"): "TREND→RANGE",
    ("TRENDING_DOWN", "MEAN_REVERTING"): "TREND→RANGE",
    ("MEAN_REVERTING", "TRENDING_UP"): "RANGE→TREND",
    ("MEAN_REVERTING", "TRENDING_DOWN"): "RANGE→TREND",
    ("CALM", "PANIC"): "NORMAL→STRESS",
    ("TRENDING_UP", "PANIC"): "NORMAL→STRESS",
}


def transitions(trades: Sequence[dict[str, Any]], regime_at: Any) -> dict[str, Any]:
    """Group trades by the regime change their life spanned. ``regime_at(underlying, iso_date)`` returns
    ``(trend_regime, iv_regime)`` for a day (the same measured labels the entries used)."""
    groups: dict[str, list[float]] = {}
    for t in trades:
        start = (str(t.get("regime")), str(t.get("iv_regime")))
        end = regime_at(t["underlying"], t["exit_date"])
        if end is None:
            continue
        label = TRANSITIONS.get((start[1], end[1])) or TRANSITIONS.get((start[0], end[0])) or "none"
        risk = t.get("max_loss") or 1.0
        groups.setdefault(label, []).append(t["pnl"] / risk)
    out = {k: {"trades": len(v), "expectancy_on_risk": round(sum(v) / len(v), 4)} for k, v in groups.items()}
    base = out.get("none", {}).get("expectancy_on_risk")
    fragile = [k for k, v in out.items() if k != "none" and v["trades"] >= 3 and base is not None and
               v["expectancy_on_risk"] < min(0.0, base) - 0.1]  # fmt: skip
    return {"groups": out, "fragile_across": fragile}


def strategy_tail(
    trades: Sequence[dict[str, Any]], equity: float, *, loss_budget: float = 0.10
) -> dict[str, Any]:
    """Book-level check from a backtest: if every position open at the busiest moment hit its maximum loss
    at once, what share of equity would go? (Uses the recorded maximum losses: defined risk only.)"""
    by_day: dict[str, float] = {}
    for t in trades:
        from datetime import date, timedelta

        d0, d1 = date.fromisoformat(t["entry_date"]), date.fromisoformat(t["exit_date"])
        d = d0
        while d <= d1:
            by_day[d.isoformat()] = by_day.get(d.isoformat(), 0.0) + float(t.get("max_loss") or 0.0)
            d += timedelta(days=7)
    worst = max(by_day.values(), default=0.0)
    share = worst / equity if equity else math.inf
    return {"worst_simultaneous_max_loss": round(worst, 2), "share_of_equity": round(share, 4),
            "passed": share <= loss_budget, "budget": loss_budget}  # fmt: skip
