"""The options book as a whole: net Greeks, exposure by every dimension, and hidden concentration.

Five option trades can secretly be one giant bet: calls on AAPL, MSFT and NVDA are one technology-growth
position, and a book of short puts across twenty names is one bet against a market fall. So exposure is
added up by underlying, sector, strategy, expiration, DTE bucket and direction; delta is also expressed in
dollars and beta-weighted to the market; and the concentration of each dimension (a Herfindahl index) is
reported with the positions that make it up.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

GREEKS = ("delta", "gamma", "theta", "vega", "rho")


@dataclass(frozen=True, slots=True)
class BookPosition:
    """One structure held: its net Greeks (per the whole position), its risk and its descriptors."""

    key: str
    underlying: str
    family: str
    strategy: str
    direction: str
    expiration: str | None
    dte: int | None
    greeks: Mapping[str, float | None]
    spot: float
    max_loss: float  # dollars (inf when unbounded — never held)
    premium: float  # dollars paid (negative: received)
    notional: float  # delta-equivalent exposure in dollars
    assignment_exposure: float = 0.0  # dollars of stock that assignment could deliver or take
    sector: str | None = None
    beta: float | None = None
    event_within_dte: bool = False


@dataclass
class BookReport:
    totals: dict[str, float | None] = field(default_factory=dict)
    dollar_delta: float = 0.0
    beta_weighted_delta: float | None = None
    max_loss: float = 0.0
    premium_at_risk: float = 0.0
    assignment_exposure: float = 0.0
    by: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    concentration: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "totals": {k: None if v is None else round(v, 4) for k, v in self.totals.items()},
            "dollar_delta": round(self.dollar_delta, 2),
            "beta_weighted_delta": None
            if self.beta_weighted_delta is None
            else round(self.beta_weighted_delta, 2),
            "max_loss": round(self.max_loss, 2),
            "premium_at_risk": round(self.premium_at_risk, 2),
            "assignment_exposure": round(self.assignment_exposure, 2),
            "by": self.by,
            "concentration": self.concentration,
            "warnings": self.warnings,
        }


def dte_bucket(dte: int | None) -> str:
    if dte is None:
        return "stock"
    for hi, name in ((0, "0DTE"), (7, "1-7"), (21, "8-21"), (45, "22-45"), (90, "46-90")):
        if dte <= hi:
            return name
    return "90+"


def herfindahl(weights: Sequence[float]) -> float:
    total = sum(abs(w) for w in weights)
    if total <= 0:
        return 0.0
    return sum((abs(w) / total) ** 2 for w in weights)


def aggregate(positions: Sequence[BookPosition], equity: float, *, sectors: Mapping[str, str] | None = None,
              concentration_limit: float = 0.5) -> BookReport:  # fmt: skip
    """Everything added up. ``concentration_limit`` is the Herfindahl index above which a dimension is
    flagged (0.5 ≈ one name carrying most of the risk)."""
    rep = BookReport()
    totals: dict[str, float | None] = dict.fromkeys(GREEKS, 0.0)
    dims: dict[str, defaultdict[str, dict[str, float]]] = {
        d: defaultdict(
            lambda: {"max_loss": 0.0, "dollar_delta": 0.0, "vega": 0.0, "theta": 0.0, "positions": 0.0}
        )
        for d in ("underlying", "sector", "strategy", "family", "expiration", "dte", "direction")
    }
    beta_total, beta_known = 0.0, True
    for p in positions:
        for g in GREEKS:
            v = p.greeks.get(g)
            totals[g] = None if v is None or totals[g] is None else totals[g] + v  # type: ignore[operator]
        delta = p.greeks.get("delta") or 0.0
        dollar_delta = delta * p.spot
        rep.dollar_delta += dollar_delta
        if p.beta is None:
            beta_known = False
        else:
            beta_total += dollar_delta * p.beta
        rep.max_loss += p.max_loss if math.isfinite(p.max_loss) else 0.0
        if not math.isfinite(p.max_loss):
            rep.warnings.append(f"{p.key}: unbounded loss in the book")
        rep.premium_at_risk += max(p.premium, 0.0)
        rep.assignment_exposure += p.assignment_exposure
        sector = p.sector or (sectors or {}).get(p.underlying) or "unknown"
        keys = {
            "underlying": p.underlying,
            "sector": sector,
            "strategy": p.strategy,
            "family": p.family,
            "expiration": p.expiration or "none",
            "dte": dte_bucket(p.dte),
            "direction": p.direction,
        }
        for dim, k in keys.items():
            row = dims[dim][k]
            row["max_loss"] += p.max_loss if math.isfinite(p.max_loss) else 0.0
            row["dollar_delta"] += dollar_delta
            row["vega"] += p.greeks.get("vega") or 0.0
            row["theta"] += p.greeks.get("theta") or 0.0
            row["positions"] += 1
        if p.event_within_dte:
            rep.warnings.append(f"{p.key}: an earnings or other event falls before expiration")
    rep.totals = totals
    rep.beta_weighted_delta = beta_total if beta_known and positions else None
    rep.by = {
        d: {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in rows.items()}
        for d, rows in dims.items()
    }
    for dim in ("underlying", "sector", "expiration", "direction", "strategy"):
        rows = dims[dim]
        weights = [r["max_loss"] for r in rows.values()]
        hhi = herfindahl(weights)
        top = max(rows.items(), key=lambda kv: kv[1]["max_loss"], default=None)
        rep.concentration[dim] = {"hhi": round(hhi, 3), "largest": top[0] if top else None,
                                  "n": len(rows)}  # fmt: skip
        if len(positions) >= 2 and hhi > concentration_limit and top is not None:
            share = top[1]["max_loss"] / max(sum(weights), 1e-9)
            rep.warnings.append(f"concentrated by {dim}: {top[0]} carries {share:.0%} of the maximum loss")
    if equity > 0 and abs(rep.dollar_delta) > equity:
        rep.warnings.append(f"net dollar delta ${rep.dollar_delta:,.0f} exceeds equity ${equity:,.0f}")
    same_direction = {k for k, v in dims["direction"].items() if v["positions"] >= 3}
    for d in same_direction:
        names = sorted({p.underlying for p in positions if p.direction == d})
        secs = {
            (p.sector or (sectors or {}).get(p.underlying) or "unknown")
            for p in positions
            if p.direction == d
        }
        if len(names) >= 3 and len(secs) == 1:
            rep.warnings.append(f"{len(names)} {d} positions in one sector ({next(iter(secs))}): {', '.join(names)} — "
                                "one bet, not several")  # fmt: skip
    return rep


def correlation_clusters(returns: Mapping[str, Sequence[float]], threshold: float = 0.7) -> list[list[str]]:
    """Groups of underlyings whose daily returns move together (correlation above ``threshold``)."""
    import numpy as np

    names = [n for n, r in returns.items() if len(r) >= 20]
    if len(names) < 2:
        return []
    n = min(len(returns[x]) for x in names)
    m = np.array([np.asarray(returns[x][-n:], dtype=float) for x in names])
    corr = np.corrcoef(m)
    groups: list[set[str]] = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            if corr[i, j] >= threshold:
                b = names[j]
                hit = [g for g in groups if a in g or b in g]
                merged = set().union(*hit) | {a, b} if hit else {a, b}
                groups = [g for g in groups if g not in hit] + [merged]
    return [sorted(g) for g in groups]


def now_dte(expiration: str | None, now: datetime) -> int | None:
    from datetime import date

    from quantpulse.core.market_calendar import NEW_YORK

    if not expiration:
        return None
    return (date.fromisoformat(expiration) - now.astimezone(NEW_YORK).date()).days
