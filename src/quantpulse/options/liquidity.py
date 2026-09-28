"""Can this contract (or this structure) be traded without giving the edge away in the spread?

Each contract is scored on its spread (as a share of the mid, and in dollars), volume, open interest and
quote age, plus the underlying's own liquidity. A contract that fails any hard limit is rejected with the
reason; the score (0–1) orders the rest. The expected cost of crossing half the spread on every leg — in and
out — is reported in dollars, because that is what an option trade pays before anything else happens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from quantpulse.options.quotes import OptionQuote


@dataclass(frozen=True, slots=True)
class LiquidityRules:
    max_spread_pct: float = 0.15
    max_spread_dollars: float = 0.50
    min_volume: float = 10
    min_open_interest: float = 100
    min_underlying_dollar_volume: float = 20_000_000


@dataclass(frozen=True, slots=True)
class Liquidity:
    symbol: str
    ok: bool
    score: float
    reasons: tuple[str, ...]
    spread: float | None
    spread_pct: float | None
    volume: float | None
    open_interest: float | None
    round_trip_cost: float | None  # dollars per contract: half the spread paid in and again out

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "ok": self.ok,
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
            "spread": self.spread,
            "spread_pct": None if self.spread_pct is None else round(self.spread_pct, 4),
            "volume": self.volume,
            "open_interest": self.open_interest,
            "round_trip_cost": None if self.round_trip_cost is None else round(self.round_trip_cost, 2),
        }


def assess(
    q: OptionQuote, rules: LiquidityRules | None = None, underlying_dollar_volume: float | None = None
) -> Liquidity:
    r = rules or LiquidityRules()
    reasons: list[str] = []
    if not q.two_sided:
        reasons.append("no two-sided market")
    spct = q.spread_pct
    if spct is not None and spct > r.max_spread_pct:
        reasons.append(f"spread {spct:.0%} of the mid (limit {r.max_spread_pct:.0%})")
    if q.spread is not None and q.spread > r.max_spread_dollars and (spct or 0) > r.max_spread_pct / 2:
        reasons.append(f"spread ${q.spread:.2f} (limit ${r.max_spread_dollars:.2f})")
    if q.volume is not None and q.volume < r.min_volume:
        reasons.append(f"volume {q.volume:g} today (minimum {r.min_volume:g})")
    if q.open_interest is None:
        reasons.append("open interest unknown")
    elif q.open_interest < r.min_open_interest:
        reasons.append(f"open interest {q.open_interest:g} (minimum {r.min_open_interest:g})")
    if underlying_dollar_volume is not None and underlying_dollar_volume < r.min_underlying_dollar_volume:
        reasons.append(f"the underlying trades ${underlying_dollar_volume / 1e6:.0f}M a day (minimum "
                       f"${r.min_underlying_dollar_volume / 1e6:.0f}M)")  # fmt: skip
    parts = []
    if spct is not None:
        parts.append(max(0.0, 1 - spct / r.max_spread_pct))
    if q.open_interest is not None:
        parts.append(min(1.0, q.open_interest / (10 * r.min_open_interest)))
    if q.volume is not None:
        parts.append(min(1.0, q.volume / (10 * r.min_volume)))
    score = sum(parts) / len(parts) if parts else 0.0
    cost = q.spread * q.contract.multiplier if q.spread is not None else None  # ½ in + ½ out = one spread
    return Liquidity(q.symbol, not reasons, score if not reasons else min(score, 0.2), tuple(reasons), q.spread, spct,
                     q.volume, q.open_interest, cost)  # fmt: skip


def structure_cost(quotes: Sequence[OptionQuote], ratios: Sequence[int]) -> float | None:
    """Round-trip spread cost of a multi-leg structure (dollars per unit): every leg pays its spread."""
    total = 0.0
    for q, n in zip(quotes, ratios, strict=True):
        if q.spread is None:
            return None
        total += q.spread * q.contract.multiplier * n
    return total


def collapsed(before: Sequence[OptionQuote], after: Sequence[OptionQuote], widen: float = 2.0) -> list[str]:
    """Contracts whose spread widened by ``widen``× or more (or lost a side): a liquidity collapse."""
    was = {q.symbol: q.spread for q in before}
    out = []
    for q in after:
        prev = was.get(q.symbol)
        if prev and (q.spread is None or q.spread >= widen * prev):
            out.append(q.symbol)
    return out


def fresh_enough(q: OptionQuote, now: datetime, max_age: float) -> bool:
    age = q.age(now)
    return age is not None and 0 <= age <= max_age
