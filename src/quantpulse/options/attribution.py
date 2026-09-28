"""Why did a position make or lose money? A Greek-by-Greek account of the change in its value.

Between two marks the change in value is explained, to second order, as

    delta · ΔS  +  ½ · gamma · ΔS²  +  theta · Δt  +  vega · Δσ  +  residual

(``delta`` and ``gamma`` in shares, ``theta`` in dollars per day, ``vega`` in dollars per volatility point —
the net position Greeks at the start). Execution is accounted separately: the gap between the fill and the
mid at entry and at exit, and fees. The residual is what the Taylor expansion does not capture (large moves,
the Greeks changing along the way, higher-order terms); it is reported, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any


@dataclass(frozen=True, slots=True)
class Mark:
    spot: float
    iv: float  # the position's representative IV (decimal)
    value: float  # the position's value in dollars at the mid
    days: float  # a time stamp in days (only differences matter)
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass(frozen=True, slots=True)
class Attribution:
    total: float
    delta: float
    gamma: float
    theta: float
    vega: float
    residual: float
    execution: float
    fees: float
    underlying_move: float
    iv_change: float  # volatility points

    def as_dict(self) -> dict[str, float]:
        return {k: round(getattr(self, k), 2) for k in self.__dataclass_fields__}

    def dominant(self) -> str:
        parts = {"delta": self.delta, "gamma": self.gamma, "theta": self.theta, "vega": self.vega,
                 "execution": self.execution + self.fees, "residual": self.residual}  # fmt: skip
        return max(parts, key=lambda k: abs(parts[k]))


def attribute(start: Mark, end: Mark, *, execution: float = 0.0, fees: float = 0.0) -> Attribution:
    """The change from ``start`` to ``end`` at mid prices, split by Greek; ``execution`` (negative: paid
    to cross spreads) and ``fees`` (negative) complete the realized P&L."""
    ds = end.spot - start.spot
    dt = end.days - start.days
    dvol = (end.iv - start.iv) * 100  # volatility points
    d = start.delta * ds
    g = 0.5 * start.gamma * ds * ds
    th = start.theta * dt
    v = start.vega * dvol
    change = end.value - start.value
    return Attribution(
        total=change + execution + fees,
        delta=d,
        gamma=g,
        theta=th,
        vega=v,
        residual=change - (d + g + th + v),
        execution=execution,
        fees=fees,
        underlying_move=ds,
        iv_change=dvol,
    )


def attribute_path(marks: list[Mark], *, execution: float = 0.0, fees: float = 0.0) -> Attribution:
    """Sum of step-by-step attributions along a path of daily marks: far smaller residual than one step."""
    if len(marks) < 2:
        raise ValueError("at least two marks")
    parts = [attribute(a, b) for a, b in pairwise(marks)]
    return Attribution(
        total=sum(p.total for p in parts) + execution + fees,
        delta=sum(p.delta for p in parts),
        gamma=sum(p.gamma for p in parts),
        theta=sum(p.theta for p in parts),
        vega=sum(p.vega for p in parts),
        residual=sum(p.residual for p in parts),
        execution=execution,
        fees=fees,
        underlying_move=marks[-1].spot - marks[0].spot,
        iv_change=(marks[-1].iv - marks[0].iv) * 100,
    )


def execution_cost(side_sign: int, fill: float, mid: float, units: int) -> float:
    """Dollars lost (negative) or gained against the mid on one leg: a buy (``side_sign`` +1) above the mid
    loses, a sell below it loses."""
    return -side_sign * (fill - mid) * units


def verdicts(a: Attribution, *, direction: str) -> dict[str, Any]:
    """Plain judgments for the lessons: was the direction right, did volatility help, what did time cost."""
    expected_sign = {"bullish": 1, "bearish": -1}.get(direction)
    return {
        "direction_correct": None if expected_sign is None else (a.underlying_move * expected_sign > 0),
        "volatility_helped": a.vega > 0,
        "time_decay_cost": a.theta,
        "execution_cost": a.execution + a.fees,
        "dominant_driver": a.dominant(),
        "unexplained_share": abs(a.residual) / max(abs(a.total), 1e-9),
    }
