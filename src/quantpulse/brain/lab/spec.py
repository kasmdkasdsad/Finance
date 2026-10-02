"""Strategy definitions: declarative, versioned, and built only from point-in-time price features.

A strategy ranks the universe on a weighted sum of cross-sectional z-scores of features from the stock
model's feature library (:func:`quantpulse.domain.features.compute_features` — each value uses only data
up to its own date), optionally filtered (e.g. only names in an uptrend), holds the ``top_n`` best
long-only, and rebalances every ``rebalance_days`` sessions with a one-session execution lag and a
per-trade cost. A version never changes once created: a different rule is a new version, so every
result stays attributable to exactly the rule that produced it. ``grid`` lists the parameter values the
walk-forward test may choose between (each combination counts as a trial in the overfitting checks).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from quantpulse.core.errors import DomainError
from quantpulse.domain.features import FEATURES

WEIGHTINGS = ("equal", "inverse_vol")


@dataclass(frozen=True)
class StrategySpec:
    id: str
    version: int
    name: str
    description: str
    signal: dict[str, float]
    top_n: int = 10
    rebalance_days: int = 21
    weighting: str = "equal"
    filters: dict[str, float] = field(default_factory=dict)  # feature -> minimum value
    cost_bps: float = 10.0
    lag_days: int = 1
    grid: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [f for f in [*self.signal, *self.filters] if f not in FEATURES]
        if unknown:
            raise DomainError(
                f"unknown features {unknown}: use point-in-time price features {sorted(FEATURES)}"
            )
        if not self.signal or all(w == 0 for w in self.signal.values()):
            raise DomainError("a strategy needs at least one non-zero signal weight")
        if self.top_n < 1 or self.rebalance_days < 1 or self.lag_days < 1:
            raise DomainError("top_n, rebalance_days and lag_days must be at least 1")
        if self.weighting not in WEIGHTINGS:
            raise DomainError(f"weighting must be one of {WEIGHTINGS}")
        bad = [k for k in self.grid if k not in ("top_n", "rebalance_days", "weighting")]
        if bad:
            raise DomainError(f"only top_n, rebalance_days and weighting can be searched, not {bad}")

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategySpec:
        return cls(**d)

    def variants(self) -> list[StrategySpec]:
        """Every parameter combination in the grid (the spec itself when there is no grid)."""
        if not self.grid:
            return [self]
        keys = list(self.grid)
        return [
            replace(self, **dict(zip(keys, combo, strict=True)))
            for combo in itertools.product(*self.grid.values())
        ]

    def params(self) -> dict[str, Any]:
        return {"top_n": self.top_n, "rebalance_days": self.rebalance_days, "weighting": self.weighting}


GRID = {"top_n": [5, 10, 20], "rebalance_days": [5, 21]}

TEMPLATES: dict[str, dict[str, Any]] = {
    "momentum_12_1": {
        "name": "12-1 momentum",
        "description": "Buy the winners of the past year, skipping the latest month.",
        "signal": {"mom_12_1": 1.0},
    },
    "trend_momentum": {
        "name": "Momentum in uptrends",
        "description": "Blend of 12-1, 6-1 and 3-month momentum, only names whose 50-day average is above the 200-day.",
        "signal": {"mom_12_1": 0.5, "mom_6_1": 0.3, "mom_3m": 0.2},
        "filters": {"trend_50_200": 0.0},
    },
    "short_term_reversal": {
        "name": "Short-term reversal",
        "description": "Buy last week's and last month's laggards.",
        "signal": {"ret_5d": -1.0, "ret_1m": -0.5},
        "rebalance_days": 5,
    },
    "low_volatility": {
        "name": "Low volatility",
        "description": "Hold the calmest, lowest-beta names.",
        "signal": {"vol_63": -1.0, "beta_252": -0.5},
        "weighting": "inverse_vol",
    },
    "quality_trend": {
        "name": "Steady trend",
        "description": "Best 6-month risk-adjusted return, close to the 52-week high, 50-day above 200-day.",
        "signal": {"sharpe_126": 1.0, "high_52w": 0.5, "trend_50_200": 0.5},
    },
    "dip_in_uptrend": {
        "name": "Buy the dip in an uptrend",
        "description": "Lowest RSI among names in an uptrend (50-day above 200-day).",
        "signal": {"rsi_14": -1.0},
        "filters": {"trend_50_200": 0.0},
        "rebalance_days": 5,
    },
}


def from_template(template: str, version: int = 1, **overrides: Any) -> StrategySpec:
    if template not in TEMPLATES:
        raise DomainError(f"unknown template {template!r}; choose from {sorted(TEMPLATES)}")
    base = {**TEMPLATES[template], "grid": GRID}
    base.update(overrides)
    return StrategySpec(id=template, version=version, **base)
