"""Signal research: does each feature, on its own, rank future returns?

For every feature and horizon ``h`` the information coefficient (IC) is the per-date Spearman correlation
between the feature and the realised ``h``-day forward return across the universe. Reported:

* mean IC and a t-statistic computed on non-overlapping dates (every ``h``-th date), since overlapping
  forward windows would otherwise overstate significance;
* the share of dates with a positive IC;
* IC decay across horizons (does the signal work for days, weeks or months?);
* mean forward return by feature quintile at the main horizon, and the top-minus-bottom spread;
* the average cross-sectional correlation between features (which signals are really the same bet).

A |t| above ~2 is conventionally "significant", but with many features tested at once some will clear
that bar by luck; treat single results with scepticism.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpulse.core.errors import DomainError
from quantpulse.domain import features as feat


@dataclass(frozen=True, slots=True)
class HorizonIC:
    horizon: int
    mean_ic: float | None
    t_stat: float | None
    positive_share: float | None
    n_dates: int


@dataclass
class FeatureResearch:
    name: str
    description: str
    by_horizon: list[HorizonIC]
    quintile_returns: list[float | None]
    ic_series: pd.Series  # at the main horizon

    @property
    def spread(self) -> float | None:
        q = self.quintile_returns
        return None if q[0] is None or q[-1] is None else q[-1] - q[0]


@dataclass
class ResearchResult:
    horizons: list[int]
    main_horizon: int
    features: list[FeatureResearch]
    correlation: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    n_symbols: int


def _ic_summary(ic: pd.Series, horizon: int) -> HorizonIC:
    ic = ic.dropna()
    if ic.empty:
        return HorizonIC(horizon, None, None, None, 0)
    sub = ic.iloc[::horizon]
    sd = float(sub.std(ddof=1)) if len(sub) > 2 else 0.0
    t = float(sub.mean() / sd * math.sqrt(len(sub))) if sd > 0 else None
    return HorizonIC(horizon, float(ic.mean()), t, float((ic > 0).mean()), len(ic))


def _quintiles(signal: pd.DataFrame, realized: pd.DataFrame, buckets: int = 5) -> list[float | None]:
    ranks = signal.where(realized.notna()).rank(axis=1, pct=True)
    idx = np.ceil(ranks * buckets).clip(1, buckets) - 1
    out: list[float | None] = []
    for b in range(buckets):
        per_date = realized.where(idx == b).mean(axis=1).dropna()
        out.append(float(per_date.mean()) if len(per_date) else None)
    return out


def factor_research(
    raw: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 21, 63),
    main_horizon: int = 21,
    names: Sequence[str] | None = None,
) -> ResearchResult:
    hs = sorted({int(h) for h in horizons} | {main_horizon})
    if hs[0] < 1:
        raise DomainError("horizons must be >= 1")
    names = list(names or feat.FEATURES)
    forward = {h: feat.forward_returns(close, h) for h in hs}
    results: list[FeatureResearch] = []
    for name in names:
        if name not in raw:
            raise DomainError(f"unknown feature {name}")
        signal = raw[name]
        by_h = [_ic_summary(feat.row_spearman(signal, forward[h]), h) for h in hs]
        results.append(
            FeatureResearch(
                name=name,
                description=feat.all_features().get(name, name),
                by_horizon=by_h,
                quintile_returns=_quintiles(signal, forward[main_horizon]),
                ic_series=feat.row_spearman(signal, forward[main_horizon]).dropna(),
            )
        )
    valid = [r.ic_series for r in results if not r.ic_series.empty]
    if not valid:
        raise DomainError("not enough overlapping history to measure any signal")
    stacked = pd.DataFrame({n: feat.cross_sectional_z(raw[n]).stack(future_stack=True) for n in names})
    return ResearchResult(
        horizons=hs,
        main_horizon=main_horizon,
        features=results,
        correlation=stacked.corr(method="pearson", min_periods=30),
        start=min(s.index[0] for s in valid),
        end=max(s.index[-1] for s in valid),
        n_symbols=int(close.shape[1]),
    )
