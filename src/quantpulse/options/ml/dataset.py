"""Labelled rows for the edge model — and how much each one counts.

Where rows come from (each labelled with its **grade**, never mixed up):

* ``model`` — :func:`build_rows` over :class:`~quantpulse.options.lab.chains.ModelChains`: model-priced chains
  over years of real underlying prices. Many rows, cheap, but the option prices are a model's (the volatility
  premium, skew and spreads are assumptions): they teach the shape of payoffs against the path of the
  underlying, not the market's real mispricings.
* ``recorded`` — :func:`build_rows` over :class:`~quantpulse.options.lab.chains.RecordedChains`: chains
  QuantPulse recorded from Alpaca. Real quotes; few, and growing every session.
* ``shadow`` / ``paper`` — candidates the Options Brain opened, with the feature vector it recorded at the time
  and the realised result (:mod:`.service`).

For every (underlying, day) a fixed set of **probes** — one standard structure per family and delta — is built
from the chain the way the Brain builds candidates (:func:`quantpulse.options.selection.build`), evaluated by the
same rule, described by :func:`.features.candidate_features` and labelled by :func:`.labels.triple_barrier`.

**Weights.** Real evidence counts more than model-priced evidence (``GRADE_WEIGHTS``), and recent rows more
than old ones (exponential recency with a floor, like the strategy weights in the lab): the model learns the
payoff mechanics from the many model rows and the market's departures from them from the few real ones.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import numpy as np

from quantpulse.options.lab.chains import ChainSource, close_time
from quantpulse.options.lab.features import DayFeatures, history
from quantpulse.options.selection import Candidate, Spec, build, empirical_distribution, evaluate

from .features import FEATURES, GRADES, candidate_features, matrix
from .labels import ExitPolicy, Label, triple_barrier
from .surface import fit_surface


class HistorySource(ChainSource, Protocol):
    """A chain source that also lists the days it has (model-priced and recorded chains both do)."""

    def days(self, underlying: str) -> list[date]: ...


GRADE_WEIGHTS: dict[str, float] = {"model": 0.35, "recorded": 1.0, "shadow": 1.5, "paper": 2.0}


@dataclass(frozen=True, slots=True)
class Probe:
    family: str
    delta_target: float
    width_pct: float | None = None
    wing_pct: float | None = None
    dte_min: int = 20
    dte_max: int = 50


# one standard structure per family and a spread of deltas: the yardstick candidates of every kind
PROBES: tuple[Probe, ...] = (
    Probe("long_call", 0.50), Probe("long_call", 0.30), Probe("long_put", 0.50), Probe("long_put", 0.30),
    Probe("bull_call_spread", 0.50, width_pct=0.04), Probe("bear_put_spread", 0.50, width_pct=0.04),
    Probe("bull_put_spread", 0.25, width_pct=0.04), Probe("bear_call_spread", 0.25, width_pct=0.04),
    Probe("long_straddle", 0.50), Probe("long_strangle", 0.25, wing_pct=0.05),
    Probe("iron_condor", 0.20, width_pct=0.04), Probe("call_butterfly", 0.50, width_pct=0.04),
    Probe("put_butterfly", 0.50, width_pct=0.04), Probe("iron_butterfly", 0.50, wing_pct=0.05),
    Probe("broken_wing_butterfly", 0.40, width_pct=0.03), Probe("reverse_iron_condor", 0.30, width_pct=0.04),
    Probe("calendar", 0.50, dte_min=20, dte_max=40),
)  # fmt: skip


@dataclass
class Row:
    features: dict[str, float]
    ror: float
    t0: int  # entry day (ordinal)
    t1: int  # exit day (ordinal)
    underlying: str
    family: str
    grade: str
    barrier: str = ""


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    underlying: np.ndarray
    family: np.ndarray
    grade: np.ndarray
    names: tuple[str, ...] = FEATURES
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    @classmethod
    def from_rows(cls, rows: Sequence[Row]) -> Dataset:
        rows = sorted(rows, key=lambda r: (r.t0, r.underlying, r.family))
        return cls(
            X=matrix([r.features for r in rows]),
            y=np.array([r.ror for r in rows], dtype=float),
            t0=np.array([r.t0 for r in rows], dtype=np.int64),
            t1=np.array([r.t1 for r in rows], dtype=np.int64),
            underlying=np.array([r.underlying for r in rows], dtype=object),
            family=np.array([r.family for r in rows], dtype=object),
            grade=np.array([r.grade for r in rows], dtype=object),
        )

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(self.X[mask], self.y[mask], self.t0[mask], self.t1[mask], self.underlying[mask],
                       self.family[mask], self.grade[mask], self.names, dict(self.meta))  # fmt: skip

    def column(self, name: str) -> np.ndarray:
        return self.X[:, self.names.index(name)]

    def weights(self, *, half_life: float = 120.0, floor: float = 0.25,
                grade_weights: Mapping[str, float] = GRADE_WEIGHTS) -> np.ndarray:  # fmt: skip
        if not self.n:
            return np.empty(0)
        age = float(self.t0.max()) - self.t0.astype(float)
        recency = np.maximum(floor, 0.5 ** (age / half_life))
        by_grade = np.array([grade_weights.get(str(g), 1.0) for g in self.grade], dtype=float)
        return recency * by_grade

    def summary(self) -> dict[str, Any]:
        if not self.n:
            return {"rows": 0}
        first, last = date.fromordinal(int(self.t0.min())), date.fromordinal(int(self.t0.max()))
        return {
            "rows": self.n,
            "first": first.isoformat(),
            "last": last.isoformat(),
            "by_grade": {g: int((self.grade == g).sum()) for g in GRADES if (self.grade == g).any()},
            "by_family": {str(f): int((self.family == f).sum()) for f in sorted(set(self.family.tolist()))},
            "underlyings": len(set(self.underlying.tolist())),
            "mean_ror": round(float(self.y.mean()), 5),
            "win_rate": round(float((self.y > 0).mean()), 4),
        }


def _pick(cands: Sequence[Candidate], probe: Probe) -> Candidate | None:
    """The expiration nearest the middle of the window (never chosen on the outcome)."""
    if not cands:
        return None
    mid = 0.5 * (probe.dte_min + probe.dte_max)
    return min(cands, key=lambda c: abs(c.dte - mid))


def build_rows(
    source: HistorySource,
    underlyings: Sequence[str],
    closes: Mapping[str, Mapping[date, float]],
    *,
    grade: str,
    start: date | None = None,
    end: date | None = None,
    step: int = 3,
    probes: Sequence[Probe] = PROBES,
    policy: ExitPolicy = ExitPolicy(),
    events: Mapping[str, Sequence[date]] | None = None,
    deadline: float | None = None,
    max_rows: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[Row]:
    """Probe candidates on every ``step``-th day of each underlying, with their features and triple-barrier
    labels. All underlyings are walked together, **newest day first**, so a budget — ``deadline``
    (``time.monotonic()``), ``max_rows`` or ``should_stop()`` — drops the oldest history, never a whole symbol."""
    rows: list[Row] = []
    lo_dte = min(p.dte_min for p in probes)
    hi_dte = max(p.dte_max for p in probes) + 45  # calendars need a later expiration too
    ctx: dict[str, tuple[list[date], dict[date, DayFeatures], np.ndarray, dict[date, int]]] = {}
    plan: list[tuple[date, str]] = []
    for u in underlyings:
        days = list(source.days(u))
        cl = dict(sorted((closes.get(u) or {}).items()))
        if not days or len(cl) < 80:
            continue
        iv_by_day = {d: source.atm_iv(u, d) for d in days}
        feats: dict[date, DayFeatures] = history(cl, iv_by_day, (events or {}).get(u, ()))
        ctx[u] = (days, feats, np.array(list(cl.values()), dtype=float), {d: i for i, d in enumerate(cl)})
        plan += [(d, u) for d in days[::-1][::step] if not ((start and d < start) or (end and d > end))]
    plan.sort(key=lambda du: (du[0], du[1]), reverse=True)
    for d, u in plan:
        if deadline is not None and time.monotonic() > deadline:
            break
        if (max_rows is not None and len(rows) >= max_rows) or (should_stop and should_stop()):
            break
        days, feats, cvals, idx = ctx[u]
        day = feats.get(d)
        spot = source.spot(u, d)
        if day is None or day.rv20 is None or not spot or d not in idx:
            continue
        now = close_time(d)
        chain = source.chain(u, d, (lo_dte, hi_dte))
        if chain is None or not chain.quotes:
            continue
        surface = fit_surface(chain.quotes, spot, now, max_slices=8)
        sf = surface.features()
        past = cvals[: idx[d] + 1]
        for p in probes:
            spec = Spec(p.family, p.dte_min, p.dte_max, p.delta_target, p.width_pct, p.wing_pct,
                        max_spread_pct=1.0, min_open_interest=0)  # fmt: skip
            cand = _pick(build(spec, chain.quotes, spot, now), p)
            if cand is None:
                continue
            evaluate(cand, spot, now, fee_per_contract=policy.fee_per_contract)
            if not cand.metrics.get("max_loss"):
                continue
            terminal = empirical_distribution(spot, past.tolist(), cand.dte)
            if terminal is not None and cand.structure.single_expiry:
                emp = Candidate(cand.structure, cand.quotes, cand.expiration, cand.dte)
                evaluate(emp, spot, now, terminal=terminal, distribution="empirical",
                         fee_per_contract=policy.fee_per_contract)  # fmt: skip
                cand.metrics["empirical"] = {
                    k: emp.metrics.get(k) for k in ("expected_pnl", "pop", "expected_on_risk")
                }
            lab: Label | None = triple_barrier(cand, source, u, d, days, policy)
            if lab is None or not math.isfinite(lab.ror):
                continue
            x = candidate_features(cand, spot, now, day=day, closes=past.tolist(), surface=surface,
                                   grade=grade, surface_features=sf)  # fmt: skip
            rows.append(Row(x, lab.ror, d.toordinal(), lab.t1.toordinal(), u, p.family, grade, lab.barrier))
    return rows
