"""The strategy generator: the lab keeps trying new ideas once the catalogue templates are tested.

Each batch mixes three kinds of candidate, all built from the same point-in-time price features:

* **mutations** of the strategies with the best out-of-sample record so far: one change each (a weight halved
  or raised by half, a research-backed feature added, the weakest feature dropped, the uptrend filter added or
  removed, the weighting switched), so the search climbs from what has worked;
* **combinations** of the features the feature research found to rank returns, each with the sign of its
  information coefficient;
* **novelty**: the least-explored corner of the feature space (the features tried least so far, now and then
  with a filter nothing has used yet), so the search keeps going where it has not been;
* **exploration**: one random pair of features, so the search does not only circle what it already knows.

Nothing is tried twice: a candidate is identified by what it does (its fingerprint), not by its name. The
generator only proposes. Each candidate faces the full validation (walk-forward, random portfolios, stress,
the deflated Sharpe counting every strategy ever tried), then forward shadow tracking; only a person can
promote one. Seeded by the date, so a batch can be reproduced.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from quantpulse.domain.features import FEATURES

from .spec import GRID, WEIGHTINGS, StrategySpec

TREND = "trend_50_200"
# features whose sign means something on its own, so "only where it is ≥ 0" is a real condition
SIGNED = tuple(f for f in ("mom_12_1", "mom_6_1", "mom_3m", "ret_1m", "ret_5d", "trend_50_200", "px_vs_sma50",
                            "sharpe_126", "skew_63") if f in FEATURES)  # fmt: skip
MAX_SIGNALS = 4  # a strategy blends at most this many features (simpler rules overfit less)


def fingerprint(spec: StrategySpec) -> str:
    """What the strategy does, independent of its name: its signal, filters, rebalance and weighting."""
    body = {
        "signal": {k: round(v, 3) for k, v in sorted(spec.signal.items()) if round(v, 3) != 0},
        "filters": {k: round(v, 3) for k, v in sorted(spec.filters.items())},
        "rebalance_days": spec.rebalance_days,
        "weighting": spec.weighting,
    }
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]


@dataclass
class Evidence:
    """What the lab knows when it proposes."""

    tried: set[str] = field(default_factory=set)  # fingerprints already in the lab
    leaders: list[StrategySpec] = field(default_factory=list)  # best out-of-sample active Sharpe first
    features: dict[str, float] = field(default_factory=dict)  # feature -> mean rank IC (research-backed)
    usage: dict[str, int] = field(default_factory=dict)  # feature -> strategies tried that rank on it
    filter_usage: dict[str, int] = field(
        default_factory=dict
    )  # feature -> strategies tried that filter on it


@dataclass(frozen=True)
class Candidate:
    spec: StrategySpec
    origin: str  # how it came about, for the record


def _label(signal: dict[str, float], filters: dict[str, float]) -> str:
    parts = [
        f"{'−' if w < 0 else '+'}{name}" + (f"×{abs(w):g}" if abs(w) != 1 else "")
        for name, w in signal.items()
    ]
    return " ".join(parts) + (" (uptrends)" if TREND in filters else "")


def make(signal: dict[str, float], origin: str, *, filters: dict[str, float] | None = None,
         rebalance_days: int = 21, weighting: str = "equal") -> Candidate:  # fmt: skip
    signal = {k: round(v, 3) for k, v in signal.items() if round(v, 3) != 0}
    filters = dict(filters or {})
    probe = StrategySpec(id="probe", version=1, name="probe", description="", signal=signal, filters=filters,
                         rebalance_days=rebalance_days, weighting=weighting)  # fmt: skip
    label = _label(signal, filters)
    spec = replace(
        probe,
        id=f"gen-{fingerprint(probe)}",
        name=f"Generated: {label}"[:120],
        description=f"{origin}. Signal {label}; rebalanced every {rebalance_days} sessions, {weighting} weights.",
        grid=dict(GRID),
    )
    return Candidate(spec, origin)


def mutations(parent: StrategySpec, features: dict[str, float]) -> list[Candidate]:
    """One change at a time to a strategy that has done well out of sample."""
    out: list[Candidate] = []

    def child(signal: dict[str, float], change: str) -> Candidate:
        return make(signal, f"mutation of {parent.key}: {change}", filters=parent.filters,
                    rebalance_days=parent.rebalance_days, weighting=parent.weighting)  # fmt: skip

    for name, w in parent.signal.items():
        for factor in (0.5, 1.5):
            out.append(child({**parent.signal, name: w * factor}, f"{name} ×{factor:g}"))
    if len(parent.signal) > 1:
        weakest = min(parent.signal, key=lambda k: abs(parent.signal[k]))
        rest = {k: v for k, v in parent.signal.items() if k != weakest}
        out.append(child(rest, f"without {weakest}"))
    if len(parent.signal) < MAX_SIGNALS:
        for name, ic in sorted(features.items(), key=lambda kv: -abs(kv[1])):
            if name not in parent.signal:
                sign = 1.0 if ic > 0 else -1.0
                added = f"adds {'+' if sign > 0 else '−'}{name} (research-backed)"
                out.append(child({**parent.signal, name: 0.5 * sign}, added))
    if TREND in parent.filters:
        out.append(make(parent.signal, f"mutation of {parent.key}: without the uptrend filter",
                        filters={k: v for k, v in parent.filters.items() if k != TREND},
                        rebalance_days=parent.rebalance_days, weighting=parent.weighting))  # fmt: skip
    else:
        out.append(make(parent.signal, f"mutation of {parent.key}: uptrends only",
                        filters={**parent.filters, TREND: 0.0}, rebalance_days=parent.rebalance_days,
                        weighting=parent.weighting))  # fmt: skip
    other = next(w for w in WEIGHTINGS if w != parent.weighting)
    out.append(make(parent.signal, f"mutation of {parent.key}: {other} weights", filters=parent.filters,
                    rebalance_days=parent.rebalance_days, weighting=other))  # fmt: skip
    return out


def combinations(features: dict[str, float]) -> list[Candidate]:
    """Pairs of research-backed features, strongest first, each with the sign of its information coefficient."""
    ranked = sorted(features.items(), key=lambda kv: -abs(kv[1]))[:6]
    out = []
    for i, (a, ica) in enumerate(ranked):
        for b, icb in ranked[i + 1 :]:
            sa, sb = (1.0 if ica > 0 else -1.0), (1.0 if icb > 0 else -1.0)
            out.append(make({a: sa, b: 0.5 * sb}, f"combination of research-backed {a} and {b}"))
    return out


def exploration(rng: random.Random, features: dict[str, float]) -> Candidate:
    """A random pair of features; the sign from research where known, otherwise a coin toss."""
    a, b = rng.sample(sorted(FEATURES), 2)

    def sign(name: str) -> float:
        ic = features.get(name)
        return (1.0 if ic > 0 else -1.0) if ic else rng.choice((1.0, -1.0))

    return make({a: sign(a), b: 0.5 * sign(b)}, f"exploration: random pair {a}, {b}",
                rebalance_days=rng.choice((5, 21)))  # fmt: skip


def novel(rng: random.Random, evidence: Evidence) -> Candidate:
    """Two or three of the features tried least so far (ties at random; signs from research where known), and
    half the time a filter on a signed feature that nothing has filtered on yet."""
    order = sorted(FEATURES, key=lambda f: (evidence.usage.get(f, 0), rng.random()))
    chosen = [f for f in order if f in FEATURES][: rng.choice((2, 3))]

    def sign(name: str) -> float:
        ic = evidence.features.get(name)
        return (1.0 if ic > 0 else -1.0) if ic else rng.choice((1.0, -1.0))

    signal = {f: sign(f) * w for f, w in zip(chosen, (1.0, 0.5, 0.5), strict=False)}
    filters: dict[str, float] = {}
    if rng.random() < 0.5:
        pool = [f for f in SIGNED if f not in chosen]
        fresh = [f for f in pool if not evidence.filter_usage.get(f)]
        filters = {rng.choice(fresh or pool): 0.0}
    where = f", only where {next(iter(filters))} ≥ 0" if filters else ""
    return make(signal, f"novel: the least-explored features ({', '.join(chosen)}){where}", filters=filters,
                rebalance_days=rng.choice((5, 10, 21)))  # fmt: skip


def propose(evidence: Evidence, n: int, seed: str) -> list[Candidate]:
    """Up to ``n`` new candidates: about half mutations of the leaders, then combinations, one novel and one
    exploration; none already tried, none twice."""
    if n <= 0:
        return []
    rng = random.Random(seed)
    seen = set(evidence.tried)
    picked: list[Candidate] = []

    def take(pool: Iterable[Candidate], limit: int) -> None:
        pool = list(pool)
        rng.shuffle(pool)
        for c in pool:
            if len(picked) >= n or limit <= 0:
                return
            fp = fingerprint(c.spec)
            if fp in seen or any(f not in FEATURES for f in c.spec.signal):
                continue
            seen.add(fp)
            picked.append(c)
            limit -= 1

    top = evidence.leaders[:3]
    for parent in top:
        take(mutations(parent, evidence.features), max(1, (n + 1) // 2 // max(1, len(top))))
    take(combinations(evidence.features), n - len(picked) - 2)
    take([novel(rng, evidence) for _ in range(10)], 1)  # a few draws, in case the first were tried already
    explore(rng, evidence, take, picked, len(picked) + 1)
    for parent in evidence.leaders:  # still short: more mutations of the leaders ...
        take(mutations(parent, evidence.features), n - len(picked))
    explore(rng, evidence, take, picked, n)  # ... and, with little evidence yet, more exploration
    return picked


def explore(rng: random.Random, evidence: Evidence, take: Any, picked: list[Candidate], upto: int) -> None:
    """Random pairs until ``picked`` holds ``upto`` candidates (a bounded number of draws)."""
    for _ in range(50):
        if len(picked) >= upto:
            return
        take([exploration(rng, evidence.features)], 1)


def leaders(strategies: Sequence[dict[str, Any]]) -> list[StrategySpec]:
    """Strategies validated so far, best out-of-sample active Sharpe first (positive ones only)."""
    scored = []
    for r in strategies:
        sr = ((r.get("validation") or {}).get("walk_forward") or {}).get("oos_active_sharpe")
        if sr is not None and sr > 0 and r.get("status") != "retired":
            scored.append((sr, StrategySpec.from_dict(r["spec"])))
    return [s for _, s in sorted(scored, key=lambda x: -x[0])]


def research_features(learnings: Sequence[dict[str, Any]], max_q: float = 0.1) -> dict[str, float]:
    """The features the feature research found to rank returns (current conclusions; q below ``max_q``)."""
    out: dict[str, float] = {}
    for r in learnings:
        topic = str(r.get("topic") or "")
        if not topic.startswith("feature:") or topic.count(":") != 1:
            continue
        name = topic.split(":", 1)[1]
        stats = r.get("statistics") or {}
        q, effect = stats.get("p_value"), stats.get("effect")
        if name in FEATURES and effect and q is not None and q < max_q and r.get("status") != "REFUTED":
            out[name] = float(effect)
    return out
