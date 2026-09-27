"""Measured track records — computed only from evaluated predictions, never assigned.

For every source (each agent version, the consensus) and slice (all regimes and each regime; all time and
the last 90 days; each volatility environment — ``vol:low|normal|high`` by the benchmark's volatility at
the call; and each standard horizon — ``at:1d``, ``at:5d``, ``at:10d``, ``at:21d``, the same calls graded
at that horizon, to show where a signal actually works) the metrics are:

* **independent observations** — an agent that repeats a view every cycle makes many predictions that
  share one outcome. Calls on the same subject whose horizons overlap (made within one horizon of each
  other) form one *block* and count once; ``n_effective`` is the number of blocks, and every statistic
  below is computed on blocks, never on the raw count;
* **hit rate** (share of blocks right about the direction) with its 95% Wilson interval and a two-sided
  p-value against a coin flip; across all slices the p-values are adjusted for the false-discovery rate
  (Benjamini–Hochberg), because with many agents and regimes some will look good by chance;
* a **verdict**: *unproven* (fewer than ``min_observations`` independent observations), *no evidence either
  way*, *evidence of skill* or *evidence of harm* (only when the adjusted q-value is below 10%);
* **Brier score** of the implied probability ``0.5 + 0.5 × score × confidence`` that the call is right
  (0.25 is a coin flip; lower is better);
* **rank IC** — Spearman correlation between score and realised relative return across blocks (≥ 10);
* **benchmark-relative outcome** — the mean favourable relative return (direction × relative), and the
  same in units of each call's own risk (volatility over the horizon at entry): the *risk-adjusted* outcome;
* **luck** — the share of outcomes within half a standard deviation (noise: right or wrong, it says little);
* **timing** — how much of the outcome happened before the next close, when a decision could first be
  acted on (an execution effect, not skill);
* **data problems** — calls made on data that was not usable (stale, invalid, missing) are graded and
  counted but left out of the verdict and the reliability, which measure skill on valid inputs;
* **calibration** — hit rate and mean favourable return by confidence bucket;
* **reliability** — the vote weight used by the consensus. ``None`` (*unproven*, weight 1.0) below
  ``min_observations``; 1.0 while there is no significant evidence; otherwise ``1 + 4 × (bound − 0.5)``
  using the conservative end of the Wilson interval (the lower bound for skill, the upper for harm),
  bounded to 0.25…1.75 — so only statistically supported differences move a weight, and only by as much as
  the evidence supports.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import delete, select

from quantpulse.db.models import BrainAgentPerformanceRow, BrainPredictionRow
from quantpulse.db.session import Database

BUCKETS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))
RECENT = timedelta(days=90)
Z95 = 1.959964
FDR = 0.10  # false-discovery rate for a verdict
NOISE_Z = 0.5  # an outcome within half a standard deviation is noise
USABLE_DATA = frozenset({"fresh", "live", "market_closed"})


@dataclass(frozen=True)
class Graded:
    source: str
    version: str
    regime: str
    horizon: int
    score: float
    confidence: float
    relative: float
    hit: bool
    made_at: datetime
    direction: int
    subject: str = ""
    relative_z: float | None = None  # the relative return in units of the call's risk over its horizon
    timing: float | None = None  # favourable relative return before the next close (not capturable)
    data_ok: bool = True  # made on usable data
    market_vol: float | None = None  # the benchmark's realised volatility when the call was made
    by_horizon: dict[str, float] | None = None  # relative return at each standard horizon up to its own

    @property
    def favourable(self) -> float:
        return self.direction * self.relative


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 10:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def vol_environment(market_vol: float | None) -> str | None:
    """Calm, normal or stressed market (the benchmark's annualised 21-day volatility at the call)."""
    if market_vol is None:
        return None
    return "low" if market_vol < 0.12 else "normal" if market_vol < 0.20 else "high"


def at_horizon(r: Graded, h: str) -> Graded | None:
    """The same call graded at another horizon (``None`` when that horizon was not measured)."""
    rel = (r.by_horizon or {}).get(h)
    if rel is None:
        return None
    return replace(r, horizon=int(h), relative=rel, hit=r.direction * rel > 0, relative_z=None, timing=None)


def wilson(k: float, n: int, z: float = Z95) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion (``k`` may be fractional: blocks' average hit)."""
    if n <= 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def p_coin(k: float, n: int) -> float | None:
    """Two-sided p-value of ``k`` hits in ``n`` against a coin flip (normal approximation)."""
    if n <= 0:
        return None
    z = (k - n / 2) / math.sqrt(n / 4)
    return math.erfc(abs(z) / math.sqrt(2))


def benjamini_hochberg(p: Sequence[float | None]) -> list[float | None]:
    """False-discovery-rate adjusted q-values (``None`` stays ``None``)."""
    idx = [i for i, v in enumerate(p) if v is not None]
    m = len(idx)
    out: list[float | None] = [None] * len(p)
    running = 1.0
    for rank, i in sorted(enumerate(sorted(idx, key=lambda i: p[i] or 0.0), start=1), reverse=True):
        q = min(running, (p[i] or 0.0) * m / rank)
        running = q
        out[i] = min(q, 1.0)
    return out


def span_days(horizon: int) -> int:
    """Calendar days covered by ``horizon`` sessions (predictions closer than this share an outcome)."""
    return max(1, math.ceil(max(horizon, 1) * 7 / 5))


def blocks(rows: Sequence[Graded]) -> list[list[Graded]]:
    """Group calls whose horizons overlap on the same subject: each group is one independent observation.
    A block starts at a call and takes every later call on the subject made within one horizon of it."""
    by_subject: dict[str, list[Graded]] = defaultdict(list)
    for r in rows:
        by_subject[r.subject or f"#{id(r)}"].append(r)
    out: list[list[Graded]] = []
    for items in by_subject.values():
        items.sort(key=lambda r: r.made_at)
        current: list[Graded] = []
        start: datetime | None = None
        for r in items:
            if start is not None and (r.made_at - start).days < span_days(r.horizon):
                current.append(r)
                continue
            if current:
                out.append(current)
            current, start = [r], r.made_at
        if current:
            out.append(current)
    return out


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def verdict(n_effective: int, hit_rate: float | None, q: float | None, min_observations: int) -> str:
    if n_effective < min_observations or hit_rate is None:
        return "unproven"
    if q is not None and q < FDR:
        return "evidence of skill" if hit_rate > 0.5 else "evidence of harm"
    return "no evidence either way"


def reliability(verdict_: str, ci: tuple[float, float] | None) -> float | None:
    """The consensus weight (see the module docstring)."""
    if verdict_ == "unproven" or ci is None:
        return None
    if verdict_ == "evidence of skill":
        return round(max(0.25, min(1.75, 1 + 4 * (ci[0] - 0.5))), 4)
    if verdict_ == "evidence of harm":
        return round(max(0.25, min(1.75, 1 + 4 * (ci[1] - 0.5))), 4)
    return 1.0


def metrics(rows: Sequence[Graded], min_observations: int) -> dict[str, Any]:
    n = len(rows)
    valid = [r for r in rows if r.data_ok]
    groups = blocks(valid)
    n_eff = len(groups)
    block_hit = [sum(1 for r in g if r.hit) / len(g) for g in groups]
    k_eff = sum(block_hit)
    hit_rate = k_eff / n_eff if n_eff else None
    ci = wilson(k_eff, n_eff)
    p = p_coin(k_eff, n_eff)

    def p_right(r: Graded) -> float:
        return min(max(0.5 + 0.5 * abs(r.score) * r.confidence, 0.0), 1.0)

    block_brier = [sum((p_right(r) - (1.0 if r.hit else 0.0)) ** 2 for r in g) / len(g) for g in groups]
    favourable = [sum(r.favourable for r in g) / len(g) for g in groups]
    zs = [
        z
        for z in (_mean([r.relative_z for r in g if r.relative_z is not None]) for g in groups)
        if z is not None
    ]
    timing = [
        t for t in (_mean([r.timing for r in g if r.timing is not None]) for g in groups) if t is not None
    ]
    noise = [r for r in valid if r.relative_z is not None]
    calibration: list[dict[str, Any]] = []
    for lo, hi in BUCKETS:
        inside = [g for g in groups if lo <= sum(r.confidence for r in g) / len(g) < hi]
        if inside:
            calibration.append(
                {
                    "kind": "bucket",
                    "confidence": f"{lo:.1f}–{min(hi, 1.0):.1f}",
                    "n": len(inside),
                    "hit_rate": round(
                        sum(sum(1 for r in g if r.hit) / len(g) for g in inside) / len(inside), 4
                    ),
                    "mean_excess": round(
                        sum(sum(r.favourable for r in g) / len(g) for g in inside) / len(inside), 5
                    ),
                }
            )
    bullish = sum(1 for r in rows if r.direction > 0)
    z = (k_eff - n_eff / 2) / math.sqrt(n_eff / 4) if n_eff else None
    v = verdict(n_eff, hit_rate, p, min_observations)
    return {
        "n": n,
        "hits": sum(1 for r in rows if r.hit),
        "n_effective": n_eff,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "ci_low": round(ci[0], 4) if ci else None,
        "ci_high": round(ci[1], 4) if ci else None,
        "p_value": round(p, 5) if p is not None else None,
        "q_value": round(p, 5) if p is not None else None,  # replaced by the adjusted value in recompute()
        "verdict": v,
        "brier": round(sum(block_brier) / n_eff, 5) if n_eff else None,
        "ic": _spearman(
            [sum(r.score for r in g) / len(g) for g in groups],
            [sum(r.relative for r in g) / len(g) for g in groups],
        ),
        "mean_excess": round(sum(favourable) / n_eff, 5) if n_eff else None,
        "mean_excess_z": round(sum(zs) / len(zs), 4) if zs else None,
        "calibration": [
            *calibration,
            {
                "kind": "summary",
                "z": round(z, 2) if z is not None else None,
                "bullish_share": round(bullish / n, 3) if n else None,
                "noise_share": round(
                    sum(1 for r in noise if abs(r.relative_z or 0) < NOISE_Z) / len(noise), 3
                )
                if noise
                else None,
                "timing": round(sum(timing) / len(timing), 5) if timing else None,
                "data_problems": n - len(valid),
            },
        ],
        "reliability": reliability(v, ci),
    }


async def graded(db: Database) -> list[Graded]:
    async with db.session() as s:
        rows = (
            await s.scalars(select(BrainPredictionRow).where(BrainPredictionRow.status == "evaluated"))
        ).all()
    out: list[Graded] = []
    for r in rows:
        outcome = (r.context or {}).get("outcome") or {}
        state = (r.context or {}).get("data_state")
        out.append(
            Graded(
                source=r.source_id,
                version=r.source_version,
                regime=r.regime or "unknown",
                horizon=r.horizon_days,
                score=r.score,
                confidence=r.confidence,
                relative=float(r.realized_relative or 0.0),
                hit=bool(r.hit),
                made_at=r.made_at,
                direction=r.direction,
                subject=r.subject,
                relative_z=outcome.get("relative_z"),
                timing=outcome.get("timing"),
                data_ok=state is None or state in USABLE_DATA,
                market_vol=(r.context or {}).get("market_vol"),
                by_horizon=outcome.get("by_horizon"),
            )
        )
    return out


async def recompute(db: Database, now: datetime, min_observations: int) -> int:
    """Rebuild ``brain_agent_performance`` from every evaluated prediction; returns the rows written.
    The p-values of all slices are adjusted together for the false-discovery rate."""
    rows = await graded(db)
    groups: dict[tuple[str, str, str, str], list[Graded]] = defaultdict(list)
    for r in rows:
        env = vol_environment(r.market_vol)
        for regime in ("all", r.regime, *([f"vol:{env}"] if env else [])):
            groups[(r.source, r.version, regime, "all")].append(r)
            if now - r.made_at <= RECENT:
                groups[(r.source, r.version, regime, "90d")].append(r)
        for h in r.by_horizon or {}:  # the same calls graded at each horizon: where does the signal work?
            g = at_horizon(r, h)
            if g is not None:
                groups[(r.source, r.version, f"at:{h}d", "all")].append(g)
    keys = list(groups)
    computed = [metrics(groups[k], min_observations) for k in keys]
    qs = benjamini_hochberg([m["p_value"] for m in computed])
    async with db.session() as s:
        await s.execute(delete(BrainAgentPerformanceRow))
        for (source, version, regime, window), m, q in zip(keys, computed, qs, strict=True):
            items = groups[(source, version, regime, window)]
            v = verdict(m["n_effective"], m["hit_rate"], q, min_observations)
            ci = (m["ci_low"], m["ci_high"]) if m["ci_low"] is not None else None
            s.add(
                BrainAgentPerformanceRow(
                    agent_id=source,
                    agent_version=version,
                    regime=regime,
                    horizon_days=round(sum(i.horizon for i in items) / len(items)),
                    window=window,
                    n=m["n"],
                    hits=m["hits"],
                    hit_rate=m["hit_rate"],
                    brier=m["brier"],
                    ic=m["ic"],
                    calibration=m["calibration"],
                    reliability=reliability(v, ci),
                    computed_at=now,
                    n_effective=m["n_effective"],
                    ci_low=m["ci_low"],
                    ci_high=m["ci_high"],
                    p_value=m["p_value"],
                    q_value=round(q, 5) if q is not None else None,
                    verdict=v,
                    mean_excess=m["mean_excess"],
                    mean_excess_z=m["mean_excess_z"],
                )
            )
    return len(groups)
