"""The learning report: what the Brain's graded record says — and what it cannot say yet.

Built only from what was recorded and graded (predictions against real closes, decisions' reflections,
ideas taken and rejected, trading days), never from impressions:

* **Agents** — each agent's record (hit rate, 95% interval, independent observations, verdict) and how many
  more observations it needs before it is judged at all; and **against the consensus** on the same calls
  (same cycle and subject, one pair per subject and day): how often it agrees, and when it disagrees, who was
  right — whether the agent adds information the consensus lacks.
* **Calibration** — not just the hit rate: the expected calibration error (how far the implied probability
  of being right, ``0.5 + 0.5 × |score| × confidence``, is from the hit rate, bucket by bucket) and its
  direction (over- or under-confident), per source.
* **Regimes** — every source in bullish, bearish (incl. risk-off), sideways (neutral), high- and
  low-volatility markets and on **event days** (a benchmark move of ≥ 2 daily σ or VIX ≥ 30 when the call was
  made — detected from prices; there is no macro calendar). All cells' p-values are adjusted together for the
  false-discovery rate: with many agents and regimes some cells will look good by chance.
* **Consensus patterns** — does the consensus work better with two or more independent sources, with low
  disagreement, when the devil's advocate did not challenge it, at higher confidence — overall and by regime.
* **Data** — calls made on data that was not usable, against those on usable data; rejections for data.
* **Failure modes** — recurring weaknesses per agent (the failure analysis) and the decision-quality mix.
* **Strategies by regime** — the Brain's trading days against the benchmark and the replaced strategy (the
  shadow), by the day's regime.

Every figure carries its sample and is *unproven* below ``QP_BRAIN_MIN_RELIABILITY_OBSERVATIONS``
independent observations. The report changes nothing: weights move only through the existing reliability
rule, and anything else is a proposal for a person.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, time
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import BrainCycleRow, BrainReflectionRow, BrainSessionRow
from quantpulse.db.session import Database

from . import performance as perf
from .opportunity_outcomes import report as ideas_report
from .performance import Graded, benjamini_hochberg, blocks, metrics, verdict, vol_environment, wilson
from .reflection import failure_analysis

REGIME_BUCKETS = ("bullish", "bearish", "sideways", "high_volatility", "low_volatility", "event")
CAL_BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))
CONTESTED = 0.3  # consensus disagreement at or above this is "contested"


def buckets(regime: str | None, market_vol: float | None, event: bool | None) -> list[str]:
    """The regime buckets a call (or a day) belongs to — it can be in several (a bearish, volatile event day)."""
    out: list[str] = []
    if regime == "bullish":
        out.append("bullish")
    elif regime in ("bearish", "risk_off"):
        out.append("bearish")
    elif regime == "neutral":
        out.append("sideways")
    env = vol_environment(market_vol)
    if regime == "high_volatility" or env == "high":
        out.append("high_volatility")
    if env == "low":
        out.append("low_volatility")
    if event:
        out.append("event")
    return out


def _p_right(r: Graded) -> float:
    return min(max(0.5 + 0.5 * abs(r.score) * r.confidence, 0.0), 1.0)


def calibration(rows: Sequence[Graded], min_n: int) -> dict[str, Any]:
    """Expected calibration error over independent observations (blocks), and its direction."""
    groups = blocks([r for r in rows if r.data_ok])
    n = len(groups)
    if not n:
        return {"n_effective": 0, "status": "unproven", "needs": min_n}
    pts = [(sum(_p_right(r) for r in g) / len(g), sum(1 for r in g if r.hit) / len(g)) for g in groups]
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for lo, hi in CAL_BINS:
        inside = [(p, y) for p, y in pts if lo <= p < hi]
        if not inside:
            continue
        mp = sum(p for p, _ in inside) / len(inside)
        my = sum(y for _, y in inside) / len(inside)
        ece += len(inside) / n * abs(my - mp)
        bins.append({"implied": f"{lo:.1f}–{min(hi, 1.0):.1f}", "n": len(inside), "implied_mean": round(mp, 3),
                     "hit_rate": round(my, 3)})  # fmt: skip
    bias = sum(p for p, _ in pts) / n - sum(y for _, y in pts) / n
    status = "unproven" if n < min_n else (
        "well calibrated" if ece <= 0.05 else "overconfident" if bias > 0 else "underconfident"
    )  # fmt: skip
    return {
        "n_effective": n,
        "ece": round(ece, 4),
        "bias": round(bias, 4),
        "bins": bins,
        "status": status,
        "needs": max(min_n - n, 0),
    }


def versus_consensus(agent: Sequence[Graded], consensus: Sequence[Graded], min_n: int) -> dict[str, Any]:
    """The agent and the consensus on the same calls (same cycle and subject), one pair per subject and day."""
    by_call: dict[tuple[int, str], Graded] = {}
    for c in consensus:
        if c.cycle_id is not None:
            by_call.setdefault((c.cycle_id, c.subject or ""), c)
    pairs: dict[tuple[str, Any], tuple[Graded, Graded]] = {}
    for a in agent:
        if a.cycle_id is None or not a.data_ok:
            continue
        match = by_call.get((a.cycle_id, a.subject or ""))
        if match is None:
            continue
        pairs.setdefault((a.subject or "", a.made_at.astimezone(NEW_YORK).date()), (a, match))
    n = len(pairs)
    agree = [(a, c) for a, c in pairs.values() if a.direction == c.direction]
    differ = [(a, c) for a, c in pairs.values() if a.direction != c.direction]
    agent_right = sum(1 for a, c in differ if a.hit and not c.hit)
    decisive = sum(1 for a, c in differ if a.hit != c.hit)
    ci = wilson(agent_right, decisive) if decisive else None
    if decisive < min_n:
        status = "unproven"
    elif ci is not None and ci[0] > 0.5:
        status = "adds information: right more often than the consensus when they disagree"
    elif ci is not None and ci[1] < 0.5:
        status = "the consensus is right more often when they disagree"
    else:
        status = "no evidence either way"
    return {
        "pairs": n,
        "agreement": round(len(agree) / n, 3) if n else None,
        "disagreements": len(differ),
        "agent_right_when_disagreeing": round(agent_right / decisive, 3) if decisive else None,
        "ci95": [round(ci[0], 3), round(ci[1], 3)] if ci else None,
        "agent_hit_rate": round(sum(1 for a, _ in pairs.values() if a.hit) / n, 3) if n else None,
        "consensus_hit_rate": round(sum(1 for _, c in pairs.values() if c.hit) / n, 3) if n else None,
        "status": status,
        "needs": max(min_n - decisive, 0),
    }


def _cell(rows: Sequence[Graded], min_n: int) -> dict[str, Any]:
    m = metrics(rows, min_n)
    return {
        k: m[k]
        for k in ("n", "n_effective", "hit_rate", "ci_low", "ci_high", "p_value", "mean_excess", "brier")
    }


def regime_matrix(by_source: dict[str, list[Graded]], min_n: int) -> dict[str, dict[str, Any]]:
    """Every source in every regime bucket, with the p-values of all cells adjusted together."""
    cells: list[tuple[str, str, dict[str, Any]]] = []
    for source, rows in sorted(by_source.items()):
        split: dict[str, list[Graded]] = defaultdict(list)
        for r in rows:
            for b in buckets(r.regime, r.market_vol, r.market_event):
                split[b].append(r)
        for b in REGIME_BUCKETS:
            if split.get(b):
                cells.append((source, b, _cell(split[b], min_n)))
    qs = benjamini_hochberg([c["p_value"] for _, _, c in cells])
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (source, b, c), q in zip(cells, qs, strict=True):
        v = verdict(c["n_effective"], c["hit_rate"], q, min_n)
        out[source][b] = {**c, "q_value": round(q, 5) if q is not None else None, "verdict": v,
                          "needs": max(min_n - c["n_effective"], 0)}  # fmt: skip
    return dict(out)


def consensus_patterns(rows: Sequence[Graded], min_n: int) -> dict[str, Any]:
    """Which kinds of consensus call work: by independent sources, disagreement, challenge and confidence."""
    tests: dict[str, Any] = {
        "two_or_more_sources": lambda r: (r.context.get("independent_sources") or 0) >= 2,
        "one_source": lambda r: (r.context.get("independent_sources") or 0) < 2,
        "low_disagreement": lambda r: (r.context.get("disagreement") or 0) < CONTESTED,
        "contested": lambda r: (r.context.get("disagreement") or 0) >= CONTESTED,
        "challenged_by_devils_advocate": lambda r: r.context.get("debate") == "challenged",
        "not_challenged": lambda r: r.context.get("debate") not in (None, "challenged"),
        "confidence_0.6_plus": lambda r: r.confidence >= 0.6,
        "confidence_below_0.6": lambda r: r.confidence < 0.6,
    }
    out: dict[str, Any] = {}
    for name, keep in tests.items():
        mine = [r for r in rows if keep(r)]
        if not mine:
            continue
        overall = _cell(mine, min_n)
        by_regime: dict[str, Any] = {}
        for b in REGIME_BUCKETS:
            sub = [r for r in mine if b in buckets(r.regime, r.market_vol, r.market_event)]
            if sub:
                c = _cell(sub, min_n)
                by_regime[b] = {"n_effective": c["n_effective"], "hit_rate": c["hit_rate"],
                                "status": "unproven" if c["n_effective"] < min_n else "measured"}  # fmt: skip
        out[name] = {**overall, "status": "unproven" if overall["n_effective"] < min_n else "measured",
                     "by_regime": by_regime}  # fmt: skip
    return out


def data_mistakes(rows: Sequence[Graded], min_n: int) -> dict[str, Any]:
    """Calls made on data that was not usable, against those on usable data (the verdicts leave the former
    out — this shows what they would have looked like)."""
    bad = [r for r in rows if not r.data_ok]
    good = [r for r in rows if r.data_ok]
    by_status: dict[str, list[Graded]] = defaultdict(list)
    for r in bad:
        by_status[str(r.context.get("data_status") or r.context.get("data_state") or "unknown")].append(r)

    def rate(xs: Sequence[Graded]) -> dict[str, Any]:
        groups = blocks(xs)
        k = sum(sum(1 for r in g if r.hit) / len(g) for g in groups)
        ci = wilson(k, len(groups)) if groups else None
        return {"calls": len(xs), "n_effective": len(groups), "hit_rate": round(k / len(groups), 3) if groups else None,
                "ci95": [round(ci[0], 3), round(ci[1], 3)] if ci else None}  # fmt: skip

    return {
        "on_unusable_data": rate(bad),
        "on_usable_data": rate(good),
        "by_data_status": {k: rate(v) for k, v in sorted(by_status.items())},
        "status": "unproven" if len(blocks(bad)) < min_n else "measured",
        "note": "calls on unusable data are graded but never counted in a verdict or a weight",
    }


async def strategies_by_regime(db: Database, settings: Settings) -> dict[str, Any]:
    """The Brain's trading days, the benchmark and the replaced strategy (shadow), by the day's regime bucket."""
    async with db.session() as s:
        days = (await s.scalars(select(BrainSessionRow).where(BrainSessionRow.owner == "brain"))).all()
        if not days:
            return {}
        start = min(d.day for d in days)
        cycles = (
            await s.scalars(
                select(BrainCycleRow).where(
                    BrainCycleRow.status == "completed",
                    BrainCycleRow.started_at >= datetime.combine(start, time(0, 0), NEW_YORK),
                )
            )
        ).all()
    day_ctx: dict[Any, dict[str, Any]] = {}
    for c in sorted(cycles, key=lambda c: c.started_at):
        market = c.market or {}
        if not market.get("open") or not (c.regime or {}).get("label"):
            continue
        stats = market.get("stats") or {}
        z, vix = stats.get("benchmark_move_z"), market.get("vix")
        event = (z is not None and abs(z) >= 2.0) or (vix is not None and vix >= 30.0)
        prev = day_ctx.get(c.started_at.astimezone(NEW_YORK).date(), {})
        day_ctx[c.started_at.astimezone(NEW_YORK).date()] = {
            "regime": c.regime["label"],
            "vol": stats.get("benchmark_rv21"),
            "event": bool(prev.get("event")) or event,
        }
    series: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in days:
        info = day_ctx.get(d.day)
        if info is None or d.day_return is None or d.benchmark_return is None:
            continue
        shadow = ((d.close or {}).get("strategy_shadow") or {}).get("day_return")
        for b in buckets(info["regime"], info["vol"], info["event"]):
            series[b]["brain"].append(d.day_return)
            series[b]["benchmark"].append(d.benchmark_return)
            if shadow is not None:
                series[b]["shadow"].append(float(shadow))
    need = 20  # trading days before a regime's comparison says anything
    out: dict[str, Any] = {}
    for b in REGIME_BUCKETS:
        if b not in series:
            continue
        v = series[b]
        mean = {k: round(sum(x) / len(x), 6) for k, x in v.items() if x}
        out[b] = {"days": len(v["brain"]), "mean_daily_return": mean,
                  "brain_minus_benchmark": round(mean["brain"] - mean["benchmark"], 6),
                  "brain_minus_shadow": round(mean["brain"] - mean["shadow"], 6) if "shadow" in mean else None,
                  "status": "unproven" if len(v["brain"]) < need else "measured"}  # fmt: skip
    return out


async def build(db: Database, settings: Settings) -> dict[str, Any]:
    min_n = settings.brain_min_reliability_observations
    rows = await perf.graded(db)
    by_source: dict[str, list[Graded]] = defaultdict(list)
    for r in rows:
        by_source[r.source].append(r)
    consensus = by_source.get("consensus", [])
    agents: dict[str, Any] = {}
    for source, mine in sorted(by_source.items()):
        if source == "consensus":
            continue
        m = metrics(mine, min_n)
        agents[source] = {
            "record": {k: m[k] for k in ("n", "n_effective", "hit_rate", "ci_low", "ci_high", "verdict", "brier", "ic", "mean_excess")},
            "needs": max(min_n - m["n_effective"], 0),
            "calibration": calibration(mine, min_n),
            "versus_consensus": versus_consensus(mine, consensus, min_n),
        }  # fmt: skip
    cm = metrics(consensus, min_n) if consensus else None
    async with db.session() as s:
        cats = (await s.scalars(select(BrainReflectionRow.category))).all()
    mix: dict[str, int] = defaultdict(int)
    for c in cats:
        mix[c] += 1
    ideas = await ideas_report(db, min_n)
    return {
        "min_observations": min_n,
        "graded_calls": len(rows),
        "consensus": {
            "record": {
                k: cm[k]
                for k in (
                    "n",
                    "n_effective",
                    "hit_rate",
                    "ci_low",
                    "ci_high",
                    "verdict",
                    "brier",
                    "ic",
                    "mean_excess",
                )
            }
            if cm
            else None,
            "calibration": calibration(consensus, min_n),
            "patterns": consensus_patterns(consensus, min_n),
        },
        "agents": agents,
        "regimes": regime_matrix(by_source, min_n),
        "data": data_mistakes(rows, min_n),
        "failure_modes": {
            "agents": failure_analysis(rows, min_n),
            "decision_mix": dict(mix),
            "note": "a weakness is named only when its 95% interval lies below a coin flip",
        },
        "rejections": {
            k: {"status": v["status"], "decisive": v["decisive"], "meaning": v["meaning"]}
            for k, v in ideas["by_reason"].items()
        },
        "strategies_by_regime": await strategies_by_regime(db, settings),
        "caveats": [
            "regime cells are exploratory: their p-values are adjusted for the number of cells, and most are "
            "unproven for a long time",
            "event days are detected from prices (a ≥ 2σ benchmark move or VIX ≥ 30), not from a macro calendar",
            "nothing in this report changes a weight, a threshold or a limit",
        ],
    }
