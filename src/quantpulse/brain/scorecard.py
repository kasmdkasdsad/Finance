"""How the Brain is doing — measured separately, never blended into one flattering number.

:func:`execution_quality` — from the Brain's real Alpaca paper orders: how many were sent, filled, partly
filled, canceled or rejected, the time to fill, and the slippage of each fill against the price the
decision assumed (positive = paid more / received less).

:func:`scorecard` — the learning dimensions kept apart, each with its sample size and marked *unproven*
until the sample is large enough (``QP_BRAIN_MIN_RELIABILITY_OBSERVATIONS``): prediction accuracy (graded
consensus calls against the benchmark), calibration (hit rate by confidence), decision quality (the
reflections' earned / unlucky / lucky / process-failure mix), luck (the share of outcomes that disagreed
with the decision's quality), execution quality, risk outcomes (stops hit, halts, the deepest drawdown) and
benchmark-relative outcomes (closed positions and trading days), and agent reliability (how many agents
have a measured record, and its verdicts).

:func:`evaluation` — the **60-session evaluation**: the Brain's trading days on the Alpaca paper account
against the benchmark and against the strategy it replaced (its shadow), with risk-adjusted returns,
drawdown, volatility, turnover, win rate, regime performance, sector exposure, calibration, decision
quality, agent reliability and execution quality. Until 60 sessions are recorded it says how many there
are and that the numbers are not to be trusted; at 60 it asks for the architecture to be reviewed. It is a
report for a person, not a target: nothing in the Brain optimises for it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, time
from statistics import median
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import (
    BrainAgentPerformanceRow,
    BrainCycleRow,
    BrainExecutionRow,
    BrainPredictionRow,
    BrainReflectionRow,
    BrainSessionRow,
    BrainThesisRow,
)
from quantpulse.db.session import Database

from .performance import wilson
from .reflection import consensus_calibration

EVALUATION_SESSIONS = 60
ANNUAL = 252


def _r(x: float | None, n: int = 5) -> float | None:
    return None if x is None or not math.isfinite(x) else round(x, n)


def _status(n: int, needed: int) -> str:
    return "measured" if n >= needed else f"unproven ({n} of {needed} observations)"


# ---------------------------------------------------------------------------------------------- execution
async def execution_quality(
    db: Database, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """From the execution ledger (the Brain's real Alpaca paper orders): execution only, never profit."""
    async with db.session() as s:
        q = select(BrainExecutionRow)
        if since is not None:
            q = q.where(BrainExecutionRow.decided_at >= since)
        if until is not None:
            q = q.where(BrainExecutionRow.decided_at < until)
        rows = (await s.scalars(q)).all()
    sent = [r for r in rows if r.alpaca_order_id]
    filled = [r for r in sent if r.filled_qty > 0]
    slips = [r.slippage_bps for r in filled if r.slippage_bps is not None]
    costs = [r.cost_vs_quote_bps for r in filled if r.cost_vs_quote_bps is not None]
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in filled:
        if r.slippage_bps is not None:
            by_type[r.order_type or "?"].append(r.slippage_bps)
    waits = [r.seconds_to_fill for r in filled if r.seconds_to_fill is not None]
    latency = [r.submit_latency_ms for r in sent if r.submit_latency_ms is not None]
    ages = [r.quote_age_s for r in sent if r.quote_age_s is not None]
    spreads = [r.spread_bps for r in sent if r.spread_bps is not None]
    statuses = Counter(r.status for r in rows)
    grades = Counter(r.grade or "unknown" for r in filled)
    return {
        "orders": len(rows),
        "sent": len(sent),
        "filled": len(filled),
        "partially_filled": sum(1 for r in rows if r.partial),
        "canceled_or_expired": statuses["canceled"] + statuses["expired"],
        "rejected": statuses["rejected"] + statuses["submit_failed"],
        "unknown": statuses["submit_unknown"] + statuses["pending_submit"],
        "fill_rate": _r(len(filled) / len(sent), 4) if sent else None,
        "slippage_bps_mean": _r(sum(slips) / len(slips), 2) if slips else None,
        "slippage_bps_median": _r(median(slips), 2) if slips else None,
        "cost_vs_quote_bps_mean": _r(sum(costs) / len(costs), 2) if costs else None,
        "slippage_bps_by_order_type": {k: _r(sum(v) / len(v), 2) for k, v in by_type.items()},
        "grades": dict(grades),
        "fills_measured": len(slips),
        "seconds_to_fill_median": _r(median(waits), 1) if waits else None,
        "submit_latency_ms_median": _r(median(latency), 1) if latency else None,
        "quote_age_s_median": _r(median(ages), 1) if ages else None,
        "spread_bps_median": _r(median(spreads), 2) if spreads else None,
        "note": "slippage against the price the decision assumed, cost against the quote's midpoint as the "
        "order left (positive: worse for us); graded against half the spread — execution only, not profit",
    }


# ---------------------------------------------------------------------------------------------- scorecard
def _returns_stats(rets: list[float]) -> dict[str, Any]:
    n = len(rets)
    if n == 0:
        return {"sessions": 0}
    wealth, peak, deepest = 1.0, 1.0, 0.0
    for r in rets:
        wealth *= 1 + r
        peak = max(peak, wealth)
        deepest = min(deepest, wealth / peak - 1)
    mean = sum(rets) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (n - 1)) if n > 1 else None
    down = [min(r, 0.0) for r in rets]
    dd = math.sqrt(sum(x * x for x in down) / n) if n else None
    return {
        "sessions": n,
        "total_return": _r(wealth - 1),
        "volatility": _r(sd * math.sqrt(ANNUAL)) if sd else None,
        "sharpe": _r(mean / sd * math.sqrt(ANNUAL), 3) if sd else None,
        "sortino": _r(mean / dd * math.sqrt(ANNUAL), 3) if dd else None,
        "max_drawdown": _r(deepest),
        "up_days": _r(sum(1 for r in rets if r > 0) / n, 3),
    }


def _relative(a: list[float], b: list[float]) -> dict[str, Any]:
    if len(a) < 2 or len(a) != len(b):
        return {}
    active = [x - y for x, y in zip(a, b, strict=True)]
    mean = sum(active) / len(active)
    te = math.sqrt(sum((x - mean) ** 2 for x in active) / (len(active) - 1))
    mb = sum(b) / len(b)
    var_b = sum((y - mb) ** 2 for y in b) / (len(b) - 1)
    cov = sum((x - sum(a) / len(a)) * (y - mb) for x, y in zip(a, b, strict=True)) / (len(a) - 1)
    return {
        "excess_return_annual": _r(mean * ANNUAL),
        "tracking_error": _r(te * math.sqrt(ANNUAL)) if te else None,
        "information_ratio": _r(mean / te * math.sqrt(ANNUAL), 3) if te else None,
        "beta": _r(cov / var_b, 3) if var_b else None,
        "days_ahead": _r(sum(1 for x in active if x > 0) / len(active), 3),
    }


async def scorecard(db: Database, settings: Settings, since: datetime | None = None) -> dict[str, Any]:
    need = settings.brain_min_reliability_observations
    async with db.session() as s:
        preds = (
            await s.scalars(
                select(BrainPredictionRow).where(
                    BrainPredictionRow.status == "evaluated", BrainPredictionRow.source_type == "consensus"
                )
            )
        ).all()
        reflections = (
            await s.scalars(select(BrainReflectionRow).where(BrainReflectionRow.subject_type == "decision"))
        ).all()
        closed = (await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "closed"))).all()
        perf = (
            await s.scalars(
                select(BrainAgentPerformanceRow).where(
                    BrainAgentPerformanceRow.window == "all", BrainAgentPerformanceRow.regime == "all"
                )
            )
        ).all()
        sessions = (await s.scalars(select(BrainSessionRow).where(BrainSessionRow.owner == "brain"))).all()
    if since is not None:
        preds = [p for p in preds if p.made_at >= since]
        reflections = [r for r in reflections if r.created_at >= since]
        closed = [t for t in closed if t.closed_at is not None and t.closed_at >= since]
    hits = sum(1 for p in preds if p.hit)
    ci = wilson(hits, len(preds))
    mix = Counter(r.category for r in reflections)
    judged = sum(mix[c] for c in ("earned", "unlucky", "lucky", "process_failure"))
    stops = [t for t in closed if "stop" in (t.exit_reason or "")]
    rel = [
        (t.exit_price / t.entry_price - 1) - (t.benchmark_return or 0.0)
        for t in closed
        if t.exit_price and t.entry_price
    ]
    wins = [t for t in closed if (t.realized_pnl or 0) > 0]
    days = sorted(sessions, key=lambda r: r.day)
    brain_rets = [r.day_return for r in days if r.day_return is not None]
    agents = [p for p in perf if p.agent_id != "consensus"]
    verdicts = Counter(p.verdict or "unproven" for p in agents)
    return {
        "prediction_accuracy": {
            "graded": len(preds),
            "hit_rate": _r(hits / len(preds), 4) if preds else None,
            "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "status": _status(len(preds), need),
            "what": "graded consensus calls: did the subject beat the benchmark over the stated horizon",
        },
        "calibration": {"buckets": await consensus_calibration(db), "status": _status(len(preds), need)},
        "decision_quality": {
            "judged": judged,
            "mix": {k: mix[k] for k in ("earned", "unlucky", "lucky", "process_failure")},
            "sound_decisions": _r((mix["earned"] + mix["unlucky"]) / judged, 3) if judged else None,
            "status": _status(judged, need),
            "what": "was the decision well made (evidence, checks, fit), whatever the outcome",
        },
        "luck": {
            "judged": judged,
            "share": _r((mix["lucky"] + mix["unlucky"]) / judged, 3) if judged else None,
            "what": "outcomes that disagreed with the decision's quality (lucky wins, unlucky losses)",
        },
        "execution_quality": await execution_quality(db, since),
        "risk_outcome": {
            "closed_positions": len(closed),
            "stopped_out": len(stops),
            "max_drawdown": _returns_stats(brain_rets).get("max_drawdown") if brain_rets else None,
            "halts": dict(sum((Counter(r.halts or {}) for r in days), Counter())),
        },
        "benchmark_relative": {
            "closed_positions": len(rel),
            "mean_relative": _r(sum(rel) / len(rel)) if rel else None,
            "beat_benchmark": _r(sum(1 for x in rel if x > 0) / len(rel), 3) if rel else None,
            "win_rate": _r(len(wins) / len(closed), 3) if closed else None,
            "status": _status(len(rel), need),
        },
        "agent_reliability": {
            "agents_with_a_record": len(agents),
            "verdicts": dict(verdicts),
            "established": sorted(
                p.agent_id for p in agents if p.verdict in ("evidence of skill", "evidence of harm")
            ),
            "what": "an agent's weight only moves once its graded calls are enough to tell it from a coin",
        },
    }


# ---------------------------------------------------------------------------------------------- evaluation
async def evaluation(db: Database, settings: Settings) -> dict[str, Any]:
    async with db.session() as s:
        days = (
            await s.scalars(
                select(BrainSessionRow).where(BrainSessionRow.owner == "brain").order_by(BrainSessionRow.day)
            )
        ).all()
        start = datetime.combine(days[0].day, time(0, 0), NEW_YORK) if days else None
        cycles = (
            (
                await s.scalars(
                    select(BrainCycleRow).where(
                        BrainCycleRow.status == "completed", BrainCycleRow.kind == "full"
                    )
                )
            ).all()
            if days
            else []
        )
        theses = (await s.scalars(select(BrainThesisRow))).all()
    n = len(days)
    both = [d for d in days if d.day_return is not None and d.benchmark_return is not None]
    brain = [d.day_return for d in both if d.day_return is not None]
    bench = [d.benchmark_return for d in both if d.benchmark_return is not None]
    shadow_days = [
        d for d in both if ((d.close or {}).get("strategy_shadow") or {}).get("day_return") is not None
    ]
    shadow = [float(d.close["strategy_shadow"]["day_return"]) for d in shadow_days]
    shadow_bench = [d.benchmark_return for d in shadow_days if d.benchmark_return is not None]
    equity = [d.equity_close for d in days if d.equity_close]
    avg_equity = sum(equity) / len(equity) if equity else None
    traded = sum(d.traded_notional for d in days)
    last_shadow = (days[-1].close or {}).get("strategy_shadow") if days else None

    regime_by_day: dict[Any, str] = {}
    for c in cycles:
        if (c.market or {}).get("open") and c.regime.get("label"):
            regime_by_day[c.started_at.astimezone(NEW_YORK).date()] = c.regime["label"]
    regimes: dict[str, list[float]] = defaultdict(list)
    for d in both:
        if d.day_return is not None and d.benchmark_return is not None:
            regimes[regime_by_day.get(d.day, "unknown")].append(d.day_return - d.benchmark_return)
    sectors: dict[str, float] = defaultdict(float)
    for t in theses:
        if t.weight and t.sector:
            sectors[t.sector] += t.weight
    total_w = sum(sectors.values())
    card = await scorecard(db, settings, start)
    return {
        "sessions": n,
        "target_sessions": EVALUATION_SESSIONS,
        "status": (
            f"ready for review: {n} sessions recorded — revisit the architecture with these results"
            if n >= EVALUATION_SESSIONS
            else f"in progress: {n} of {EVALUATION_SESSIONS} sessions; too few to judge — reported, not trusted"
        ),
        "first_day": days[0].day.isoformat() if days else None,
        "last_day": days[-1].day.isoformat() if days else None,
        "brain": {**_returns_stats(brain), **_relative(brain, bench)},
        "benchmark": {"symbol": settings.benchmark_symbol, **_returns_stats(bench)},
        "previous_strategy": {
            **_returns_stats(shadow),
            **_relative(shadow, shadow_bench),
            "turnover": _r(float(last_shadow["turnover"]) / avg_equity, 3)
            if last_shadow and avg_equity
            else None,
            "note": "the strategy the Brain replaced, run as a shadow on its own hypothetical portfolio: same data "
            "and risk limits, modelled fills (the Brain's are Alpaca's)",
        },
        "turnover": _r(traded / avg_equity, 3) if avg_equity else None,
        "regime_performance": {
            k: {"days": len(v), "mean_excess": _r(sum(v) / len(v))} for k, v in sorted(regimes.items())
        },
        "sector_exposure": {k: _r(v / total_w, 3) for k, v in sorted(sectors.items(), key=lambda kv: -kv[1])}
        if total_w
        else {},
        "scorecard": card,
        "caveats": [
            "paper trading only: simulated money; Alpaca's paper fills are not a live market's",
            "no optimisation targets this report: it exists for a person's review",
            "fewer than 60 sessions cannot tell skill from luck; even 60 is a short record",
            "the benchmark return is missing on days its close was not yet available at the close routine",
        ],
    }
