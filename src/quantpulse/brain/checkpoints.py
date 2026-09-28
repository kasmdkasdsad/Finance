"""The 20 / 40 / 60-session evaluation: the long-term paper experiment, measured at fixed checkpoints.

The Brain's trading days on the Alpaca paper account (``brain_sessions``) are evaluated over fixed windows —
its first 20, first 40 and first 60 sessions (comparable over time: a checkpoint, once reached, does not
move) — plus the most recent 20 sessions and everything so far. Each window shows, side by side and never
blended into one number:

* **the Brain, the benchmark and the replaced strategy (shadow)** — total return, volatility, Sharpe, Sortino,
  maximum drawdown, up days; excess return, tracking error, information ratio and beta against the
  benchmark, and the same against the shadow on the days it has a return;
* **P&L** — realised (positions closed in the window), unrealised (open positions — known only for the
  current window: the past is not re-marked) and the change in equity;
* **turnover** — traded notional against the average equity;
* **execution quality** — the ledger's fills in the window (slippage, cost against the quote, grades);
* **prediction calibration** — the consensus calls made in the window and graded since: hit rate, 95%
  interval, calibration error and direction;
* **decision quality** — the reflections on the window's decisions: earned / unlucky / lucky /
  process failure, and the share of sound decisions;
* **risk behaviour** — halts by reason, data-blocked cycles, positions stopped out, days at or beyond the
  daily loss limit, exposure;
* **agent reliability** — each agent's calls made in the window: verdicts and who has a measured record.

**No window declares the Brain successful or unsuccessful.** Each carries a statistical statement about the
mean daily excess return (its t-statistic and a 95% interval) and says plainly what the sample can and
cannot support: at 20 or 40 sessions nothing can be concluded; at 60 the record is ready for a person's
review — still a short record, and paper trading only.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, time, timedelta
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import BrainDecisionRow, BrainReflectionRow, BrainSessionRow, BrainThesisRow
from quantpulse.db.session import Database

from . import performance as perf
from .learning_report import calibration
from .scorecard import ANNUAL, _r, _relative, _returns_stats, execution_quality
from .trade_lessons import ended_by

CHECKPOINTS = (20, 40, 60)
ROLLING = 20
T_NOTABLE = 2.0


def _bounds(days: Sequence[BrainSessionRow]) -> tuple[datetime, datetime]:
    start = datetime.combine(days[0].day, time(0, 0), NEW_YORK)
    end = datetime.combine(days[-1].day + timedelta(days=1), time(0, 0), NEW_YORK)
    return start, end


def significance(brain: Sequence[float], bench: Sequence[float]) -> dict[str, Any]:
    """The mean daily excess return over the benchmark: its t-statistic and 95% interval (annualised), and
    what that sample can and cannot say."""
    active = [a - b for a, b in zip(brain, bench, strict=True)]
    n = len(active)
    if n < 2:
        return {"days": n, "statement": "too few sessions for any statistic"}
    mean = sum(active) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in active) / (n - 1))
    se = sd / math.sqrt(n) if sd > 0 else None
    t = mean / se if se else None
    ci = (mean - 1.96 * se, mean + 1.96 * se) if se else None
    if t is None or abs(t) < T_NOTABLE:
        finding = "indistinguishable from the benchmark"
    else:
        finding = "ahead of the benchmark" if t > 0 else "behind the benchmark"
        finding = f"notably {finding} (|t| ≥ {T_NOTABLE:.0f})"
    sample = (
        f"{n} sessions cannot establish skill or its absence"
        if n < max(CHECKPOINTS)
        else f"{n} sessions: ready for a person's review — still a short record, paper trading only"
    )
    return {
        "days": n,
        "mean_daily_excess": _r(mean, 6),
        "excess_annual": _r(mean * ANNUAL),
        "ci95_annual": [_r(ci[0] * ANNUAL), _r(ci[1] * ANNUAL)] if ci else None,
        "t": _r(t, 2) if t is not None else None,
        "finding": finding,
        "statement": f"{finding}; {sample}",
        "verdict": "none",  # never declared: a person reviews the record
    }


async def window(
    db: Database, settings: Settings, days: Sequence[BrainSessionRow], *, current: bool, label: str
) -> dict[str, Any]:
    start, end = _bounds(days)
    both = [d for d in days if d.day_return is not None and d.benchmark_return is not None]
    brain = [float(d.day_return) for d in both if d.day_return is not None]
    bench = [float(d.benchmark_return) for d in both if d.benchmark_return is not None]
    shadow_days = [
        d for d in both if ((d.close or {}).get("strategy_shadow") or {}).get("day_return") is not None
    ]
    shadow = [float(d.close["strategy_shadow"]["day_return"]) for d in shadow_days]
    brain_on_shadow = [float(d.day_return) for d in shadow_days if d.day_return is not None]
    async with db.session() as s:
        closed = (
            await s.scalars(
                select(BrainThesisRow).where(
                    BrainThesisRow.closed_at >= start, BrainThesisRow.closed_at < end
                )
            )
        ).all()
        open_now = (
            (await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "open"))).all()
            if current
            else []
        )
        decision_ids = (
            await s.scalars(
                select(BrainDecisionRow.id).where(
                    BrainDecisionRow.created_at >= start, BrainDecisionRow.created_at < end
                )
            )
        ).all()
        reflections = (
            (
                await s.scalars(
                    select(BrainReflectionRow).where(
                        BrainReflectionRow.subject_type == "decision",
                        BrainReflectionRow.subject_id.in_(decision_ids),
                    )
                )
            ).all()
            if decision_ids
            else []
        )
    min_n = settings.brain_min_reliability_observations
    graded = [g for g in await perf.graded(db) if start <= g.made_at < end]
    consensus = [g for g in graded if g.source == "consensus"]
    cm = perf.metrics(consensus, min_n) if consensus else None
    by_agent: dict[str, list[perf.Graded]] = defaultdict(list)
    for g in graded:
        if g.source != "consensus":
            by_agent[g.source].append(g)
    agents = {a: perf.metrics(rows, min_n) for a, rows in sorted(by_agent.items())}
    mix = Counter(r.category for r in reflections)
    judged = sum(mix[k] for k in ("earned", "unlucky", "lucky", "process_failure"))
    equity = [d.equity_close for d in days if d.equity_close]
    avg_equity = sum(equity) / len(equity) if equity else None
    traded = sum(d.traded_notional for d in days)
    halts: Counter[str] = Counter()
    for d in days:
        halts.update(d.halts or {})
    exposure = [d.exposure for d in days if d.exposure is not None]
    stopped = [t for t in closed if ended_by(t.exit_reason) == "stop"]
    realised = [t.realized_pnl for t in closed if t.realized_pnl is not None]
    limit = settings.trading_max_daily_loss_pct
    eq_start, eq_end = days[0].equity_open or days[0].equity_close, days[-1].equity_close
    return {
        "label": label,
        "sessions": len(days),
        "first_day": days[0].day.isoformat(),
        "last_day": days[-1].day.isoformat(),
        "brain": {**_returns_stats(brain), **_relative(brain, bench)},
        "benchmark": {"symbol": settings.benchmark_symbol, **_returns_stats(bench)},
        "shadow": {
            **_returns_stats(shadow),
            "brain_vs_shadow": _relative(brain_on_shadow, shadow),
            "note": "the replaced strategy on its own hypothetical portfolio; modelled fills (the Brain's are Alpaca's)",
        },
        "significance": significance(brain, bench),
        "pnl": {
            "realised": _r(sum(realised), 2) if realised else 0.0,
            "positions_closed": len(closed),
            "unrealised": _r(sum(t.unrealized_pnl or 0.0 for t in open_now), 2) if current else None,
            "unrealised_note": None if current else "not recorded for a past window (positions are not re-marked)",
            "equity_change": _r(eq_end - eq_start, 2) if eq_end and eq_start else None,
        },
        "turnover": {"total": _r(traded / avg_equity, 3) if avg_equity else None,
                     "per_session": _r(traded / avg_equity / len(days), 4) if avg_equity else None},
        "execution": await execution_quality(db, start, end),
        "calibration": {
            "consensus_calls": len(consensus),
            "hit_rate": cm["hit_rate"] if cm else None,
            "ci95": [cm["ci_low"], cm["ci_high"]] if cm and cm["ci_low"] is not None else None,
            "n_effective": cm["n_effective"] if cm else 0,
            **{k: v for k, v in calibration(consensus, min_n).items() if k in ("ece", "bias", "status", "needs")},
        },
        "decision_quality": {
            "judged": judged,
            "mix": {k: mix.get(k, 0) for k in ("earned", "unlucky", "lucky", "process_failure")},
            "sound_share": _r((mix["earned"] + mix["unlucky"]) / judged, 3) if judged else None,
            "status": "unproven" if judged < min_n else "measured",
        },
        "risk": {
            "halts": dict(halts),
            "data_blocked_cycles": sum(d.data_blocked_cycles for d in days),
            "stopped_out": len(stopped),
            "days_at_daily_loss_limit": sum(1 for r in brain if r <= -limit),
            "max_exposure": _r(max(exposure), 3) if exposure else None,
            "mean_exposure": _r(sum(exposure) / len(exposure), 3) if exposure else None,
            "max_drawdown": _returns_stats(brain).get("max_drawdown") if brain else None,
        },
        "agent_reliability": {
            "agents": {a: {"n_effective": m["n_effective"], "hit_rate": m["hit_rate"], "verdict": m["verdict"]}
                       for a, m in agents.items()},
            "measured": sorted(a for a, m in agents.items() if m["verdict"] != "unproven"),
            "note": "verdicts need independent graded calls made in this window; most stay unproven for weeks",
        },
    }  # fmt: skip


async def build(db: Database, settings: Settings) -> dict[str, Any]:
    async with db.session() as s:
        days = list(
            (
                await s.scalars(
                    select(BrainSessionRow)
                    .where(BrainSessionRow.owner == "brain")
                    .order_by(BrainSessionRow.day)
                )
            ).all()
        )
    n = len(days)
    checkpoints: dict[str, Any] = {}
    for size in CHECKPOINTS:
        if n >= size:
            checkpoints[str(size)] = {
                "status": "reached"
                + (
                    " — ready for a person's review"
                    if size == max(CHECKPOINTS)
                    else " — reported, not judged"
                ),
                **await window(db, settings, days[:size], current=n == size, label=f"first {size} sessions"),
            }
        else:
            checkpoints[str(size)] = {
                "status": f"not reached: {n} of {size} sessions",
                "progress": round(n / size, 3),
            }
    return {
        "sessions": n,
        "checkpoints": checkpoints,
        "rolling": await window(db, settings, days[-ROLLING:], current=True, label=f"last {ROLLING} sessions")
        if n >= ROLLING
        else None,
        "so_far": await window(db, settings, days, current=True, label="every session so far") if days else None,
        "principle": "no window declares the Brain successful or unsuccessful: the figures are reported with "
        "their samples for a person to review; paper trading only",
    }  # fmt: skip
