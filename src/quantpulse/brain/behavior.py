"""Watching the Brain for pathological behaviour — from what it recorded, over a recent window.

A system that trades by itself can go wrong in ways no single decision shows. Each check below looks at the
record (the execution ledger, the positions' theses, trading days, consensus, opinions, ideas considered)
and reports a finding with its evidence and sample:

=======================  ==================================================================================
round_trips              the same symbol bought and sold again and again (or sold within two sessions of
                         being bought, not at a stop)
turnover                 traded notional against equity per day, and against the replaced strategy's (shadow)
concentration            the largest position and the effective number of positions; a sector above its limit
correlated_positions     holdings that move together (the portfolio agent's average pairwise correlation)
repeated_thesis_losses   the same symbol, or the same set of supporting agents, losing again and again
ignored_opportunities    a kind of idea detected often and never taken — and what the ignored ideas did
agent_herding            agents that nearly always agree (unanimity), or whose scores move in lockstep
consensus_instability    the consensus on a symbol flipping direction within a day
losing_streak            after a run of losing positions: does the Brain trade more (chasing) or freeze?
=======================  ==================================================================================

Severity is ``info`` (worth knowing), ``warning`` (worth a look) or ``alert`` (likely a defect in behaviour).
Findings with too little data say so. Nothing here changes a limit, a threshold or a weight, and nothing
stops trading: the risk engine and the kill switches already do that. Findings feed the daily and weekly
reviews and, where they point at a rule the Brain may change, an improvement proposal for a person.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

import numpy as np
from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.market_calendar import NEW_YORK, sessions_between
from quantpulse.db.models import (
    BrainConsensusRow,
    BrainCycleRow,
    BrainExecutionRow,
    BrainOpinionRow,
    BrainOpportunityOutcomeRow,
    BrainSessionRow,
    BrainThesisRow,
)
from quantpulse.db.session import Database

from .agents.portfolio import SECTOR_LIMIT

WINDOW_DAYS = 30
QUICK_SESSIONS = 2
ROUND_TRIPS_ALERT = 2  # more than this many round trips on one symbol in the window
TURNOVER_WARN = 0.5  # traded notional above half the equity per day on average
TURNOVER_VS_SHADOW = 2.0
CORRELATION_WARN = 0.7
SAME_LOSSES = 2  # losing positions on one symbol in the window
SAME_AGENTS_LOSSES = 3  # losing positions backed by the same agents in a row
IGNORED_MIN = 20  # ideas of a kind before "never taken" is worth saying
UNANIMITY_WARN = 0.8
LOCKSTEP = 0.9  # pairwise score correlation across the same calls
MIN_CALLS = 50
FLIPS_WARN = 2  # direction flips on one symbol in one day
STREAK = 3  # consecutive losing positions
MIN_SESSIONS_SIDE = 5


def _finding(code: str, severity: str, finding: str, sample: int, **evidence: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "finding": finding, "sample": sample, "evidence": evidence}


def round_trips(
    executions: Sequence[BrainExecutionRow], theses: Sequence[BrainThesisRow]
) -> list[dict[str, Any]]:
    filled = [e for e in executions if e.filled_qty > 0 and e.decided_at is not None]
    by_symbol: dict[str, list[BrainExecutionRow]] = defaultdict(list)
    for e in sorted(filled, key=lambda e: e.decided_at or datetime.min):
        by_symbol[e.symbol].append(e)
    out: list[dict[str, Any]] = []
    trips = {}
    for sym, items in by_symbol.items():
        flips = sum(1 for a, b in pairwise(items) if a.side != b.side)
        trips[sym] = math.ceil(flips / 2)  # buy→sell is one round trip; buy→sell→buy→sell two
    worst = {s: n for s, n in trips.items() if n > ROUND_TRIPS_ALERT}
    quick = [
        t.symbol
        for t in theses
        if t.origin == "brain" and t.closed_at is not None
        and sessions_between(t.opened_at, t.closed_at) <= QUICK_SESSIONS
        and "stop" not in (t.exit_reason or "").lower()
    ]  # fmt: skip
    if worst:
        out.append(_finding("round_trips", "alert",
                            f"repeated buying and selling: {', '.join(f'{s} ({n})' for s, n in sorted(worst.items()))} "
                            "round trips in the window", len(filled), round_trips=worst))  # fmt: skip
    if quick:
        share = len(quick) / max(sum(1 for t in theses if t.origin == "brain" and t.closed_at is not None), 1)
        out.append(_finding("round_trips", "warning" if len(quick) >= 3 else "info",
                            f"{len(quick)} position(s) closed within {QUICK_SESSIONS} sessions of opening, not at a "
                            f"stop ({share:.0%} of closed positions)", len(quick), symbols=sorted(set(quick))))  # fmt: skip
    return out


def turnover(days: Sequence[BrainSessionRow]) -> list[dict[str, Any]]:
    eq = [d.equity_close for d in days if d.equity_close]
    if not days or not eq:
        return []
    avg_eq = sum(eq) / len(eq)
    daily = sum(d.traded_notional for d in days) / len(days) / avg_eq
    shadow_last = next(((d.close or {}).get("strategy_shadow") or {} for d in reversed(days) if (d.close or {}).get("strategy_shadow")), {})  # fmt: skip
    shadow_turnover = shadow_last.get("turnover")
    shadow_daily = float(shadow_turnover) / len(days) / avg_eq if shadow_turnover else None
    severity = "warning" if daily > TURNOVER_WARN else "info"
    note = f"traded {daily:.0%} of equity per session on average over {len(days)} session(s)"
    if shadow_daily:
        ratio = daily / shadow_daily if shadow_daily > 0 else None
        note += f" (the replaced strategy's shadow: {shadow_daily:.0%})"
        if ratio is not None and ratio > TURNOVER_VS_SHADOW and len(days) >= MIN_SESSIONS_SIDE:
            severity = "warning"
    return [_finding("turnover", severity, note, len(days), daily_turnover=round(daily, 4),
                     shadow_daily_turnover=round(shadow_daily, 4) if shadow_daily else None)]  # fmt: skip


def concentration(open_theses: Sequence[BrainThesisRow], max_position_pct: float) -> list[dict[str, Any]]:
    weights = {t.symbol: t.weight or 0.0 for t in open_theses if (t.weight or 0) > 0}
    if len(weights) < 1:
        return []
    total = sum(weights.values())
    hhi = sum((w / total) ** 2 for w in weights.values()) if total else None
    top, top_w = max(weights.items(), key=lambda kv: kv[1])
    sectors: dict[str, float] = defaultdict(float)
    for t in open_theses:
        if t.sector and (t.weight or 0) > 0:
            sectors[t.sector] += t.weight or 0.0
    heavy = {s: round(w, 3) for s, w in sectors.items() if s not in ("unknown", "ETF") and w > SECTOR_LIMIT}
    severity = (
        "warning"
        if heavy or top_w > 0.8 * max_position_pct or (hhi and len(weights) > 1 and hhi > 0.5)
        else "info"
    )
    return [_finding("concentration", severity,
                     f"{len(weights)} position(s); largest {top} at {top_w:.1%} of equity (limit {max_position_pct:.0%}); "
                     f"effective number of positions {1 / hhi:.1f}" if hhi else "no weights", len(weights),
                     top=top, top_weight=round(top_w, 4), hhi=round(hhi, 4) if hhi else None,
                     heavy_sectors=heavy)]  # fmt: skip


def correlated_positions(last_portfolio: dict[str, Any] | None) -> list[dict[str, Any]]:
    cons = (last_portfolio or {}).get("constraints") or {}
    corr, n = cons.get("avg_correlation"), len((last_portfolio or {}).get("positions") or {})
    if corr is None or n < 2:
        return []
    severity = "warning" if corr > CORRELATION_WARN and n >= 3 else "info"
    return [_finding("correlated_positions", severity,
                     f"average pairwise correlation of the {n} holdings: {corr:.2f} (63 days)", n,
                     avg_correlation=round(corr, 3))]  # fmt: skip


def repeated_losses(closed: Sequence[BrainThesisRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    mine = sorted(
        [t for t in closed if t.origin == "brain" and t.realized_pnl is not None],
        key=lambda t: t.closed_at or t.opened_at,
    )
    losses: dict[str, int] = defaultdict(int)
    for t in mine:
        if (t.realized_pnl or 0) < 0:
            losses[t.symbol] += 1
    again = {s: n for s, n in losses.items() if n >= SAME_LOSSES}
    if again:
        out.append(_finding("repeated_thesis_losses", "warning",
                            "the same symbol lost again: " + ", ".join(f"{s} ({n})" for s, n in sorted(again.items())),
                            len(mine), symbols=again))  # fmt: skip
    run: list[BrainThesisRow] = []
    worst: tuple[str, int] | None = None
    for t in mine:
        key = ",".join(sorted(t.supporting or [])) or "none"
        if (t.realized_pnl or 0) < 0 and run and ",".join(sorted(run[-1].supporting or [])) == key:
            run.append(t)
        elif (t.realized_pnl or 0) < 0:
            run = [t]
        else:
            run = []
        if len(run) >= SAME_AGENTS_LOSSES and (worst is None or len(run) > worst[1]):
            worst = (key, len(run))
    if worst:
        out.append(_finding("repeated_thesis_losses", "alert",
                            f"{worst[1]} losing positions in a row backed by the same agents ({worst[0]})", len(mine),
                            agents=worst[0], run=worst[1]))  # fmt: skip
    return out


def ignored_opportunities(ideas: Sequence[BrainOpportunityOutcomeRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_kind: dict[str, list[BrainOpportunityOutcomeRow]] = defaultdict(list)
    for i in ideas:
        if i.market_open and i.reason not in ("already_held", "nothing_to_sell"):
            by_kind[i.kind].append(i)
    for kind, items in sorted(by_kind.items()):
        if len(items) < IGNORED_MIN or any(i.taken for i in items):
            continue
        graded = [i for i in items if i.verdict in ("missed", "avoided")]
        missed = sum(1 for i in graded if i.verdict == "missed")
        reasons: dict[str, int] = defaultdict(int)
        for i in items:
            reasons[i.reason] += 1
        main = max(reasons.items(), key=lambda kv: kv[1])
        severity = "warning" if len(graded) >= IGNORED_MIN and missed / len(graded) > 0.6 else "info"
        out.append(_finding("ignored_opportunities", severity,
                            f"{kind}: {len(items)} ideas, none taken (mostly: {main[0]}); of {len(graded)} graded, "
                            f"{missed} would have worked", len(items), kind=kind, reasons=dict(reasons),
                            graded=len(graded), missed=missed))  # fmt: skip
    return out


def herding(
    consensus: Sequence[BrainConsensusRow], opinions: Sequence[BrainOpinionRow]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    voted = [c for c in consensus if c.supporting + c.opposing >= 3 and not c.unknown]
    if len(voted) >= MIN_CALLS:
        unanimous = sum(1 for c in voted if c.opposing == 0 or c.supporting == 0)
        share = unanimous / len(voted)
        out.append(_finding("agent_herding", "warning" if share > UNANIMITY_WARN else "info",
                            f"{share:.0%} of {len(voted)} consensus views had no dissenting agent", len(voted),
                            unanimity=round(share, 3)))  # fmt: skip
    scores: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for o in opinions:
        if o.stance != "abstain":
            scores[(o.cycle_id, o.subject)][o.agent_id] = o.score
    agents = sorted({a for d in scores.values() for a in d})
    lockstep: list[dict[str, Any]] = []
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            pairs = [(d[a], d[b]) for d in scores.values() if a in d and b in d]
            if len(pairs) < MIN_CALLS:
                continue
            x, y = np.array(pairs).T
            if x.std() == 0 or y.std() == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if r >= LOCKSTEP:
                lockstep.append({"agents": [a, b], "correlation": round(r, 3), "calls": len(pairs)})
    if lockstep:
        out.append(_finding("agent_herding", "warning",
                            "agents whose scores move in lockstep (they may be one idea counted twice): "
                            + "; ".join(f"{p['agents'][0]}/{p['agents'][1]} {p['correlation']:.2f}" for p in lockstep[:5]),
                            len(lockstep), pairs=lockstep))  # fmt: skip
    return out


def instability(consensus: Sequence[BrainConsensusRow]) -> list[dict[str, Any]]:
    by_day: dict[tuple[str, Any], list[BrainConsensusRow]] = defaultdict(list)
    for c in sorted(consensus, key=lambda c: (c.created_at, c.id)):
        if c.subject.startswith("@"):
            continue
        by_day[(c.subject, c.created_at.astimezone(NEW_YORK).date())].append(c)
    flips: dict[str, int] = {}
    days_seen = 0
    for (subject, day), items in by_day.items():
        stances = [c.stance for c in items if not c.unknown and c.stance in ("bullish", "bearish")]
        if len(items) < 2:
            continue
        days_seen += 1
        n = sum(1 for a, b in pairwise(stances) if a != b)
        if n >= FLIPS_WARN:
            flips[f"{subject} {day}"] = n
    if not days_seen:
        return []
    severity = "warning" if flips else "info"
    return [_finding("consensus_instability", severity,
                     f"{len(flips)} symbol-day(s) where the consensus flipped direction {FLIPS_WARN}+ times"
                     if flips else f"no symbol flipped direction {FLIPS_WARN}+ times in a day", days_seen,
                     flips=dict(list(flips.items())[:20]))]  # fmt: skip


def losing_streak(
    closed: Sequence[BrainThesisRow], executions: Sequence[BrainExecutionRow]
) -> list[dict[str, Any]]:
    mine = sorted([t for t in closed if t.origin == "brain" and t.realized_pnl is not None and t.closed_at],
                  key=lambda t: t.closed_at or t.opened_at)  # fmt: skip
    streak_end: datetime | None = None
    run = 0
    for t in mine:
        run = run + 1 if (t.realized_pnl or 0) < 0 else 0
        if run >= STREAK:
            streak_end = t.closed_at
    if streak_end is None:
        return []
    buys = [e for e in executions if e.side == "buy" and e.decided_at is not None]
    before = [e for e in buys if e.decided_at and e.decided_at <= streak_end]
    after = [e for e in buys if e.decided_at and e.decided_at > streak_end]
    days_before = len({e.decided_at.astimezone(NEW_YORK).date() for e in before if e.decided_at})
    days_after = len({e.decided_at.astimezone(NEW_YORK).date() for e in after if e.decided_at})
    if days_before < MIN_SESSIONS_SIDE or days_after < MIN_SESSIONS_SIDE:
        return [_finding("losing_streak", "info",
                         f"a run of {STREAK}+ losing positions ended {streak_end:%Y-%m-%d}; too few sessions on either side "
                         "to compare behaviour", len(mine))]  # fmt: skip

    def per_day(xs: Sequence[BrainExecutionRow], days: int) -> tuple[float, float]:
        notional = [e.qty * (e.expected_price or 0) for e in xs]
        return len(xs) / days, (sum(notional) / len(notional) if notional else 0.0)

    rate_b, size_b = per_day(before, days_before)
    rate_a, size_a = per_day(after, days_after)
    change = rate_a / rate_b if rate_b else math.inf
    severity = "warning" if change > 1.5 or (size_b and size_a / size_b > 1.5) or change < 0.5 else "info"
    return [_finding("losing_streak", severity,
                     f"after a run of {STREAK}+ losing positions: {rate_a:.1f} buys/session vs {rate_b:.1f} before, "
                     f"average buy ${size_a:,.0f} vs ${size_b:,.0f}", len(mine),
                     buys_per_session=[round(rate_b, 2), round(rate_a, 2)], avg_buy=[round(size_b), round(size_a)])]  # fmt: skip


async def monitor(db: Database, settings: Settings, now: datetime, days: int = WINDOW_DAYS) -> dict[str, Any]:
    since = now - timedelta(days=days)
    async with db.session() as s:
        executions = (
            await s.scalars(select(BrainExecutionRow).where(BrainExecutionRow.decided_at >= since))
        ).all()
        theses = (await s.scalars(select(BrainThesisRow))).all()
        sessions = (
            await s.scalars(
                select(BrainSessionRow)
                .where(
                    BrainSessionRow.owner == "brain", BrainSessionRow.day >= since.astimezone(NEW_YORK).date()
                )
                .order_by(BrainSessionRow.day)
            )
        ).all()
        consensus = (
            await s.scalars(select(BrainConsensusRow).where(BrainConsensusRow.created_at >= since))
        ).all()
        recent_cycles = (
            await s.scalars(
                select(BrainCycleRow.id)
                .where(BrainCycleRow.started_at >= since)
                .order_by(BrainCycleRow.id.desc())
                .limit(200)
            )
        ).all()
        opinions = (
            (
                await s.scalars(select(BrainOpinionRow).where(BrainOpinionRow.cycle_id.in_(recent_cycles)))
            ).all()
            if recent_cycles
            else []
        )
        ideas = (
            await s.scalars(
                select(BrainOpportunityOutcomeRow).where(BrainOpportunityOutcomeRow.detected_at >= since)
            )
        ).all()
        last = (
            await s.scalars(
                select(BrainCycleRow)
                .where(BrainCycleRow.status == "completed", BrainCycleRow.kind == "full")
                .order_by(BrainCycleRow.id.desc())
            )
        ).first()
    closed = [t for t in theses if t.status == "closed" and t.closed_at and t.closed_at >= since]
    findings = [
        *round_trips(executions, closed),
        *turnover(sessions),
        *concentration([t for t in theses if t.status == "open"], settings.trading_max_position_pct),
        *correlated_positions(last.portfolio if last is not None else None),
        *repeated_losses(closed),
        *ignored_opportunities(ideas),
        *herding(consensus, opinions),
        *instability(consensus),
        *losing_streak(closed, executions),
    ]
    rank = {"alert": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: rank[f["severity"]])
    alerts = [f for f in findings if f["severity"] == "alert"]
    warnings = [f for f in findings if f["severity"] == "warning"]
    return {
        "at": now.isoformat(),
        "window_days": days,
        "headline": (
            f"{len(alerts)} alert(s), {len(warnings)} warning(s)" if alerts or warnings else "nothing pathological found"
        ),
        "findings": findings,
        "checked": ["round_trips", "turnover", "concentration", "correlated_positions", "repeated_thesis_losses",
                    "ignored_opportunities", "agent_herding", "consensus_instability", "losing_streak"],
        "note": "findings describe behaviour; they change no limit, threshold or weight, and stop nothing",
    }  # fmt: skip
