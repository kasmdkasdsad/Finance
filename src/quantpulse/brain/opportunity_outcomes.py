"""Opportunities the Brain considered — taken or not — and what happened next.

The opportunity engine records how far each idea got and where it stopped (:func:`.opportunities.trace`).
This module turns that into an experiment the Brain can learn from:

1. **Record** (every cycle): one row per idea per day — the same kind, symbol and direction detected again
   the same day is folded into the first record (``repeats``), so a view repeated every half hour never
   counts as many observations. Each row keeps whether a trade on it went out (``taken``) and, if not,
   **why** as a category (:data:`REASONS`: the focus budget, data quality, no view, an unknown or opposing
   consensus, low confidence, earnings, a risk-off market, the posture, the devil's advocate, portfolio fit,
   the new-position limit, cash, the risk engine, an entry halt, the market being closed …), with the price
   and the benchmark at the first detection and the annualised volatility.
2. **Grade** (the learning pass): after a fixed horizon per kind of idea (:data:`HORIZON`), against real
   closes only — the idea's return relative to the benchmark, signed by its direction (``favourable``), and
   in units of the horizon's risk (``z``). Within half a standard deviation it is **noise**; otherwise an idea
   *not taken* was **avoided** correctly (it went against the idea) or **missed** (it worked), and one taken
   **worked** or **failed**. A bearish idea on a symbol the Brain did not own had nothing to act on: it grades
   the detector (``signal_right`` / ``signal_wrong``), not a decision.
3. **Report**: per rejection reason — has it been saving money or costing opportunities? — per kind of idea
   and per regime, and taken against rejected. Every figure carries its sample (independent ideas, not
   detections) and a 95% interval, and says *unproven* until there are enough decisive outcomes
   (``QP_BRAIN_MIN_RELIABILITY_OBSERVATIONS``). Nothing here changes a rule: a reason that looks costly
   becomes a pattern in memory and, if it is not a protected control, an improvement proposal for a person.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, sessions_after
from quantpulse.db.models import BrainOpportunityOutcomeRow
from quantpulse.db.session import Database
from quantpulse.services.predictions import last_completed_session

from .context import BrainContext
from .evaluation import VOID_AFTER_DAYS, PriceSource
from .memory import LONG_TERM, MemoryStore
from .opportunities import Opportunity
from .performance import vol_environment, wilson
from .types import BUYING, SELLING, Action

NOISE_Z = 0.5
DEFAULT_HORIZON = 10
HORIZON = {  # sessions: roughly the span each kind of idea is about
    "momentum_shift": 21,
    "breakout": 10,
    "abnormal_volume": 5,
    "valuation_dislocation": 21,
    "mean_reversion": 5,
    "volatility_event": 5,
    "sector_rotation": 21,
    "relative_value": 10,
    "earnings": 5,
    "catalyst": 10,
    "unusual_options": 5,
}
REASONS = {
    "taken": "a trade on it went out",
    "already_held": "already owned (the idea is in the book)",
    "nothing_to_sell": "a bearish idea on a symbol not owned (nothing to act on)",
    "focus_budget": "outside the cycle's focus budget (stronger ideas came first)",
    "data_quality": "the symbol's market data was not executable",
    "no_view": "no forecasting agent had a view",
    "unknown_consensus": 'not enough evidence for a view ("I do not know")',
    "consensus_against": "the consensus did not support it",
    "low_confidence": "the consensus was below the confidence threshold",
    "earnings": "an earnings release too close",
    "risk_off": "a risk-off market",
    "posture": "a defensive posture",
    "challenged": "the devil's advocate challenged it",
    "portfolio_fit": "a poor portfolio fit (the same bet, a sector, the book's risk)",
    "slot_limit": "beyond the new positions allowed in one cycle",
    "cash": "not enough spendable cash",
    "margin": "the account was on margin",
    "data_veto": "a data veto on the decision",
    "risk_engine": "the risk engine rejected it",
    "entry_halt": "new positions were halted",
    "account": "the account could not be read",
    "market_closed": "the market was closed",
    "other": "another reason (see the detail)",
}
# reasons that are protected controls: a finding about them is a question for a person, never a change
PROTECTED_REASONS = frozenset(
    {"data_quality", "data_veto", "risk_engine", "entry_halt", "account", "market_closed"}
)
EXCLUDED = frozenset({"already_held", "nothing_to_sell"})  # not a rejection decision


def _watch_reason(text: str) -> str:
    t = text.lower()
    for needle, reason in (
        ("devil's advocate", "challenged"),
        ("poor portfolio fit", "portfolio_fit"),
        ("beyond the", "slot_limit"),
        ("spendable cash", "cash"),
        ("margin", "margin"),
        ("earnings", "earnings"),
        ("risk-off", "risk_off"),
        ("defensive", "posture"),
        ("confidence", "low_confidence"),
    ):
        if needle in t:
            return reason
    return "data_veto" if "stale" in t or "data" in t else "other"


def classify(o: Opportunity, p: Any, market_open: bool, held: bool) -> tuple[bool, str, str]:
    """(taken, reason, detail) for one idea at one detection. ``p`` is the decision on its lead symbol."""
    if p is not None:
        wanted = BUYING if o.direction > 0 else SELLING
        trade = p.is_trade and p.action in wanted
        if trade and p.risk_approved and p.status not in ("halted", "blocked", "risk_rejected"):
            return True, "taken", f"{p.action.value} ({p.status})"
        if trade:
            code = {"risk_rejected": "risk_engine", "blocked": "data_veto", "halted": "entry_halt",
                    "not_checked": "account"}.get(p.status, "risk_engine")  # fmt: skip
            return False, code, "; ".join(p.blocked_by or [(p.risk or {}).get("summary") or p.status])[:300]
        why = "; ".join(p.reasons)[:300]
        if o.direction > 0 and held:
            return False, "already_held", why
        if o.direction < 0 and not held:
            return False, "nothing_to_sell", why
        if not market_open:
            return False, "market_closed", why
        if p.action is Action.NO_ACTION:
            unknown = bool(p.reasons) and p.reasons[0].startswith("I do not know")
            return False, "unknown_consensus" if unknown else "consensus_against", why
        if p.action is Action.WATCH:
            return False, _watch_reason(why), why
        return False, "consensus_against", why  # a hold or a trade the other way
    if o.direction < 0 and not held:
        return False, "nothing_to_sell", o.status
    if not market_open:
        return False, "market_closed", o.status
    code = {"not_analysed": "focus_budget", "rejected_data": "data_quality", "no_view": "no_view"}.get(
        o.status, "other"
    )
    last = o.stages[-1]["result"] if o.stages else o.status
    return False, code, str(last)[:300]


async def record(
    db: Database,
    ctx: BrainContext,
    opportunities: Sequence[Opportunity],
    proposals: dict[str, Any],
    cycle_id: int,
    opportunity_ids: dict[int, int] | None = None,
) -> int:
    """One record per idea per day (kind, lead symbol, direction); returns how many were new."""
    now = ctx.as_of
    day = now.astimezone(NEW_YORK).date()
    bench = ctx.price(ctx.benchmark_symbol)
    env = vol_environment(ctx.market_stats.get("benchmark_rv21"))
    regime = ctx.regime.label if ctx.regime else None
    new = 0
    async with db.session() as s:
        for i, o in enumerate(opportunities):
            lead = o.lead
            if lead is None or o.direction == 0 or o.status == "context":
                continue
            taken, reason, detail = classify(o, proposals.get(lead), ctx.market_open, lead in ctx.held)
            row = (
                await s.scalars(
                    select(BrainOpportunityOutcomeRow).where(
                        BrainOpportunityOutcomeRow.kind == o.kind,
                        BrainOpportunityOutcomeRow.symbol == lead,
                        BrainOpportunityOutcomeRow.direction == o.direction,
                        BrainOpportunityOutcomeRow.day == day,
                    )
                )
            ).first()
            stopped = o.stages[-1]["stage"] if o.stages else None
            if row is not None:  # the same idea again today: one observation, its latest disposition
                row.repeats += 1
                row.status, row.updated_at = o.status, now
                if taken and not row.taken:
                    row.taken, row.reason, row.reason_detail, row.stopped_at = True, reason, detail, stopped
                elif not row.taken:
                    row.reason, row.reason_detail, row.stopped_at = reason, detail, stopped
                continue
            horizon = HORIZON.get(o.kind, DEFAULT_HORIZON)
            vol = ctx.ind(lead, "rv63")
            s.add(
                BrainOpportunityOutcomeRow(
                    opportunity_id=(opportunity_ids or {}).get(i),
                    cycle_id=cycle_id,
                    kind=o.kind,
                    symbol=lead,
                    direction=o.direction,
                    day=day,
                    detected_at=now,
                    strength=round(float(o.strength), 4),
                    headline=o.headline[:500],
                    taken=taken,
                    reason=reason,
                    reason_detail=detail,
                    stopped_at=stopped,
                    status=o.status,
                    repeats=0,
                    regime=regime,
                    vol_env=env,
                    market_open=ctx.market_open,
                    horizon_days=horizon,
                    due_date=sessions_after(day, horizon),
                    entry_price=ctx.price(lead),
                    entry_benchmark=bench,
                    vol=float(vol) if vol is not None and vol == vol and vol > 0 else None,
                    state="open",
                    updated_at=now,
                )
            )
            new += 1
    return new


def verdict(taken: bool, reason: str, favourable: float, z: float | None) -> str:
    if z is not None and abs(z) < NOISE_Z:
        return "noise"
    right = favourable > 0
    if reason == "nothing_to_sell":
        return "signal_right" if right else "signal_wrong"
    if taken or reason == "already_held":
        return "worked" if right else "failed"
    return "missed" if right else "avoided"


async def evaluate_due(db: Database, prices: PriceSource, clock: Clock, benchmark: str) -> dict[str, int]:
    """Grade every idea whose horizon has passed, against real closes only."""
    now = clock.now()
    cutoff = last_completed_session(now)
    today = now.astimezone(NEW_YORK).date()
    async with db.session() as s:
        due = list(
            (
                await s.scalars(
                    select(BrainOpportunityOutcomeRow).where(
                        BrainOpportunityOutcomeRow.state == "open",
                        BrainOpportunityOutcomeRow.due_date <= cutoff,
                    )
                )
            ).all()
        )
    out = {"evaluated": 0, "voided": 0, "pending": 0}
    if not due:
        return out
    since = min(r.day for r in due) - timedelta(days=5)
    closes = await prices.closes(sorted({benchmark} | {r.symbol for r in due}), since)
    bench = closes.get(benchmark, {})
    async with db.session() as s:
        for stale in due:
            row = await s.get(BrainOpportunityOutcomeRow, stale.id)
            if row is None or row.state != "open":
                continue
            close = closes.get(row.symbol, {}).get(row.due_date)
            b_close = bench.get(row.due_date)
            entry, b_entry = row.entry_price, row.entry_benchmark
            if not (entry and b_entry and close and b_close):
                if (today - row.due_date).days > VOID_AFTER_DAYS or not entry:
                    row.state, row.evaluated_at = "void", now
                    out["voided"] += 1
                else:
                    out["pending"] += 1
                continue
            ret = close / entry - 1
            rel = ret - (b_close / b_entry - 1)
            fav = row.direction * rel
            z = fav / (row.vol * math.sqrt(row.horizon_days / 252)) if row.vol else None
            row.realized_return, row.relative, row.favourable = round(ret, 6), round(rel, 6), round(fav, 6)
            row.z = round(z, 4) if z is not None else None
            row.verdict = verdict(row.taken, row.reason, fav, z)
            row.state, row.evaluated_at, row.updated_at = "evaluated", now, now
            out["evaluated"] += 1
    return out


def _group(rows: Sequence[BrainOpportunityOutcomeRow], min_n: int, *, rejection: bool) -> dict[str, Any]:
    """A group's outcomes. ``rejection``: the question is whether not taking the ideas was right (avoided vs
    missed); otherwise whether the ideas themselves were right."""
    verdicts: dict[str, int] = {}
    for r in rows:
        verdicts[r.verdict or "?"] = verdicts.get(r.verdict or "?", 0) + 1
    good_k = (
        verdicts.get("avoided", 0)
        if rejection
        else sum(verdicts.get(v, 0) for v in ("worked", "signal_right", "missed"))
    )
    bad_k = (
        verdicts.get("missed", 0)
        if rejection
        else sum(verdicts.get(v, 0) for v in ("failed", "signal_wrong", "avoided"))
    )
    decisive = good_k + bad_k
    ci = wilson(good_k, decisive) if decisive else None
    fav = [r.favourable for r in rows if r.favourable is not None]
    if decisive < min_n:
        status = "unproven"
    elif ci is not None and ci[0] > 0.5:
        status = "right more often than not" if rejection else "the ideas have tended to work"
    elif ci is not None and ci[1] < 0.5:
        status = "costing opportunities" if rejection else "the ideas have tended to fail"
    else:
        status = "no evidence either way"
    return {
        "ideas": len(rows),
        "decisive": decisive,
        "verdicts": verdicts,
        ("avoided_share" if rejection else "right_share"): round(good_k / decisive, 3) if decisive else None,
        "ci95": [round(ci[0], 3), round(ci[1], 3)] if ci else None,
        "mean_favourable": round(sum(fav) / len(fav), 5) if fav else None,
        "status": status,
        "needs": max(min_n - decisive, 0),
    }  # fmt: skip


async def report(db: Database, min_n: int, since: datetime | None = None) -> dict[str, Any]:
    async with db.session() as s:
        q = select(BrainOpportunityOutcomeRow)
        if since is not None:
            q = q.where(BrainOpportunityOutcomeRow.detected_at >= since)
        rows = list((await s.scalars(q)).all())
    graded = [r for r in rows if r.state == "evaluated"]
    rejected = [r for r in graded if not r.taken and r.reason not in EXCLUDED]
    taken = [r for r in graded if r.taken]
    by_reason: dict[str, Any] = {}
    for reason in sorted({r.reason for r in rejected}):
        mine = [r for r in rejected if r.reason == reason]
        by_reason[reason] = {
            "meaning": REASONS.get(reason, reason),
            "protected": reason in PROTECTED_REASONS,
            **_group(mine, min_n, rejection=True),
        }
    by_kind = {k: _group([r for r in graded if r.kind == k], min_n, rejection=False)
               for k in sorted({r.kind for r in graded})}  # fmt: skip
    regimes: dict[str, Any] = {}
    for label, pick in (("regime", lambda r: r.regime), ("volatility", lambda r: r.vol_env)):
        for value in sorted({pick(r) for r in graded if pick(r)}):
            regimes[f"{label}:{value}"] = _group(
                [r for r in graded if pick(r) == value], min_n, rejection=False
            )
    t_fav = [r.favourable for r in taken if r.favourable is not None]
    r_fav = [r.favourable for r in rejected if r.favourable is not None]
    comparison: dict[str, Any] = {
        "taken": {"n": len(t_fav), "mean_favourable": round(sum(t_fav) / len(t_fav), 5) if t_fav else None},
        "rejected": {
            "n": len(r_fav),
            "mean_favourable": round(sum(r_fav) / len(r_fav), 5) if r_fav else None,
        },
    }
    if len(t_fav) >= min_n and len(r_fav) >= min_n:
        diff, t = _welch(t_fav, r_fav)
        comparison["difference"] = round(diff, 5)
        comparison["t"] = round(t, 2) if t is not None else None
        comparison["status"] = (
            "the ideas taken did better than those rejected" if t is not None and t > 2
            else "the ideas rejected did better than those taken" if t is not None and t < -2
            else "no evidence of a difference"
        )  # fmt: skip
    else:
        comparison["status"] = f"unproven: needs {min_n} graded ideas on each side"
    return {
        "recorded": len(rows),
        "open": sum(1 for r in rows if r.state == "open"),
        "graded": len(graded),
        "void": sum(1 for r in rows if r.state == "void"),
        "detections_folded": sum(r.repeats for r in rows),
        "min_decisive": min_n,
        "by_reason": by_reason,
        "by_kind": by_kind,
        "by_regime": regimes,
        "taken_vs_rejected": comparison,
        "note": "one idea per kind, symbol, direction and day; graded against the benchmark over the kind's "
        "horizon; noise (within half a standard deviation) is neither right nor wrong",
    }


def _welch(a: Sequence[float], b: Sequence[float]) -> tuple[float, float | None]:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / max(len(a) - 1, 1)
    vb = sum((x - mb) ** 2 for x in b) / max(len(b) - 1, 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return ma - mb, (ma - mb) / se if se > 0 else None


async def rows(
    db: Database, limit: int = 200, verdict_: str | None = None, reason: str | None = None
) -> list[dict[str, Any]]:
    async with db.session() as s:
        q = select(BrainOpportunityOutcomeRow).order_by(BrainOpportunityOutcomeRow.id.desc()).limit(limit)
        if verdict_:
            q = q.where(BrainOpportunityOutcomeRow.verdict == verdict_)
        if reason:
            q = q.where(BrainOpportunityOutcomeRow.reason == reason)
        found = (await s.scalars(q)).all()
    cols = ("id", "opportunity_id", "cycle_id", "kind", "symbol", "direction", "strength", "headline", "taken",
            "reason", "reason_detail", "stopped_at", "status", "repeats", "regime", "vol_env", "market_open",
            "horizon_days", "entry_price", "state", "realized_return", "relative", "favourable", "z", "verdict")  # fmt: skip
    out = []
    for r in found:
        d = {c: getattr(r, c) for c in cols}
        d.update(
            day=r.day.isoformat(), due_date=r.due_date.isoformat(), detected_at=r.detected_at.isoformat()
        )
        out.append(d)
    return out


async def remember(db: Database, memory: MemoryStore, now: datetime, min_n: int) -> int:
    """One long-term pattern per rejection reason (updated in place): what not taking those ideas has done so
    far, with its sample — context for later decisions and for improvement proposals, never a rule change."""
    rep = await report(db, min_n)
    n = 0
    for reason, g in rep["by_reason"].items():
        if not g["decisive"]:
            continue
        await memory.remember(
            LONG_TERM,
            "pattern",
            "@rejections",
            f"rejected for {g['meaning']}: {g['verdicts'].get('avoided', 0)} avoided, {g['verdicts'].get('missed', 0)} "
            f"missed, {g['verdicts'].get('noise', 0)} noise — {g['status']}",
            now,
            key=f"pattern:rejection:{reason}",
            data={"reason": reason, **g},
            tags=[
                "pattern",
                "rejection",
                reason,
                "established" if g["status"] != "unproven" else "tentative",
            ],
            importance=0.6 if g["status"] != "unproven" else 0.3,
        )
        n += 1
    return n
