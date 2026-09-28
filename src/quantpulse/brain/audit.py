"""The audit trail of one Brain decision, from the idea to what it taught:

OPPORTUNITY → DATA → AGENTS → OPINIONS → EVIDENCE → DISAGREEMENT → CONSENSUS → DEBATE → PORTFOLIO FIT →
PORTFOLIO DECISION → RISK CHECK → ORDER → ALPACA RESPONSE → EXECUTION → FILL → POSITION → P&L →
BENCHMARK-RELATIVE OUTCOME → PREDICTION GRADE → DECISION QUALITY → LESSON

Everything is read from what was recorded at the time (the cycle, its opportunities, opinions and their
evidence, consensus and debate, the decision with its fit, risk preview and execution, the trading
service's order records and events, the execution ledger, the position's thesis, graded predictions,
reflections and trade lessons) — nothing is recomputed, so the trail shows what the Brain actually knew and
did. Each stage says whether it is **done**, **pending** (it comes later: an open position, a prediction
not yet due), **none** (it did not happen and was not expected: nothing is sent for an unsent decision),
**missing** (it should have been recorded and was not — a gap in the record) or **n/a**.

:func:`completeness` runs the trail over every Brain order that was sent and lists the gaps: an experiment
is only as good as its record, so a missing link is reported, never filled in.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

from sqlalchemy import select

from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import (
    BrainConsensusRow,
    BrainCycleRow,
    BrainDebateRow,
    BrainDecisionRow,
    BrainExecutionRow,
    BrainMemoryRow,
    BrainOpinionRow,
    BrainOpportunityRow,
    BrainPredictionRow,
    BrainReflectionRow,
    BrainStateRow,
    BrainThesisRow,
    BrokerOrderRow,
    TradingEventRow,
)
from quantpulse.db.session import Database
from quantpulse.services.order_manager import FINAL

from .ledger import view as ledger_view
from .memory import LONG_TERM

STAGES = (
    "opportunity",
    "data",
    "agents",
    "opinions",
    "evidence",
    "disagreement",
    "consensus",
    "debate",
    "portfolio_fit",
    "portfolio_decision",
    "risk_check",
    "order",
    "alpaca_response",
    "execution",
    "fill",
    "position",
    "pnl",
    "benchmark_relative",
    "prediction_grade",
    "decision_quality",
    "lesson",
)
TRADE_LESSONS_KEY = "trade_lessons"  # brain_state: when trade lessons were last written (trade_lessons.py)
TRADE_ACTIONS = ("buy", "increase", "reduce", "close", "sell", "de_risk", "rebalance")
ENTRIES = ("buy", "increase")
DONE, PENDING, NONE, MISSING, NA = "done", "pending", "none", "missing", "n/a"


def _stage(name: str, status: str | bool | None, summary: str, **detail: Any) -> dict[str, Any]:
    if isinstance(status, bool) or status is None:  # done / none / n/a
        status = DONE if status else NONE if status is False else NA
    return {"stage": name, "status": status, "summary": summary, "detail": detail}


async def trades(db: Database, limit: int = 50) -> list[dict[str, Any]]:
    """Recent trade decisions (newest first) and how far each got."""
    async with db.session() as s:
        rows = (
            await s.scalars(
                select(BrainDecisionRow)
                .where(BrainDecisionRow.action.in_(TRADE_ACTIONS), BrainDecisionRow.quantity.is_not(None))
                .order_by(BrainDecisionRow.id.desc())
                .limit(limit)
            )
        ).all()
    out = []
    for d in rows:
        ex = d.execution or {}
        out.append(
            {
                "decision_id": d.id,
                "cycle_id": d.cycle_id,
                "at": d.created_at.isoformat(),
                "subject": d.subject,
                "action": d.action,
                "quantity": d.quantity,
                "status": d.status,
                "sent": bool(ex.get("sent")),
                "client_order_id": ex.get("client_order_id"),
                "filled_qty": ex.get("filled_qty"),
                "filled_avg_price": ex.get("filled_avg_price"),
                "reason": ex.get("reason"),
            }
        )
    return out


async def trail(db: Database, decision_id: int) -> dict[str, Any] | None:
    async with db.session() as s:
        d = await s.get(BrainDecisionRow, decision_id)
        if d is None:
            return None
        cycle = await s.get(BrainCycleRow, d.cycle_id)
        opportunities = (
            await s.scalars(select(BrainOpportunityRow).where(BrainOpportunityRow.cycle_id == d.cycle_id))
        ).all()
        opinions = (
            await s.scalars(
                select(BrainOpinionRow)
                .where(BrainOpinionRow.cycle_id == d.cycle_id, BrainOpinionRow.subject == d.subject)
                .order_by(BrainOpinionRow.id)
            )
        ).all()
        consensus = await s.get(BrainConsensusRow, d.consensus_id) if d.consensus_id else None
        debate = (
            await s.scalars(
                select(BrainDebateRow).where(
                    BrainDebateRow.cycle_id == d.cycle_id, BrainDebateRow.subject == d.subject
                )
            )
        ).first()
        ex = d.execution or {}
        cid = ex.get("client_order_id")
        order = (
            (await s.scalars(select(BrokerOrderRow).where(BrokerOrderRow.client_order_id == cid))).first()
            if cid
            else None
        )
        events = (
            (
                await s.scalars(
                    select(TradingEventRow)
                    .where(TradingEventRow.client_order_id == cid)
                    .order_by(TradingEventRow.id)
                )
            ).all()
            if cid
            else []
        )
        ledger = (
            (
                await s.scalars(select(BrainExecutionRow).where(BrainExecutionRow.client_order_id == cid))
            ).first()
            if cid
            else None
        )
        thesis = (
            await s.scalars(
                select(BrainThesisRow).where(
                    (BrainThesisRow.entry_decision_id == d.id) | (BrainThesisRow.exit_decision_id == d.id)
                )
            )
        ).first()
        if thesis is None and order is not None and order.filled_quantity > 0:
            # an increase or a trim: the position it changed (open at the time of the decision)
            thesis = (
                await s.scalars(
                    select(BrainThesisRow)
                    .where(
                        BrainThesisRow.symbol == d.subject,
                        BrainThesisRow.opened_at <= d.created_at,
                        (BrainThesisRow.closed_at.is_(None)) | (BrainThesisRow.closed_at >= d.created_at),
                    )
                    .order_by(BrainThesisRow.id.desc())
                )
            ).first()
        # the consensus call behind the decision: made in its cycle, or — a claim is recorded once a day, so a
        # repeated view is not counted twice — the same day's earlier call it repeated
        day_start = datetime.combine(d.created_at.astimezone(NEW_YORK).date(), time(0, 0), NEW_YORK)
        predictions = (
            await s.scalars(
                select(BrainPredictionRow).where(
                    BrainPredictionRow.subject == d.subject,
                    BrainPredictionRow.source_type == "consensus",
                    (BrainPredictionRow.cycle_id == d.cycle_id)
                    | (
                        (BrainPredictionRow.made_at >= day_start)
                        & (BrainPredictionRow.made_at <= d.created_at)
                    ),
                )
            )
        ).all()
        reflections = (
            await s.scalars(
                select(BrainReflectionRow).where(
                    BrainReflectionRow.subject_type == "decision", BrainReflectionRow.subject_id == d.id
                )
            )
        ).all()
        lesson = (
            (
                await s.scalars(
                    select(BrainMemoryRow).where(
                        BrainMemoryRow.tier == LONG_TERM, BrainMemoryRow.key == f"trade_outcome:{thesis.id}"
                    )
                )
            ).first()
            if thesis is not None
            else None
        )

        state = await s.get(BrainStateRow, TRADE_LESSONS_KEY)
        lessons_at = (
            datetime.fromisoformat(state.value["at"])
            if state is not None and (state.value or {}).get("at")
            else None
        )
        # a fill becomes a position at the next cycle's reconciliation: missing only if one has run since
        reconciled_since = (
            (
                await s.scalars(
                    select(BrainCycleRow.id).where(
                        BrainCycleRow.started_at
                        > (order.filled_at or order.submitted_at or order.created_at),
                        BrainCycleRow.mode == "paper_execution",
                        BrainCycleRow.status == "completed",
                    )
                )
            ).first()
            is not None
            if order is not None and order.filled_quantity > 0
            else False
        )

    sent = bool(ex.get("sent"))
    trade = d.action in TRADE_ACTIONS and bool(d.quantity)
    filled = order is not None and order.filled_quantity > 0
    final = order is not None and order.status in FINAL
    stages: list[dict[str, Any]] = []

    mine = [o for o in opportunities if d.subject in (o.symbols or []) or o.subject == d.subject]
    stages.append(
        _stage(
            "opportunity",
            bool(mine) or None,
            "; ".join(f"{o.kind}: {o.headline} ({o.status})" for o in mine[:3])
            or "not from a detected opportunity (a holding, the pre-screen or a request)",
            items=[
                {
                    "kind": o.kind,
                    "headline": o.headline,
                    "strength": o.strength,
                    "status": o.status,
                    "stages": o.stages,
                }
                for o in mine[:5]
            ],
        )
    )
    dq = (cycle.data_quality if cycle is not None else {}) or {}
    diag = (dq.get("diagnosis") or {}).get(d.subject)
    stages.append(
        _stage(
            "data",
            DONE if diag else MISSING if sent else NONE,
            f"{diag['status']} quote ({diag.get('feed') or 'no feed'}, priced on {diag.get('price_source') or '?'}, "
            f"last trade {diag.get('trade_age_s')}s old, spread {diag.get('spread_bps')}bp "
            f"{diag.get('spread_source') or ''})"
            if diag
            else "no diagnosis recorded",
            diagnosis=diag,
            market=(dq.get("market") or {}).get("thesis"),
            feed_headline=(dq.get("feed") or {}).get("headline"),
        )
    )
    ran = [a for a in (cycle.agents if cycle is not None else []) or [] if a.get("status") == "ok"]
    agents = sorted({o.agent_id for o in opinions})
    stages.append(
        _stage(
            "agents",
            DONE if opinions else MISSING if sent else NONE,
            f"{len(agents)} agent(s) gave a view on {d.subject}; {len(ran)} ran in the cycle",
            ran=[a["agent_id"] for a in ran],
            on_subject=agents,
        )
    )
    stages.append(
        _stage(
            "opinions",
            DONE if opinions else MISSING if sent else NONE,
            ", ".join(f"{o.agent_id} {o.stance} {o.score:+.2f}" for o in opinions[:12]),
            opinions=[
                {
                    "agent": o.agent_id,
                    "version": o.agent_version,
                    "stance": o.stance,
                    "score": o.score,
                    "confidence": o.confidence,
                    "horizon_days": o.horizon_days,
                    "thesis": o.thesis,
                    "invalidation": o.invalidation,
                    "data": o.data_quality,
                }
                for o in opinions
            ],
        )
    )
    facts = [{**e, "agent": o.agent_id} for o in opinions for e in (o.evidence or []) if isinstance(e, dict)]
    strongest = sorted(facts, key=lambda e: -float(e.get("strength") or 0))
    stages.append(
        _stage(
            "evidence",
            DONE if facts else MISSING if sent else NONE,
            f"{len(facts)} fact(s) from {len({e['agent'] for e in facts})} agent(s); strongest: "
            + "; ".join(f"{e['agent']}: {e.get('detail')}" for e in strongest[:3])
            if facts
            else "no evidence recorded",
            supporting=[e for e in strongest if (e.get("direction") or 0) > 0][:10],
            opposing=[e for e in strongest if (e.get("direction") or 0) < 0][:10],
            missing_data={o.agent_id: o.data_missing for o in opinions if o.data_missing},
            not_live=[e for e in facts if e.get("quality") not in (None, "fresh", "live")][:10],
        )
    )
    cd = (consensus.detail if consensus is not None else {}) or {}
    primary = cd.get("primary_disagreement") or {}
    stages.append(
        _stage(
            "disagreement",
            consensus is not None or (MISSING if sent else NONE),
            (
                f"disagreement {consensus.disagreement:.2f} ({consensus.supporting} for, {consensus.neutral} neutral, "
                f"{consensus.opposing} against, {consensus.abstaining} abstaining; "
                f"{cd.get('independent_sources', '?')} independent source(s))"
                + (f" — {primary.get('summary')}" if primary.get("summary") else "")
            )
            if consensus is not None
            else "no consensus recorded",
            primary=primary,
            sources=cd.get("sources"),
            missing_agents=cd.get("missing"),
            uncertainty=cd.get("uncertainty"),
        )
    )
    stages.append(
        _stage(
            "consensus",
            consensus is not None or (MISSING if sent else NONE),
            f"{consensus.stance} {consensus.score:+.2f}, confidence {consensus.confidence:.2f}"
            + (" — unknown" if consensus.unknown else "")
            if consensus is not None
            else "no consensus recorded",
            reasons=consensus.reasons if consensus is not None else [],
            vetoes=consensus.vetoes if consensus is not None else [],
            votes=cd.get("votes"),
        )
    )
    stages.append(
        _stage(
            "debate",
            debate is not None or None,
            f"{debate.verdict}: confidence {debate.confidence_before:.2f} → {debate.confidence_after:.2f}"
            if debate is not None
            else "no debate",
            objections=debate.objections if debate is not None else [],
            bull=debate.bull if debate is not None else [],
            bear=debate.bear if debate is not None else [],
        )
    )
    rationale = d.rationale or {}
    fit = rationale.get("fit") or {}
    stages.append(
        _stage(
            "portfolio_fit",
            (DONE if fit else MISSING if sent else NONE) if d.action == "buy" else NA,
            ("fits" if fit.get("ok") else "does not fit")
            + (": " + "; ".join(fit.get("notes") or []) if fit.get("notes") else "")
            + (
                f" (portfolio vol {fit.get('vol_before')} → {fit.get('vol_after')}, risk share {fit.get('risk_share')})"
                if fit.get("vol_after") is not None
                else ""
            )
            if fit
            else "checked for new positions only"
            if d.action != "buy"
            else "no fit recorded",
            **({"fit": fit} if fit else {}),
        )
    )
    stages.append(
        _stage(
            "portfolio_decision",
            True,
            f"{d.action.upper()} {d.quantity or 0:g} {d.subject} ({d.status}): "
            + "; ".join((rationale.get("reasons") or [])[:3]),
            confidence=d.confidence,
            entry=rationale.get("entry"),
            blocked_by=rationale.get("blocked_by"),
            memory=rationale.get("memory"),
            protective=rationale.get("protective"),
            mode=d.mode,
        )
    )
    stages.append(
        _stage(
            "risk_check",
            DONE if (d.risk or ex.get("checks")) else MISSING if sent else NONE if trade else NA,
            f"preview: {(d.risk or {}).get('summary') or 'not checked'}"
            + (f"; at execution: {ex.get('risk')}" if ex.get("risk") else ""),
            preview=d.risk,
            at_execution=ex.get("checks"),
        )
    )
    stages.append(
        _stage(
            "order",
            DONE if order is not None else MISSING if sent else NONE,
            f"{order.side} {order.quantity or 0:g} {order.symbol} {order.order_type}"
            + (f" @ {order.limit_price}" if order.limit_price else "")
            + f" ({order.client_order_id})"
            if order is not None
            else str(ex.get("reason") or "no order was sent"),
            trading_cycle_id=ex.get("trading_cycle_id"),
            client_order_id=cid,
        )
    )
    stages.append(
        _stage(
            "alpaca_response",
            DONE if order is not None and (order.alpaca_order_id or final) else MISSING if sent else NONE,
            f"Alpaca order {order.alpaca_order_id}: {order.status}"
            + (f" — {order.error}" if order.error else "")
            if order is not None
            else "nothing was sent to Alpaca",
            events=[{"at": e.created_at.isoformat(), "kind": e.kind, "message": e.message} for e in events],
        )
    )
    stages.append(
        _stage(
            "execution",
            DONE if ledger is not None else MISSING if sent else NONE,
            (
                f"{ledger.status}; quote {ledger.quote_price} ({ledger.quote_source}, {ledger.quote_age_s}s old, "
                f"spread {ledger.spread_bps}bp); latency {ledger.submit_latency_ms}ms; slippage "
                f"{ledger.slippage_bps}bp, vs quote {ledger.cost_vs_quote_bps}bp — {ledger.grade or 'not graded'}"
            )
            if ledger is not None
            else "no execution record",
            **({"ledger": ledger_view(ledger)} if ledger is not None else {}),
        )
    )
    fill_px = order.average_fill_price if order is not None else None
    est = ex.get("est_price") or d.est_price
    stages.append(
        _stage(
            "fill",
            DONE if filled else (NONE if final else PENDING) if order is not None else NONE,
            f"filled {order.filled_quantity:g} @ ${fill_px or 0:,.2f}"
            + (f" at {order.filled_at:%H:%M:%S} UTC" if order.filled_at else "")
            + (" (partial)" if final and order.filled_quantity < (order.quantity or 0) - 1e-9 else "")
            if filled and order is not None
            else f"no fill ({order.status})"
            if order is not None
            else "no fill",
            est_price=est,
            slippage_bps=round((fill_px / est - 1) * 10_000, 1) if filled and fill_px and est else None,
        )
    )
    stages.append(
        _stage(
            "position",
            DONE if thesis is not None else (MISSING if reconciled_since else PENDING) if filled else NONE,
            f"{thesis.status} {thesis.origin} position: {thesis.qty:g} @ ${thesis.avg_price:,.2f}, check "
            f"{(thesis.check or {}).get('status', '—')}"
            if thesis is not None
            else "no position record yet (the next cycle reconciles the fill into a thesis)"
            if filled
            else "no position",
            thesis_id=thesis.id if thesis is not None else None,
            thesis=thesis.thesis if thesis is not None else None,
            stop_price=thesis.stop_price if thesis is not None else None,
            target_price=thesis.target_price if thesis is not None else None,
            horizon_days=thesis.horizon_days if thesis is not None else None,
            exit_reason=thesis.exit_reason if thesis is not None else None,
        )
    )
    stages.append(_pnl(d.action, order, thesis, filled))
    stages.append(_relative(thesis, filled))
    graded = [p for p in predictions if p.status == "evaluated"]
    open_ = [p for p in predictions if p.status == "open"]
    stages.append(
        _stage(
            "prediction_grade",
            DONE
            if graded
            else PENDING
            if open_
            else (
                MISSING
                if sent
                and consensus is not None
                and not consensus.unknown
                and consensus.stance in ("bullish", "bearish")
                else NA
            ),
            "; ".join(
                f"{p.horizon_days}d {'hit' if p.hit else 'miss'} ({p.realized_relative:+.2%} vs benchmark, "
                f"confidence {p.confidence:.2f})"
                for p in graded
                if p.realized_relative is not None
            )
            or (
                f"{len(open_)} prediction(s) open until their horizon (due {min(p.due_date for p in open_)})"
                if open_
                else "no gradeable consensus call (neutral or unknown)"
            ),
            predictions=[
                {
                    "horizon": p.horizon_days,
                    "status": p.status,
                    "hit": p.hit,
                    "due": p.due_date.isoformat(),
                    "relative": p.realized_relative,
                    "confidence": p.confidence,
                    "regime": p.regime,
                    "made_in_cycle": p.cycle_id,
                    "breakdown": {
                        k: (p.context or {}).get(k) for k in ("relative_z", "noise", "timing", "data_ok")
                    },
                }
                for p in predictions
            ],
        )
    )
    outcome = d.outcome or {}
    if reflections:
        quality_status = DONE
    elif open_:
        quality_status = PENDING
    elif d.evaluated_at is not None:
        quality_status = MISSING  # graded but never reflected on (the same learning pass writes both)
    else:
        quality_status = NA  # no gradeable call of its own (e.g. a protective exit on a neutral view)
    stages.append(
        _stage(
            "decision_quality",
            quality_status,
            "; ".join(
                f"{r.category}: decision {r.decision_quality}, outcome {r.outcome_quality}"
                for r in reflections
            )
            or "judged once its outcome is graded (decision quality from what was known then, outcome apart)",
            reflections=[
                {
                    "category": r.category,
                    "decision_quality": r.decision_quality,
                    "outcome_quality": r.outcome_quality,
                    "questions": r.questions,
                    "evidence": r.evidence,
                }
                for r in reflections
            ],
            outcome=outcome,
        )
    )
    lessons = [x for r in reflections for x in (r.lessons or [])]
    closed = thesis is not None and thesis.status == "closed"
    if lessons or lesson is not None:
        lesson_status = DONE
    elif quality_status == PENDING or (filled and not closed):
        lesson_status = PENDING
    elif closed and thesis is not None and thesis.closed_at is not None:
        # trade lessons are written after the close: missing only if that pass has run since
        lesson_status = MISSING if lessons_at is not None and lessons_at > thesis.closed_at else PENDING
    elif quality_status == NA and not filled:
        lesson_status = NA
    else:
        lesson_status = MISSING
    stages.append(
        _stage(
            "lesson",
            lesson_status,
            "; ".join([*(lessons[:3]), *([lesson.summary] if lesson is not None else [])])
            or "written once the decision is graded and the position closed",
            lessons=lessons,
            trade_lesson=({"summary": lesson.summary, "data": lesson.data} if lesson is not None else None),
        )
    )
    return {
        "decision_id": d.id,
        "cycle_id": d.cycle_id,
        "subject": d.subject,
        "action": d.action,
        "at": d.created_at.isoformat(),
        "sent": sent,
        "stages": stages,
        "gaps": [st["stage"] for st in stages if st["status"] == MISSING],
        "pending": [st["stage"] for st in stages if st["status"] == PENDING],
    }


def _pnl(action: str, order: Any, thesis: BrainThesisRow | None, filled: bool) -> dict[str, Any]:
    """This order's own P&L (its shares marked at the latest price, or realised against the average cost),
    and the position's."""
    if not filled or order is None:
        return _stage("pnl", NONE, "no fill: no P&L")
    if thesis is None:
        return _stage("pnl", PENDING, "marked once the fill is reconciled into a position")
    qty, px = order.filled_quantity, order.average_fill_price or 0.0
    closed = thesis.status == "closed"
    if action in ENTRIES:
        mark = (thesis.exit_price if closed else thesis.last_price) or px
        own = round((mark - px) * qty, 2)
        kind = "realised" if closed else "unrealised"
        summary = f"this order's {qty:g} shares: ${own:+,.2f} {kind} (${px:,.2f} → ${mark:,.2f})"
    else:
        own = round((px - thesis.avg_price) * qty, 2)
        kind = "realised"
        summary = f"sold {qty:g} at ${px:,.2f} against an average cost of ${thesis.avg_price:,.2f}: ${own:+,.2f} realised"
    position = thesis.realized_pnl if closed else thesis.unrealized_pnl
    return _stage(
        "pnl",
        DONE,
        summary
        + (
            f"; the position: ${position:+,.2f} {'realised' if closed else 'unrealised'}"
            if position is not None
            else ""
        ),
        order_pnl=own,
        kind=kind,
        position_pnl=position,
        position_status=thesis.status,
    )


def _relative(thesis: BrainThesisRow | None, filled: bool) -> dict[str, Any]:
    """The position against the benchmark since entry (at its exit, or its latest mark while open)."""
    if not filled:
        return _stage("benchmark_relative", NONE, "no fill: no outcome")
    if thesis is None:
        return _stage("benchmark_relative", PENDING, "measured once the fill is reconciled into a position")
    closed = thesis.status == "closed"
    ret = (
        thesis.exit_price / thesis.entry_price - 1
        if closed and thesis.exit_price and thesis.entry_price
        else thesis.return_pct
    )
    bench = thesis.benchmark_return
    if ret is None or bench is None:
        return _stage(
            "benchmark_relative",
            MISSING,
            "no benchmark level recorded at entry or at the last mark",
            return_pct=ret,
            benchmark_return=bench,
        )
    rel = ret - bench
    return _stage(
        "benchmark_relative",
        DONE,
        f"{ret:+.2%} vs the benchmark's {bench:+.2%}: {rel:+.2%} "
        + ("over the holding period" if closed else "so far (open)"),
        return_pct=round(ret, 5),
        benchmark_return=bench,
        relative=round(rel, 5),
        final=closed,
    )


async def completeness(db: Database, limit: int = 200) -> dict[str, Any]:
    """Every Brain order that was sent, followed through its trail: which links are recorded, which are
    still to come, and which are missing (a gap in the experiment's record)."""
    async with db.session() as s:
        ids = (
            await s.scalars(
                select(BrainDecisionRow.id)
                .where(BrainDecisionRow.action.in_(TRADE_ACTIONS), BrainDecisionRow.quantity.is_not(None))
                .order_by(BrainDecisionRow.id.desc())
                .limit(limit * 5)
            )
        ).all()
    rows: list[dict[str, Any]] = []
    gaps: dict[str, int] = {}
    pending: dict[str, int] = {}
    for i in ids:
        t = await trail(db, i)
        if t is None or not t["sent"]:
            continue
        for g in t["gaps"]:
            gaps[g] = gaps.get(g, 0) + 1
        for p in t["pending"]:
            pending[p] = pending.get(p, 0) + 1
        rows.append(
            {k: t[k] for k in ("decision_id", "cycle_id", "subject", "action", "at", "gaps", "pending")}
            | {"complete": not t["gaps"] and not t["pending"]}
        )
        if len(rows) >= limit:
            break
    with_gaps = sum(1 for r in rows if r["gaps"])
    return {
        "trades": len(rows),
        "complete": sum(1 for r in rows if r["complete"]),
        "in_progress": sum(1 for r in rows if not r["gaps"] and r["pending"]),
        "with_gaps": with_gaps,
        "gaps_by_stage": gaps,
        "pending_by_stage": pending,
        "headline": (
            "no Brain order sent yet"
            if not rows
            else f"{with_gaps} of {len(rows)} sent order(s) have a gap in their record"
            if with_gaps
            else f"every one of {len(rows)} sent order(s) is traceable; "
            f"{sum(1 for r in rows if r['pending'])} still waiting for later stages"
        ),
        "trades_detail": rows,
    }
