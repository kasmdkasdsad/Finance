"""The audit trail of one Brain decision, from the idea to what it taught:

OPPORTUNITY → DATA → AGENTS → OPINIONS → CONSENSUS → DEBATE → PORTFOLIO DECISION → RISK CHECK → ORDER →
ALPACA RESPONSE → FILL → POSITION → OUTCOME → LEARNING

Everything is read from what was recorded at the time (the cycle, its opportunities, opinions, consensus
and debate, the decision with its risk preview and execution, the trading service's order records and
events, the position's thesis, graded predictions and reflections) — nothing is recomputed, so the trail
shows what the Brain actually knew and did. A stage that did not happen says so.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from quantpulse.db.models import (
    BrainConsensusRow,
    BrainCycleRow,
    BrainDebateRow,
    BrainDecisionRow,
    BrainOpinionRow,
    BrainOpportunityRow,
    BrainPredictionRow,
    BrainReflectionRow,
    BrainThesisRow,
    BrokerOrderRow,
    TradingEventRow,
)
from quantpulse.db.session import Database

STAGES = (
    "opportunity",
    "data",
    "agents",
    "opinions",
    "consensus",
    "debate",
    "portfolio_decision",
    "risk_check",
    "order",
    "alpaca_response",
    "fill",
    "position",
    "outcome",
    "learning",
)
TRADE_ACTIONS = ("buy", "increase", "reduce", "close", "sell", "de_risk", "rebalance")


def _stage(name: str, happened: bool | None, summary: str, **detail: Any) -> dict[str, Any]:
    return {
        "stage": name,
        "status": "done" if happened else "none" if happened is False else "n/a",
        "summary": summary,
        "detail": detail,
    }


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
        thesis = (
            await s.scalars(
                select(BrainThesisRow).where(
                    (BrainThesisRow.entry_decision_id == d.id) | (BrainThesisRow.exit_decision_id == d.id)
                )
            )
        ).first()
        predictions = (
            await s.scalars(
                select(BrainPredictionRow).where(
                    BrainPredictionRow.cycle_id == d.cycle_id,
                    BrainPredictionRow.subject == d.subject,
                    BrainPredictionRow.source_type == "consensus",
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
            diag is not None,
            f"{diag['status']} quote ({diag.get('feed') or 'no feed'}, last trade "
            f"{diag.get('trade_age_s')}s old, spread {diag.get('spread_bps')}bp {diag.get('spread_source') or ''})"
            if diag
            else "no diagnosis recorded",
            diagnosis=diag,
            market=(dq.get("market") or {}).get("thesis"),
            feed_headline=(dq.get("feed") or {}).get("headline"),
        )
    )
    ran = [a for a in (cycle.agents if cycle is not None else []) or [] if a.get("status") == "ok"]
    stages.append(
        _stage(
            "agents",
            bool(opinions),
            f"{len({o.agent_id for o in opinions})} agent(s) gave a view on {d.subject}; {len(ran)} ran in the cycle",
            ran=[a["agent_id"] for a in ran],
        )
    )
    stages.append(
        _stage(
            "opinions",
            bool(opinions),
            ", ".join(f"{o.agent_id} {o.stance} {o.score:+.2f}" for o in opinions[:12]),
            opinions=[
                {
                    "agent": o.agent_id,
                    "stance": o.stance,
                    "score": o.score,
                    "confidence": o.confidence,
                    "thesis": o.thesis,
                    "invalidation": o.invalidation,
                    "data": o.data_quality,
                }
                for o in opinions
            ],
        )
    )
    stages.append(
        _stage(
            "consensus",
            consensus is not None,
            f"{consensus.stance} {consensus.score:+.2f}, confidence {consensus.confidence:.2f}, disagreement "
            f"{consensus.disagreement:.2f} ({consensus.supporting} for, {consensus.opposing} against)"
            + (" — unknown" if consensus.unknown else "")
            if consensus is not None
            else "no consensus recorded",
            reasons=consensus.reasons if consensus is not None else [],
            vetoes=consensus.vetoes if consensus is not None else [],
            detail=consensus.detail if consensus is not None else {},
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
    stages.append(
        _stage(
            "portfolio_decision",
            True,
            f"{d.action.upper()} {d.quantity or 0:g} {d.subject} ({d.status}): "
            + "; ".join((rationale.get("reasons") or [])[:3]),
            confidence=d.confidence,
            fit=rationale.get("fit"),
            entry=rationale.get("entry"),
            blocked_by=rationale.get("blocked_by"),
            memory=rationale.get("memory"),
            mode=d.mode,
        )
    )
    stages.append(
        _stage(
            "risk_check",
            bool(d.risk) or bool(ex.get("checks")),
            f"preview: {(d.risk or {}).get('summary') or 'not checked'}"
            + (f"; at execution: {ex.get('risk')}" if ex.get("risk") else ""),
            preview=d.risk,
            at_execution=ex.get("checks"),
        )
    )
    stages.append(
        _stage(
            "order",
            order is not None,
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
            bool(order is not None and order.alpaca_order_id),
            f"Alpaca order {order.alpaca_order_id}: {order.status}"
            + (f" — {order.error}" if order.error else "")
            if order is not None
            else "nothing was sent to Alpaca",
            events=[{"at": e.created_at.isoformat(), "kind": e.kind, "message": e.message} for e in events],
        )
    )
    filled = order is not None and order.filled_quantity > 0
    est = ex.get("est_price") or d.est_price
    fill_px = order.average_fill_price if order is not None else None
    stages.append(
        _stage(
            "fill",
            filled,
            f"filled {order.filled_quantity:g} @ ${fill_px or 0:,.2f}"
            + (f" at {order.filled_at:%H:%M:%S} UTC" if order.filled_at else "")
            if filled and order is not None
            else "no fill",
            est_price=est,
            slippage_bps=round((fill_px / est - 1) * 10_000, 1) if filled and fill_px and est else None,
        )
    )
    stages.append(
        _stage(
            "position",
            thesis is not None,
            f"{thesis.status} {thesis.origin} position: {thesis.qty:g} @ ${thesis.avg_price:,.2f}, check "
            f"{(thesis.check or {}).get('status', '—')}"
            if thesis is not None
            else "no position record (the thesis registry is kept while the Brain owns the account)",
            thesis_id=thesis.id if thesis is not None else None,
            thesis=thesis.thesis if thesis is not None else None,
            return_pct=thesis.return_pct if thesis is not None else None,
            benchmark_return=thesis.benchmark_return if thesis is not None else None,
            realized_pnl=thesis.realized_pnl if thesis is not None else None,
            exit_reason=thesis.exit_reason if thesis is not None else None,
        )
    )
    graded = [p for p in predictions if p.status == "evaluated"]
    outcome = d.outcome or {}
    stages.append(
        _stage(
            "outcome",
            bool(outcome) or bool(graded),
            (f"graded: {outcome}" if outcome else "")
            + "; ".join(
                f"{p.horizon_days}d {'hit' if p.hit else 'miss'} ({p.realized_relative:+.2%} vs benchmark)"
                for p in graded
                if p.realized_relative is not None
            )
            or f"not graded yet ({len(predictions)} prediction(s) open until their horizon)",
            predictions=[
                {
                    "horizon": p.horizon_days,
                    "status": p.status,
                    "hit": p.hit,
                    "due": p.due_date.isoformat(),
                    "relative": p.realized_relative,
                }
                for p in predictions
            ],
        )
    )
    stages.append(
        _stage(
            "learning",
            bool(reflections),
            "; ".join(
                f"{r.category}: decision {r.decision_quality}, outcome {r.outcome_quality}"
                for r in reflections
            )
            or "no reflection yet (written once the decision is graded)",
            lessons=[lesson for r in reflections for lesson in (r.lessons or [])],
        )
    )
    return {
        "decision_id": d.id,
        "cycle_id": d.cycle_id,
        "subject": d.subject,
        "action": d.action,
        "at": d.created_at.isoformat(),
        "stages": stages,
    }
