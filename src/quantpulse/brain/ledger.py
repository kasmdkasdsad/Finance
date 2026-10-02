"""The execution ledger: every Brain order from the decision to its final state (``brain_executions``).

For each order it keeps the decision (proposal id, Brain cycle, action, reason, the consensus behind it),
what the decision expected to pay, what was sent (order type, limit), the market as the order left (the
quote the risk engine judged it by: price, bid/ask, spread, age, source), how long Alpaca took to answer
and to fill, partial fills, the final status, and two measures of execution quality:

* ``slippage_bps`` — the fill against the price the decision assumed (positive: we paid more or received
  less);
* ``cost_vs_quote_bps`` — the fill against the midpoint of the quote as the order left, the cost of
  crossing the spread and of moving prices; ``grade`` compares it with half the spread (good within
  half the spread + 2bp, fair within + 10bp, poor beyond, unknown without a quote).

The grade is about *execution only* — whether the trade made money is judged elsewhere (the scorecard). The
ledger is refreshed from the order records (which reconciliation keeps in step with Alpaca), so a fill that
comes after the cycle, or after a restart, is never lost.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select

from quantpulse.core.clock import Clock
from quantpulse.db.models import BrainExecutionRow, BrokerOrderRow
from quantpulse.db.session import Database
from quantpulse.services.order_manager import FINAL

from .decisions import Proposal
from .types import SELLING

GOOD_EXTRA_BPS = 2.0
FAIR_EXTRA_BPS = 10.0


def grade(cost_bps: float | None, spread_bps: float | None) -> str:
    if cost_bps is None or spread_bps is None:
        return "unknown"
    half = spread_bps / 2
    if cost_bps <= half + GOOD_EXTRA_BPS:
        return "good"
    if cost_bps <= half + FAIR_EXTRA_BPS:
        return "fair"
    return "poor"


def _signed_bps(side: str, fill: float, ref: float | None) -> float | None:
    if not ref or ref <= 0 or not fill:
        return None
    sign = 1 if side == "buy" else -1
    return round(sign * (fill / ref - 1) * 10_000, 2)


def view(r: BrainExecutionRow) -> dict[str, Any]:
    cols = (
        "client_order_id", "alpaca_order_id", "decision_id", "brain_cycle_id", "trading_cycle_id", "symbol",
        "side", "action", "reason", "consensus", "qty", "order_type", "expected_price", "submitted_price",
        "quote_price", "quote_bid", "quote_ask", "spread_bps", "quote_age_s", "quote_source",
        "submit_latency_ms", "decision_to_submit_s", "filled_qty", "filled_avg_price", "seconds_to_fill",
        "partial", "status", "final", "slippage_bps", "cost_vs_quote_bps", "grade",
    )  # fmt: skip
    out = {c: getattr(r, c) for c in cols}
    for c in ("decided_at", "submitted_at", "filled_at", "updated_at"):
        v = getattr(r, c)
        out[c] = v.isoformat() if v is not None else None
    return out


class ExecutionLedger:
    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    async def record(self, cycle_id: int, decision_ids: dict[str, int], proposals: Sequence[Proposal]) -> int:
        """A row for every order this cycle tried to send — acknowledged, refused by Alpaca, or of unknown
        fate; returns how many."""
        now = self._clock.now()
        n = 0
        async with self._db.session() as s:
            for p in proposals:
                ex = p.execution or {}
                cid = ex.get("client_order_id")
                attempted = ex.get("sent") or ex.get("stage") in ("rejected", "failed", "unknown")
                if not cid or not attempted or ex.get("duplicate_prevented"):
                    continue  # never tried (risk-rejected, halted, dry run) or already recorded
                if (
                    await s.scalars(select(BrainExecutionRow).where(BrainExecutionRow.client_order_id == cid))
                ).first():
                    continue
                c = p.consensus
                submitted = datetime.fromisoformat(ex["submitted_at"]) if ex.get("submitted_at") else None
                s.add(
                    BrainExecutionRow(
                        client_order_id=cid,
                        alpaca_order_id=ex.get("alpaca_order_id"),
                        decision_id=decision_ids.get(p.subject),
                        brain_cycle_id=cycle_id,
                        trading_cycle_id=ex.get("trading_cycle_id"),
                        symbol=p.subject,
                        side="sell" if p.action in SELLING else "buy",
                        action=p.action.value,
                        reason="; ".join(p.reasons)[:1000],
                        consensus={
                            "stance": c.stance.value,
                            "score": round(c.score, 4),
                            "confidence": round(c.confidence, 4),
                            "sources": c.sources,
                        }
                        if c is not None
                        else {},
                        qty=float(ex.get("qty") or p.quantity or 0.0),
                        order_type=ex.get("order_type"),
                        expected_price=p.est_price,
                        submitted_price=ex.get("limit_price") or ex.get("quote_price"),
                        quote_price=ex.get("quote_price"),
                        quote_bid=ex.get("quote_bid"),
                        quote_ask=ex.get("quote_ask"),
                        spread_bps=ex.get("quote_spread_bps"),
                        quote_age_s=ex.get("quote_age_seconds"),
                        quote_source=(ex.get("quote_source") or "")[:96] or None,
                        decided_at=now,
                        submitted_at=submitted,
                        submit_latency_ms=ex.get("submit_latency_ms"),
                        decision_to_submit_s=round((submitted - now).total_seconds(), 3)
                        if submitted
                        else None,
                        filled_qty=float(ex.get("filled_qty") or 0.0),
                        filled_avg_price=ex.get("filled_avg_price"),
                        status=str(ex.get("status") or "submitted"),
                        updated_at=now,
                    )
                )
                n += 1
        await self.refresh()
        return n

    async def refresh(self) -> int:
        """Bring every unfinished row in line with its order record (Alpaca's, via reconciliation)."""
        now = self._clock.now()
        changed = 0
        async with self._db.session() as s:
            rows = (
                await s.scalars(select(BrainExecutionRow).where(BrainExecutionRow.final.is_(False)))
            ).all()
            for r in rows:
                o = (
                    await s.scalars(
                        select(BrokerOrderRow).where(BrokerOrderRow.client_order_id == r.client_order_id)
                    )
                ).first()
                if o is None:
                    continue
                r.alpaca_order_id = o.alpaca_order_id or r.alpaca_order_id
                r.status = o.status
                r.filled_qty = o.filled_quantity
                r.filled_avg_price = o.average_fill_price
                r.filled_at = o.filled_at
                r.submitted_at = o.submitted_at or r.submitted_at
                if r.submitted_at and r.decided_at and r.decision_to_submit_s is None:
                    r.decision_to_submit_s = round((r.submitted_at - r.decided_at).total_seconds(), 3)
                if o.filled_at and o.submitted_at:
                    r.seconds_to_fill = round((o.filled_at - o.submitted_at).total_seconds(), 3)
                r.partial = 0 < o.filled_quantity < r.qty - 1e-9
                if o.average_fill_price:
                    r.slippage_bps = _signed_bps(r.side, o.average_fill_price, r.expected_price)
                    mid = (r.quote_bid + r.quote_ask) / 2 if r.quote_bid and r.quote_ask else r.quote_price
                    r.cost_vs_quote_bps = _signed_bps(r.side, o.average_fill_price, mid)
                    r.grade = grade(r.cost_vs_quote_bps, r.spread_bps)
                r.final = o.status in FINAL
                r.updated_at = now
                changed += 1
        return changed

    async def rows(self, limit: int = 100, since: datetime | None = None) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            q = select(BrainExecutionRow).order_by(BrainExecutionRow.id.desc()).limit(limit)
            if since is not None:
                q = q.where(BrainExecutionRow.decided_at >= since)
            return [view(r) for r in (await s.scalars(q)).all()]
