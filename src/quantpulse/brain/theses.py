"""Position theses: why the Brain holds each position on the Alpaca paper account, and whether it still should.

When the Brain owns the account (``paper_execution``) every position has one thesis per holding period
(``brain_theses``). Alpaca's positions are authoritative; :meth:`ThesisBook.sync` reconciles the registry
with them at the start of every decision:

* a position the Brain bought (a filled Brain order, found in the order records — so a fill after the
  cycle, or after a restart, is never lost) gets the thesis recorded with that decision: why, what would
  prove it wrong, the stop, the expected return and target (only when the consensus is calibrated —
  nothing is invented), the horizon, the confidence, the agents for and against, the regime and sector,
  and the benchmark's price at entry;
* positions already on the account when the Brain first took it over are **inherited**: managed like the
  rest, with a thesis that says it was never the Brain's idea;
* any other position is **unexpected** (bought by hand or by another tool): new positions stop until it
  is adopted (``POST /brain/positions/{symbol}/adopt``) or gone — the Brain still manages it meanwhile;
* a thesis whose position is gone is closed, with the Brain's exit (price, reason, decision) or "closed
  outside the Brain".

Every open thesis is marked (price, value, weight, P&L, return against the benchmark since entry) and
checked (:func:`check`): **broken** below its stop, when most of the agents that supported it now oppose
it, when the consensus has turned confidently bearish, or when it has not worked in twice its horizon —
a broken thesis is exited without waiting for a new signal (a protective exit); **weakening** when the
evidence has faded (no clear consensus, past its horizon, behind the benchmark) — first in line to be
replaced by a stronger idea; otherwise **intact**.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import sessions_between
from quantpulse.db.models import BrainDecisionRow, BrainThesisRow, BrokerOrderRow
from quantpulse.db.session import Database
from quantpulse.services.order_manager import BRAIN

from .consensus import Consensus
from .context import BrainContext
from .decisions import Proposal
from .types import BUYING, Stance

logger = logging.getLogger(__name__)

STATE_KEY = "theses"  # brain_state: when the Brain first took the account over
INHERITED_HORIZON = 21  # sessions: a position with no recorded idea is reviewed within a month
EXPIRY_MULTIPLE = 2.0  # a thesis that has not worked in twice its horizon is broken


def entry_plan(ctx: BrainContext, p: Proposal, expected: float | None) -> dict[str, Any]:
    """The thesis a buy is made on, recorded with the decision (the registry reads it once it fills)."""
    c = p.consensus
    votes = c.votes if c is not None else []
    price = float(p.est_price or 0.0)
    stop = round(price * (1 - ctx.limits.max_position_loss_pct), 4) if price else None
    levels = [
        o.invalidation for o in ctx.working.opinions.get(p.subject, []) if o.invalidation and o.directional
    ]
    invalidation = "; ".join(levels[:3])[:400] if levels else None
    return {
        "thesis": "; ".join(p.reasons)[:500],
        "invalidation": invalidation,
        "stop_price": stop,
        "expected_return": expected,
        "target_price": round(price * (1 + expected), 4) if expected and expected > 0 and price else None,
        "horizon_days": int((ctx.working.facts.get("horizons") or {}).get(p.subject, 5)),
        "confidence": round(p.confidence, 4),
        "supporting": sorted({v.agent_id for v in votes if v.stance is Stance.BULLISH}),
        "opposing": sorted({v.agent_id for v in votes if v.stance is Stance.BEARISH}),
        "regime": ctx.regime.label if ctx.regime else None,
        "sector": ctx.sectors.get(p.subject),
        "benchmark_price": ctx.price(ctx.benchmark_symbol),
    }


def attach_entries(
    ctx: BrainContext, proposals: Sequence[Proposal], expected: dict[str, float | None]
) -> None:
    for p in proposals:
        if p.action in BUYING and p.is_trade:
            p.entry = entry_plan(ctx, p, expected.get(p.subject))


def check(
    row: BrainThesisRow, ctx: BrainContext, consensus: Consensus | None, min_confidence: float, now: datetime
) -> dict[str, Any]:
    """Is the idea still true? ``broken`` (exit), ``weakening`` (first to be replaced) or ``intact``."""
    broken: list[str] = []
    weak: list[str] = []
    price = row.last_price
    if price is not None and row.stop_price is not None and price <= row.stop_price:
        broken.append(f"below its stop (${price:,.2f} ≤ ${row.stop_price:,.2f})")
    views = {o.agent_id: o for o in ctx.working.opinions.get(row.symbol, []) if o.directional}
    backers = [a for a in row.supporting or [] if a in views]
    turned = [a for a in backers if views[a].stance is Stance.BEARISH]
    if len(turned) >= 2 and len(turned) * 2 >= len(backers):
        broken.append(
            f"{len(turned)} of the {len(backers)} agents that supported it now oppose it ({', '.join(turned)})"
        )
    if consensus is not None and not consensus.unknown:
        if (
            consensus.stance is Stance.BEARISH
            and consensus.actionable_view
            and consensus.confidence >= min_confidence
        ):
            broken.append(
                f"the consensus has turned bearish ({consensus.score:+.2f}, confidence {consensus.confidence:.2f})"
            )
        elif consensus.stance is not Stance.BULLISH:
            weak.append(f"no bullish consensus any more ({consensus.stance.value})")
    else:
        weak.append("the evidence no longer supports a view")
    held = sessions_between(row.opened_at, now)
    horizon = row.horizon_days or INHERITED_HORIZON
    rel = None
    if row.return_pct is not None and row.benchmark_return is not None:
        rel = row.return_pct - row.benchmark_return
    if held > EXPIRY_MULTIPLE * horizon and rel is not None and rel < 0:
        broken.append(
            f"it has not worked in twice its {horizon}-session horizon ({held} sessions, {rel:+.1%} vs the benchmark)"
        )
    elif held > horizon:
        weak.append(f"past its {horizon}-session horizon ({held} sessions)")
    if rel is not None and rel < 0 and not broken:
        weak.append(f"behind the benchmark since entry ({rel:+.1%})")
    status = "broken" if broken else "weakening" if weak else "intact"
    return {"status": status, "reasons": broken or weak, "sessions_held": held, "relative_return": rel, "at": now.isoformat()}  # fmt: skip


def view(row: BrainThesisRow) -> dict[str, Any]:
    rel = (
        round(row.return_pct - row.benchmark_return, 5)
        if row.return_pct is not None and row.benchmark_return is not None
        else None
    )
    return {
        "id": row.id,
        "symbol": row.symbol,
        "status": row.status,
        "origin": row.origin,
        "opened_at": row.opened_at.isoformat(),
        "closed_at": row.closed_at.isoformat() if row.closed_at else None,
        "thesis": row.thesis,
        "invalidation": row.invalidation,
        "entry_price": row.entry_price,
        "entry_qty": row.entry_qty,
        "qty": row.qty,
        "avg_price": row.avg_price,
        "last_price": row.last_price,
        "market_value": row.market_value,
        "weight": row.weight,
        "unrealized_pnl": row.unrealized_pnl,
        "return_pct": row.return_pct,
        "benchmark_return": row.benchmark_return,
        "relative_return": rel,
        "stop_price": row.stop_price,
        "target_price": row.target_price,
        "expected_return": row.expected_return,
        "horizon_days": row.horizon_days,
        "confidence": row.confidence,
        "supporting": row.supporting,
        "opposing": row.opposing,
        "regime": row.regime,
        "sector": row.sector,
        "check": row.check,
        "entry_decision_id": row.entry_decision_id,
        "entry_order_id": row.entry_order_id,
        "exit_price": row.exit_price,
        "exit_reason": row.exit_reason,
        "exit_decision_id": row.exit_decision_id,
        "realized_pnl": row.realized_pnl,
        "history": row.history,
    }


def _live(v: dict[str, Any], marks: dict[str, Any]) -> dict[str, Any]:
    """A thesis view at Alpaca's prices now (``marks``: Alpaca's live positions by symbol). The benchmark
    comparison stays as of the last cycle."""
    pos = marks.get(v["symbol"])
    if pos is None:
        return {**v, "live": False, "note": "no longer held on Alpaca: closed at the next cycle"}
    entry = v.get("entry_price")
    return {
        **v,
        "live": True,
        "qty": pos.qty,
        "last_price": pos.current_price,
        "market_value": pos.market_value,
        "weight": round(pos.weight, 5),
        "unrealized_pnl": pos.unrealized_pl,
        "intraday_pnl": pos.intraday_pl,
        "return_pct": round(pos.current_price / entry - 1, 5) if entry else v.get("return_pct"),
    }


class ThesisBook:
    def __init__(self, settings: Settings, db: Database, clock: Clock) -> None:
        self._s = settings
        self._db = db
        self._clock = clock

    # ------------------------------------------------------------------ reads
    async def open_rows(self) -> dict[str, BrainThesisRow]:
        async with self._db.session() as s:
            rows = (await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "open"))).all()
        return {r.symbol: r for r in rows}

    async def positions(self, closed: int = 50, live: Sequence[Any] | None = None) -> dict[str, Any]:
        """Open and recently closed theses. With ``live`` (Alpaca's positions, read now), the open ones are
        shown at Alpaca's current prices; otherwise as marked at the Brain's last cycle (``marked``)."""
        async with self._db.session() as s:
            opened = (
                await s.scalars(
                    select(BrainThesisRow)
                    .where(BrainThesisRow.status == "open")
                    .order_by(BrainThesisRow.symbol)
                )
            ).all()
            done = (
                await s.scalars(
                    select(BrainThesisRow)
                    .where(BrainThesisRow.status == "closed")
                    .order_by(BrainThesisRow.closed_at.desc())
                    .limit(closed)
                )
            ).all()
        from .store import BrainStore

        state = await BrainStore(self._db).get_state(STATE_KEY) or {}
        marks = {p.symbol: p for p in live} if live is not None else None
        return {
            "owner": "the Brain"
            if self._s.brain_owns_account
            else "the trading strategy (theses are not kept)",
            "took_over_at": state.get("took_over_at"),
            "unexpected": state.get("unexpected") or [],
            "marked": "live" if live is not None else "at the last cycle",
            "open": [_live(view(r), marks) if marks is not None else view(r) for r in opened],
            "closed": [view(r) for r in done],
        }

    # ------------------------------------------------------------------ sync with Alpaca
    async def _brain_buy(
        self, symbol: str, after: datetime | None
    ) -> tuple[BrokerOrderRow, BrainDecisionRow | None] | None:
        """The latest filled Brain buy of ``symbol`` (after ``after``) and the decision that made it."""
        async with self._db.session() as s:
            q = (
                select(BrokerOrderRow)
                .where(
                    BrokerOrderRow.strategy == BRAIN,
                    BrokerOrderRow.symbol == symbol,
                    BrokerOrderRow.side == "buy",
                    BrokerOrderRow.filled_quantity > 0,
                )
                .order_by(BrokerOrderRow.created_at.desc())
                .limit(1)
            )
            order = (await s.scalars(q)).first()
            if order is None or (after is not None and order.created_at <= after):
                return None
            decisions = (
                await s.scalars(
                    select(BrainDecisionRow)
                    .where(
                        BrainDecisionRow.subject == symbol, BrainDecisionRow.action.in_(["buy", "increase"])
                    )
                    .order_by(BrainDecisionRow.id.desc())
                    .limit(50)
                )
            ).all()
        decision = next(
            (d for d in decisions if (d.execution or {}).get("client_order_id") == order.client_order_id),
            None,
        )
        return order, decision

    async def _brain_sell(
        self, symbol: str, since: datetime
    ) -> tuple[BrokerOrderRow, BrainDecisionRow | None] | None:
        async with self._db.session() as s:
            order = (
                await s.scalars(
                    select(BrokerOrderRow)
                    .where(
                        BrokerOrderRow.symbol == symbol,
                        BrokerOrderRow.side == "sell",
                        BrokerOrderRow.filled_quantity > 0,
                        BrokerOrderRow.created_at >= since,
                    )
                    .order_by(BrokerOrderRow.created_at.desc())
                    .limit(1)
                )
            ).first()
            if order is None:
                return None
            decisions = (
                await s.scalars(
                    select(BrainDecisionRow)
                    .where(BrainDecisionRow.subject == symbol)
                    .order_by(BrainDecisionRow.id.desc())
                    .limit(50)
                )
            ).all()
        decision = next(
            (d for d in decisions if (d.execution or {}).get("client_order_id") == order.client_order_id),
            None,
        )
        return order, decision

    async def _last_closed(self, symbol: str) -> datetime | None:
        async with self._db.session() as s:
            row = (
                await s.scalars(
                    select(BrainThesisRow)
                    .where(BrainThesisRow.symbol == symbol, BrainThesisRow.status == "closed")
                    .order_by(BrainThesisRow.closed_at.desc())
                    .limit(1)
                )
            ).first()
        return row.closed_at if row is not None else None

    async def sync(self, ctx: BrainContext, cycle_id: int | None = None) -> dict[str, Any]:
        """Bring the registry in line with Alpaca's positions; post ``theses`` and ``unexpected_positions``
        to working memory. Only while the Brain owns the account (its positions are then the portfolio)."""
        from .store import BrainStore

        store = BrainStore(self._db)
        now = self._clock.now()
        state = await store.get_state(STATE_KEY) or {}
        first = not state.get("took_over_at")
        positions = {s: p for s, p in ctx.account.positions.items() if p.qty > 0}
        open_rows = await self.open_rows()
        created: list[str] = []
        closed: list[str] = []
        unexpected: list[str] = []
        adopted = set(state.get("adopt") or [])
        equity = ctx.account.account.equity if ctx.account.account is not None else 0.0
        bench = ctx.price(ctx.benchmark_symbol)
        async with self._db.session() as s:
            for sym, pos in positions.items():
                row = open_rows.get(sym)
                if row is not None:
                    row = await s.get(BrainThesisRow, row.id)
                if row is None:
                    found = await self._brain_buy(sym, await self._last_closed(sym))
                    if found is not None:
                        order, decision = found
                        plan = ((decision.rationale or {}).get("entry") or {}) if decision is not None else {}
                        row = BrainThesisRow(
                            symbol=sym,
                            status="open",
                            origin="brain",
                            opened_at=order.filled_at or order.submitted_at or order.created_at,
                            entry_price=order.average_fill_price or pos.avg_entry_price,
                            entry_qty=order.filled_quantity,
                            entry_decision_id=decision.id if decision is not None else None,
                            entry_order_id=order.client_order_id,
                            thesis=plan.get("thesis") or (order.reason or "bought by the Brain"),
                            invalidation=plan.get("invalidation"),
                            stop_price=plan.get("stop_price")
                            or round(pos.avg_entry_price * (1 - ctx.limits.max_position_loss_pct), 4),
                            target_price=plan.get("target_price"),
                            expected_return=plan.get("expected_return"),
                            horizon_days=plan.get("horizon_days"),
                            confidence=plan.get("confidence"),
                            supporting=plan.get("supporting") or [],
                            opposing=plan.get("opposing") or [],
                            regime=plan.get("regime"),
                            sector=plan.get("sector") or ctx.sectors.get(sym),
                            benchmark_entry=plan.get("benchmark_price") or bench,
                        )
                    elif first or sym in adopted:
                        row = BrainThesisRow(
                            symbol=sym,
                            status="open",
                            origin="inherited" if first else "adopted",
                            opened_at=now,
                            entry_price=pos.avg_entry_price,
                            entry_qty=pos.qty,
                            thesis=(
                                "held when the Brain took the account over: no recorded idea; managed like any "
                                "holding and reviewed within its horizon"
                                if first
                                else "adopted by hand: bought outside the Brain, now managed by it"
                            ),
                            stop_price=round(pos.avg_entry_price * (1 - ctx.limits.max_position_loss_pct), 4),
                            horizon_days=INHERITED_HORIZON,
                            supporting=[],
                            opposing=[],
                            regime=ctx.regime.label if ctx.regime else None,
                            sector=ctx.sectors.get(sym),
                            benchmark_entry=bench,
                        )
                    else:
                        unexpected.append(sym)
                        continue
                    row.qty, row.avg_price = pos.qty, pos.avg_entry_price
                    row.updated_at = now
                    row.check, row.history = {}, [{"at": now.isoformat(), "event": "opened", "qty": pos.qty, "origin": row.origin}]  # fmt: skip
                    s.add(row)
                    created.append(sym)
                elif abs(row.qty - pos.qty) > 1e-9:
                    row.history = [*(row.history or []), {"at": now.isoformat(), "event": "size", "from": row.qty, "to": pos.qty}]  # fmt: skip
                self._mark(row, pos, equity, bench, now)
            for sym, stale in open_rows.items():
                if sym in positions:
                    continue
                row = await s.get(BrainThesisRow, stale.id)
                assert row is not None
                exit_ = await self._brain_sell(sym, row.opened_at)
                row.status, row.closed_at, row.updated_at = "closed", now, now
                if exit_ is not None:
                    order, decision = exit_
                    row.exit_price = order.average_fill_price
                    row.exit_decision_id = decision.id if decision is not None else None
                    row.exit_reason = (
                        "; ".join(((decision.rationale or {}).get("reasons") or [])[:3])
                        if decision is not None
                        else f"sold ({order.strategy} order {order.client_order_id})"
                    )
                else:
                    row.exit_reason = "closed outside the Brain (by hand or another tool)"
                    row.exit_price = row.last_price
                if row.exit_price is not None:
                    row.realized_pnl = round((row.exit_price - row.avg_price) * row.qty, 2)
                row.history = [*(row.history or []), {"at": now.isoformat(), "event": "closed", "reason": row.exit_reason}]  # fmt: skip
                closed.append(sym)
        state = {
            **state,
            "took_over_at": state.get("took_over_at") or now.isoformat(),
            "unexpected": unexpected,
            "adopt": [a for a in adopted if a in positions and a not in created],
        }
        await store.set_state(STATE_KEY, state, now)
        rows = await self.open_rows()
        ctx.working.post("theses", {sym: view(r) for sym, r in rows.items()})
        ctx.working.post("unexpected_positions", unexpected)
        return {"opened": created, "closed": closed, "unexpected": unexpected, "open": len(rows)}

    @staticmethod
    def _mark(row: BrainThesisRow, pos: Any, equity: float, bench: float | None, now: datetime) -> None:
        row.qty, row.avg_price = pos.qty, pos.avg_entry_price
        row.last_price = pos.current_price
        row.market_value = pos.market_value
        row.weight = round(pos.market_value / equity, 5) if equity > 0 else None
        row.unrealized_pnl = pos.unrealized_pl
        row.return_pct = round(pos.current_price / row.entry_price - 1, 5) if row.entry_price else None
        row.benchmark_return = (
            round(bench / row.benchmark_entry - 1, 5) if bench and row.benchmark_entry else None
        )
        row.updated_at = now

    async def review(
        self, ctx: BrainContext, consensus: dict[str, Consensus], min_confidence: float
    ) -> dict[str, dict[str, Any]]:
        """Check every open thesis; post ``thesis_checks`` to working memory (the plan acts on them)."""
        now = self._clock.now()
        out: dict[str, dict[str, Any]] = {}
        async with self._db.session() as s:
            rows = (await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "open"))).all()
            for row in rows:
                result = check(row, ctx, consensus.get(row.symbol), min_confidence, now)
                if (row.check or {}).get("status") != result["status"]:
                    row.history = [*(row.history or []), {"at": now.isoformat(), "event": result["status"], "reasons": result["reasons"]}]  # fmt: skip
                row.check = result
                out[row.symbol] = result
        ctx.working.post("thesis_checks", out)
        return out

    async def adopt(self, symbol: str) -> dict[str, Any]:
        """A person says an unexpected position is fine: the next cycle gives it an ``adopted`` thesis."""
        from .store import BrainStore

        store = BrainStore(self._db)
        state = await store.get_state(STATE_KEY) or {}
        symbol = symbol.upper()
        adopt = sorted({*(state.get("adopt") or []), symbol})
        await store.set_state(STATE_KEY, {**state, "adopt": adopt}, self._clock.now())
        return {"symbol": symbol, "adopt": adopt, "note": "adopted at the next Brain cycle"}
