"""Lessons from completed trades (after each close): every position closed since the last pass becomes a
structured long-term memory — the thesis it was bought on, how it ended (a stop, a broken thesis, a
replacement, an overnight de-risk, closed outside the Brain), what it returned against the benchmark over
how many sessions, and how well it was executed (the ledger's grades for its orders). Outcome and execution
are recorded side by side and never merged: a good fill on a losing trade is still a good fill.

One trade proves nothing; these lessons are recalled as context and counted by the pattern consolidation
(:mod:`.patterns`), which calls a pattern *established* only when the sample is large enough.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from quantpulse.core.clock import Clock
from quantpulse.db.models import BrainExecutionRow, BrainThesisRow
from quantpulse.db.session import Database

from .memory import LONG_TERM, MemoryStore
from .store import BrainStore
from .theses import sessions_between

STATE_KEY = "trade_lessons"


def ended_by(reason: str | None) -> str:
    r = (reason or "").lower()
    if "stop" in r:
        return "stop"
    if "thesis broken" in r:
        return "thesis_broken"
    if "replace with" in r:
        return "replaced"
    if "overnight" in r:
        return "overnight_risk"
    if "outside the brain" in r:
        return "outside"
    if "take profit" in r or "target" in r:
        return "target"
    if "same bet" in r or "sector concentration" in r or "portfolio volatility" in r:
        return "portfolio_risk"
    return "decision"


def _belongs(e: BrainExecutionRow, t: BrainThesisRow) -> bool:
    """The position's orders: its entry, and every order on the symbol decided while it was open."""
    if e.client_order_id == t.entry_order_id:
        return True
    if e.symbol != t.symbol or e.decided_at is None or t.closed_at is None:
        return False
    return t.opened_at <= e.decided_at <= t.closed_at


async def learn_from_trades(db: Database, memory: MemoryStore, clock: Clock) -> dict[str, Any]:
    store = BrainStore(db)
    now = clock.now()
    state = await store.get_state(STATE_KEY) or {}
    since = datetime.fromisoformat(state["at"]) if state.get("at") else None
    async with db.session() as s:
        q = select(BrainThesisRow).where(BrainThesisRow.status == "closed")
        if since is not None:
            q = q.where(BrainThesisRow.closed_at > since)
        closed = (await s.scalars(q)).all()
        symbols = {t.symbol for t in closed}
        executions = (
            (await s.scalars(select(BrainExecutionRow).where(BrainExecutionRow.symbol.in_(symbols)))).all()
            if symbols
            else []
        )
    lessons = 0
    for t in closed:
        ret = (t.exit_price / t.entry_price - 1) if t.exit_price and t.entry_price else None
        rel = ret - t.benchmark_return if ret is not None and t.benchmark_return is not None else None
        label = "unknown" if rel is None else "beat the benchmark" if rel > 0 else "lagged the benchmark"
        how = ended_by(t.exit_reason)
        mine = [e for e in executions if _belongs(e, t)]
        grades = [e.grade for e in mine if e.grade]
        held = sessions_between(t.opened_at, t.closed_at or now)
        summary = (
            f"{t.symbol}: {t.origin} position ({t.thesis[:90]}) {label}"
            + (f" ({ret:+.1%}, {rel:+.1%} vs the benchmark)" if ret is not None and rel is not None else "")
            + f" over {held} session(s); ended by {how.replace('_', ' ')}"
            + (f"; execution {', '.join(grades)}" if grades else "; execution not measured")
        )
        await memory.remember(
            LONG_TERM,
            "trade_outcome",
            t.symbol,
            summary,
            now,
            key=f"trade_outcome:{t.id}",
            data={
                "thesis_id": t.id,
                "origin": t.origin,
                "return": None if ret is None else round(ret, 5),
                "relative": None if rel is None else round(rel, 5),
                "sessions": held,
                "ended_by": how,
                "exit_reason": t.exit_reason,
                "realized_pnl": t.realized_pnl,
                "supporting": t.supporting,
                "opposing": t.opposing,
                "regime": t.regime,
                "sector": t.sector,
                "execution_grades": grades,
                "slippage_bps": [e.slippage_bps for e in mine if e.slippage_bps is not None],
            },
            tags=["trade_outcome", how, label.replace(" ", "_"), t.origin],
            importance=0.7,
        )
        lessons += 1
    await store.set_state(STATE_KEY, {"at": now.isoformat(), "last": lessons}, now)
    return {"lessons": lessons}
