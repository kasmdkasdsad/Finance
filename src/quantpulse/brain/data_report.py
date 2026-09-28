"""The market-data report: how often data quality stopped the Brain, what it stopped, and what real-time SIP
data would (and would not) change — for a person to decide whether to pay for it.

Read from what the cycles recorded (the focus symbols' quote diagnoses, the market-level data view, entry
halts), the decisions (blocked by a data veto, halted, or refused by the risk engine for quote age or
spread), the opportunities stopped at the data stage, and the agents' opinions made on data that was not
executable. The "expected benefit" is a count of what would have passed the data checks, never a claim
about returns: whether those trades would have made money is unknown. QuantPulse never buys a
subscription.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import median
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import BrainCycleRow, BrainDecisionRow, BrainOpinionRow, BrainOpportunityRow
from quantpulse.db.session import Database

from .execution import DATA_BLOCKED

TRADE_ACTIONS = ("buy", "increase", "reduce", "close", "sell", "de_risk", "rebalance")
QUIET = ("stale", "no_trade_today")
NOT_EXECUTABLE = ("stale", "unavailable", "provider_error", "invalid")


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q * len(v)))], 1)


def _failed(checks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [c for c in checks or [] if not c.get("passed", True)]


def _data_reason(d: BrainDecisionRow) -> str | None:
    """Why a trade decision was stopped by market data (``None``: it was not)."""
    ex = d.execution or {}
    rationale = d.rationale or {}
    if DATA_BLOCKED in str(ex.get("reason") or ""):
        return "halted: data quality insufficient"
    if any("market data" in b or "stale" in b or "quote" in b for b in rationale.get("blocked_by") or []):
        return "blocked: the symbol's quote was not executable"
    for c in _failed((d.risk or {}).get("checks")) + _failed(ex.get("checks")):
        if c.get("name") == "live_data":
            return "risk engine: quote too old or not live"
        if c.get("name") == "liquidity" and "spread" in str(c.get("detail")):
            return "risk engine: spread too wide or not measurable"
    return None


async def data_report(db: Database, settings: Settings, since: datetime, now: datetime) -> dict[str, Any]:
    async with db.session() as s:
        cycles = (
            await s.scalars(
                select(BrainCycleRow).where(
                    BrainCycleRow.started_at >= since, BrainCycleRow.status == "completed"
                )
            )
        ).all()
        ids = [c.id for c in cycles]
        decisions = (
            (
                await s.scalars(
                    select(BrainDecisionRow).where(
                        BrainDecisionRow.cycle_id.in_(ids),
                        BrainDecisionRow.action.in_(TRADE_ACTIONS),
                        BrainDecisionRow.quantity.is_not(None),
                    )
                )
            ).all()
            if ids
            else []
        )
        opportunities = (
            (await s.scalars(select(BrainOpportunityRow).where(BrainOpportunityRow.cycle_id.in_(ids)))).all()
            if ids
            else []
        )
        stale_opinions = (
            (
                await s.scalars(
                    select(BrainOpinionRow).where(
                        BrainOpinionRow.cycle_id.in_(ids), BrainOpinionRow.data_quality.in_(NOT_EXECUTABLE)
                    )
                )
            ).all()
            if ids
            else []
        )
    session = [c for c in cycles if (c.market or {}).get("open")]
    blocked = [c for c in session if "data_quality" in ((c.summary or {}).get("entry_halts") or [])]
    statuses: Counter[str] = Counter()
    feeds: Counter[str] = Counter()
    iex_ages: list[float] = []
    quiet_iex = spread_problems = wide = 0
    headlines: Counter[str] = Counter()
    for c in session:
        dq = c.data_quality or {}
        feed = dq.get("feed") or {}
        if feed.get("headline"):
            headlines[str(feed["headline"])[:160]] += 1
        for diag in (dq.get("diagnosis") or {}).values():
            statuses[diag.get("status")] += 1
            feeds[diag.get("feed") or "none"] += 1
            if diag.get("feed") == "iex":
                if diag.get("trade_age_s") is not None:
                    iex_ages.append(float(diag["trade_age_s"]))
                if diag.get("status") in QUIET:
                    quiet_iex += 1
            if diag.get("spread_bps") is None and diag.get("status") not in ("market_closed", "holiday"):
                spread_problems += 1
            elif (diag.get("spread_bps") or 0) > settings.trading_max_spread_bps and "IEX" in str(
                diag.get("spread_source") or ""
            ):
                wide += 1
    quotes = sum(statuses.values())
    stopped = [(d, _data_reason(d)) for d in decisions]
    stopped = [(d, why) for d, why in stopped if why]
    reasons = Counter(why for _, why in stopped)
    rejected_opps = [o for o in opportunities if o.status == "rejected_data"]
    agents = Counter(o.agent_id for o in stale_opinions)
    trade_decisions = len(decisions)
    return {
        "window": {"since": since.isoformat(), "until": now.isoformat()},
        "headline": (
            f"{len(blocked)} of {len(session)} in-session cycles had new positions halted ({DATA_BLOCKED}); "
            f"{len(stopped)} of {trade_decisions} trade decisions were stopped by market data"
            if session
            else "no in-session cycles recorded in this window"
        ),
        "how_often": {
            "cycles_in_session": len(session),
            "data_blocked_cycles": len(blocked),
            "share": round(len(blocked) / len(session), 3) if session else None,
            "by_day": _by_day(session, blocked),
        },
        "functions_affected": {
            "new_positions_halted_cycles": len(blocked),
            "trade_decisions_stopped": dict(reasons),
            "opportunities_stopped_at_data": len(rejected_opps),
            "opportunity_kinds_stopped": dict(Counter(o.kind for o in rejected_opps)),
            "agents_on_non_executable_data": dict(agents.most_common()),
        },
        "rejected_opportunities": [
            {"kind": o.kind, "subject": o.subject, "headline": o.headline, "at": o.created_at.isoformat()}
            for o in rejected_opps[-20:]
        ],
        "quote_age": {
            "focus_quotes": quotes,
            "statuses": dict(statuses),
            "feeds": dict(feeds),
            "iex_quiet": quiet_iex,
            "iex_trade_age_median_s": round(median(iex_ages), 1) if iex_ages else None,
            "iex_trade_age_p90_s": _pct(iex_ages, 0.9),
            "limit_s": settings.trading_max_quote_age_seconds,
        },
        "spread": {
            "unmeasurable": spread_problems,
            "wider_than_limit_on_iex": wide,
            "limit_bps": settings.trading_max_spread_bps,
        },
        "top_causes": [{"cause": k, "cycles": v} for k, v in headlines.most_common(5)],
        "sip": {
            "configured_feed": settings.alpaca_stock_feed,
            "limitation": "Alpaca's free feed is IEX: one exchange with a few percent of US volume. Its last "
            "trade can be minutes old while the stock trades elsewhere, and its book can be far wider than the "
            "market. Those quotes are treated as stale or unmeasurable by design.",
            "would_solve": [
                f"{quiet_iex} IEX quotes with no recent print (stale) in the focus set",
                f"{spread_problems + wide} spreads that could not be measured or were wide on IEX's own book",
                f"up to {len(stopped)} trade decisions stopped for quote age or spread, and {len(rejected_opps)} "
                "opportunities stopped at the data stage",
                f"{len(blocked)} cycles with new positions halted for data quality (where staleness was the cause)",
            ],
            "would_not_solve": [
                "the market being closed, holidays and early closes",
                "vendor outages and failed requests",
                "a system clock that is off (every quote age is off by as much)",
                "names that genuinely do not trade",
            ],
            "expected_benefit": "a count, not a forecast of returns: the decisions and opportunities above would "
            "have passed the data checks more often. Whether they would have made money is unknown and would "
            "have to be measured after the change.",
            "cost": "a paid Alpaca market-data subscription that includes real-time SIP (the current price is on "
            "Alpaca's pricing page; QuantPulse has not verified it and never buys it)",
            "decision": "yours. With the subscription, set QP_ALPACA_STOCK_FEED=sip; the quote-age and spread "
            "limits stay exactly as they are.",
        },
    }


def _by_day(session: list[BrainCycleRow], blocked: list[BrainCycleRow]) -> list[dict[str, Any]]:
    days: dict[str, list[int]] = {}
    ids = {c.id for c in blocked}
    for c in session:
        day = c.started_at.astimezone(NEW_YORK).date().isoformat()
        tally = days.setdefault(day, [0, 0])
        tally[0] += 1
        tally[1] += int(c.id in ids)
    return [{"day": d, "cycles": n, "data_blocked": b} for d, (n, b) in sorted(days.items())]
