"""When — and exactly why — market data kept the Brain from trading.

Every cycle records each studied symbol's quote diagnosis (:mod:`.data_health`), the data-quality agent's
market-level view and the entry halts. This module classifies what stopped trading into precise categories:

=======================  ==================================================================================
stale_trade              the price was the last trade and it was older than the limit (or no print today)
stale_quote              the price was a two-sided bid/ask midpoint, and even that was older than the limit
wide_spread              the spread was wider than ``QP_TRADING_MAX_SPREAD_BPS`` (or could not be measured)
missing_quote            no quote came back for the symbol
provider_failure         the request failed, or the vendor refused the feed for this subscription
market_closed            outside the regular session or a holiday: nothing is executable, and that is not a
                         data problem
delayed_vendor           the feed itself was delayed (the 15-minute SIP feed): never an executable price
insufficient_coverage    too little of the universe had usable live quotes: new positions halted for all
invalid_timestamp        a print stamped in the future: its age cannot be known
clock_skew               this computer's clock too far from Alpaca's: no quote age can be trusted
broker_unavailable       the paper account could not be read
synthetic                only synthetic prices: never data
=======================  ==================================================================================

The report says how often each category occurred (symbol-cycles in the session), how often it was behind a
halt of new positions or a stopped trade decision, **when** — by day, by hour of the session and as
episodes of consecutive blocked cycles — and what would address it. The answer is never a looser
quote-age or spread limit: those are protected controls, and they stay exactly as they are.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import BrainCycleRow, BrainDecisionRow
from quantpulse.db.session import Database

from .audit import TRADE_ACTIONS
from .execution import DATA_BLOCKED

CATEGORIES = {
    "stale_trade": "the last trade was older than the limit (or none today)",
    "stale_quote": "even the bid/ask midpoint was older than the limit",
    "wide_spread": "the spread was wider than the limit, or could not be measured",
    "missing_quote": "no quote came back",
    "provider_failure": "the request failed or the vendor refused the feed",
    "market_closed": "outside the regular session",
    "delayed_vendor": "the feed itself was delayed",
    "insufficient_coverage": "too little of the universe had usable live quotes",
    "invalid_timestamp": "a print stamped in the future",
    "clock_skew": "this computer's clock too far from Alpaca's",
    "broker_unavailable": "the paper account could not be read",
    "synthetic": "only synthetic prices",
}
REMEDY = {
    "stale_trade": "real-time consolidated (SIP) data sees prints on every exchange, not one venue",
    "stale_quote": "a quiet name on one venue: SIP data, or the name is simply not trading",
    "wide_spread": "SIP data measures the national best bid/offer instead of one venue's book",
    "missing_quote": "check the symbol's mapping and the vendor's coverage",
    "provider_failure": "a vendor outage or subscription: check the provider's status and keys",
    "market_closed": "nothing to fix: the market is closed",
    "delayed_vendor": "a real-time feed; a delayed price is never executable",
    "insufficient_coverage": "follows from the per-symbol causes above (SIP data for staleness)",
    "invalid_timestamp": "bad vendor data or a wrong clock",
    "clock_skew": "synchronise this computer's clock",
    "broker_unavailable": "Alpaca's paper API: its status, the keys, the network",
    "synthetic": "configure a real market-data provider",
}
STATUS_CATEGORY = {
    "missing": "missing_quote",
    "subscription_unavailable": "provider_failure",
    "provider_error": "provider_failure",
    "market_closed": "market_closed",
    "holiday": "market_closed",
    "delayed": "delayed_vendor",
    "invalid_timestamp": "invalid_timestamp",
    "synthetic": "synthetic",
    "no_trade_today": "stale_trade",
}


def classify_symbol(diag: dict[str, Any], max_spread_bps: float) -> list[str]:
    """The data problems of one symbol's quote diagnosis (empty: usable)."""
    status = str(diag.get("status") or "")
    out: list[str] = []
    if status == "stale":
        out.append("stale_quote" if "bid/ask" in str(diag.get("price_source") or "") else "stale_trade")
    elif status in STATUS_CATEGORY:
        out.append(STATUS_CATEGORY[status])
    if status in ("fresh", "live", "stale"):
        spread = diag.get("spread_bps")
        if (spread is not None and spread > max_spread_bps) or (
            spread is None and diag.get("spread_source") == "unavailable"
        ):
            out.append("wide_spread")
    return out


def classify_market(market_view: dict[str, Any] | None) -> list[str]:
    """Cycle-wide data problems from the data-quality agent's market veto."""
    veto = str((market_view or {}).get("veto") or "")
    if not veto:
        return []
    out: list[str] = []
    for needle, code in (("usable live quotes", "insufficient_coverage"), ("clock", "clock_skew"),
                         ("broker unavailable", "broker_unavailable"), ("synthetic", "synthetic"),
                         ("market closed", "market_closed")):  # fmt: skip
        if needle in veto:
            out.append(code)
    return out


def classify_decision(d: BrainDecisionRow, diag: dict[str, Any] | None, cycle_codes: Sequence[str],
                      max_spread_bps: float) -> list[str]:  # fmt: skip
    """Why market data stopped one trade decision (empty: it did not)."""
    ex, rationale = d.execution or {}, d.rationale or {}
    symbol = classify_symbol(diag, max_spread_bps) if diag else []
    if DATA_BLOCKED in str(ex.get("reason") or ""):
        return list(cycle_codes) or ["insufficient_coverage"]
    if any("market data" in b or "stale" in b or "quote" in b for b in rationale.get("blocked_by") or []):
        return [c for c in symbol if c != "wide_spread"] or ["stale_trade"]
    failed = [
        c
        for c in [*((d.risk or {}).get("checks") or []), *(ex.get("checks") or [])]
        if not c.get("passed", True)
    ]
    out: list[str] = []
    for c in failed:
        if c.get("name") == "live_data":
            out += [x for x in symbol if x != "wide_spread"] or (
                ["missing_quote"] if "no live quote" in str(c.get("detail")) else ["stale_trade"]
            )
        elif c.get("name") == "liquidity" and "spread" in str(c.get("detail")):
            out.append("wide_spread")
    return sorted(set(out))


async def report(db: Database, settings: Settings, since: datetime, until: datetime) -> dict[str, Any]:
    async with db.session() as s:
        cycles = (
            await s.scalars(
                select(BrainCycleRow)
                .where(
                    BrainCycleRow.started_at >= since,
                    BrainCycleRow.started_at <= until,
                    BrainCycleRow.status == "completed",
                )
                .order_by(BrainCycleRow.started_at)
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
    limit = settings.trading_max_spread_bps
    session = [c for c in cycles if (c.market or {}).get("open")]
    symbol_counts: Counter[str] = Counter()
    halt_causes: Counter[str] = Counter()
    by_hour: dict[int, Counter[str]] = defaultdict(Counter)
    by_day: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    cycle_codes: dict[int, list[str]] = {}
    for c in session:
        dq = c.data_quality or {}
        diags = dq.get("diagnosis") or {}
        per_cycle: Counter[str] = Counter()
        for diag in diags.values():
            per_cycle.update(classify_symbol(diag, limit))
        symbol_counts.update(per_cycle)
        market = classify_market(dq.get("market"))
        blocked = "data_quality" in ((c.summary or {}).get("entry_halts") or [])
        codes = market + [k for k, _ in per_cycle.most_common(3) if k not in market]
        cycle_codes[c.id] = codes if blocked else market
        local = c.started_at.astimezone(NEW_YORK)
        day = by_day.setdefault(local.date().isoformat(), {"cycles": 0, "blocked": 0, "causes": Counter()})
        day["cycles"] += 1
        if blocked:
            halt_causes.update(codes)
            by_hour[local.hour].update(codes or ["unexplained"])
            day["blocked"] += 1
            day["causes"].update(codes)
            if current is not None and current["day"] == local.date().isoformat():
                current["end"], current["cycles"] = c.started_at.isoformat(), current["cycles"] + 1
                current["causes"].update(codes)
            else:
                current = {"day": local.date().isoformat(), "start": c.started_at.isoformat(),
                           "end": c.started_at.isoformat(), "cycles": 1, "causes": Counter(codes)}  # fmt: skip
                episodes.append(current)
        else:
            current = None
    diag_by_cycle = {c.id: ((c.data_quality or {}).get("diagnosis") or {}) for c in cycles}
    stopped: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    stopped_n = 0
    for d in decisions:
        codes = classify_decision(
            d, diag_by_cycle.get(d.cycle_id, {}).get(d.subject), cycle_codes.get(d.cycle_id, []), limit
        )
        stopped.update(codes)
        stopped_n += bool(codes)
        if codes and len(examples) < 20:
            examples.append({"decision_id": d.id, "subject": d.subject, "action": d.action, "causes": codes,
                             "at": d.created_at.isoformat()})  # fmt: skip
    categories = {
        k: {"meaning": CATEGORIES[k], "symbol_cycles": symbol_counts.get(k, 0), "behind_halts": halt_causes.get(k, 0),
            "decisions_stopped": stopped.get(k, 0), "remedy": REMEDY[k]}
        for k in CATEGORIES
        if symbol_counts.get(k) or halt_causes.get(k) or stopped.get(k)
    }  # fmt: skip
    blocked_n = sum(d["blocked"] for d in by_day.values())
    return {
        "headline": (
            f"market data halted new positions in {blocked_n} of {len(session)} in-session cycles"
            + (
                f", most often for {halt_causes.most_common(1)[0][0].replace('_', ' ')}"
                if halt_causes
                else ""
            )
            if session
            else "no in-session cycles in this window"
        ),
        "cycles_in_session": len(session),
        "blocked_cycles": blocked_n,
        "categories": categories,
        "episodes": [{**e, "causes": dict(e["causes"])} for e in episodes[-30:]],
        "by_hour": {f"{h:02d}:00": dict(v) for h, v in sorted(by_hour.items())},
        "by_day": [
            {"day": k, "cycles": v["cycles"], "blocked": v["blocked"], "causes": dict(v["causes"])}
            for k, v in sorted(by_day.items())
        ],
        "decisions_stopped": stopped_n,
        "stopped_examples": examples,
        "principle": "never solved by a looser quote-age or spread limit: those are protected controls and stay "
        "as they are",
    }
