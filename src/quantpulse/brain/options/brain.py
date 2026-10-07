"""The Options Brain: one pass per Brain cycle, inside the Brain (not a second engine).

::

    perceive (live chains, point-in-time features, IV history) → manage open positions (sync paper orders
    with Alpaca, mark, exit rules, expiration) → candidates from validated strategies (the lab's own entry
    rules, the same contract selection as the backtester) → the agents deliberate (vetoes stop a trade) →
    thesis, bull/bear/devil's advocate, plain-words explanation → options versus shares (the priority weight
    favours options, never forces them) → shadow positions for every chosen candidate, paper orders for
    strategies earned PAPER_ACTIVE (or one-contract exploration) → the trading service executes them
    (reconciliation, the risk engine, the last gate, the order manager) → attribution, counterfactuals,
    lessons, missed opportunities and weights when positions close.

Evidence is never mixed: every position is ``shadow`` (simulated on live quotes at REALISTIC fills) or
``paper`` (a real Alpaca paper order); research results are ``model``-priced and labelled as such.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, is_market_open, regular_close
from quantpulse.db.models import BrokerOrderRow
from quantpulse.db.options_models import (
    OptionsAssignmentEventRow,
    OptionsCounterfactualRow,
    OptionsExecutionLedgerRow,
    OptionsExerciseEventRow,
    OptionsFeatureObservationRow,
    OptionsLearningEventRow,
    OptionsLessonRow,
    OptionsMissedOpportunityRow,
    OptionsPositionEventRow,
    OptionsPositionRow,
    OptionsStrategyDecayRow,
    OptionsStrategyWeightRow,
    OptionsTradeCandidateRow,
    OptionsTradeThesisRow,
)
from quantpulse.db.session import Database
from quantpulse.options import expiration as expiry
from quantpulse.options.attribution import Mark, attribute_path, verdicts
from quantpulse.options.contracts import ContractError, OptionContract, parse_occ
from quantpulse.options.data import OptionsMarketDataProvider
from quantpulse.options.fills import ExecutionModel, leg_fill
from quantpulse.options.lab import learning, lessons, promotion
from quantpulse.options.lab.backtest import _chain_filters, _exit_reason, _passes, _pick
from quantpulse.options.lab.genome import Genome, from_dict
from quantpulse.options.pricing import greeks as bsm_greeks
from quantpulse.options.pricing import implied_vol
from quantpulse.options.quotes import OptionQuote
from quantpulse.options.selection import Candidate, Spec, build, empirical_distribution, evaluate, score
from quantpulse.options.structures import FAMILIES, Leg, Structure
from quantpulse.services.options_lab import OptionsLabService, jsonable
from quantpulse.services.trading_options import option_key
from quantpulse.services.trading_risk import (
    OptionLegIntent,
    OptionLegQuote,
    OptionOrderIntent,
    RiskBook,
    RiskLimits,
)

from . import agents as A
from .perception import UnderlyingView, perceive, record

logger = logging.getLogger(__name__)
SHADOW_PER_CYCLE = 5
# stages whose paper trades are one exploration contract (full size only at PAPER_ACTIVE and PROVEN)
EXPLORING = frozenset(s.value for s in promotion.EXPLORATION_STAGES)
# open shadow positions at most (each is marked every cycle); a strategy holds at most one per underlying
SHADOW_MAX_OPEN = 60
# "assigned": a leg went (assigned early, or closed outside QuantPulse) while others remain — frozen for a person
LIVE = ("pending", "open", "closing", "assigned")
# an order that will not fill any further (whatever part of it did fill stays filled)
ENDED = ("canceled", "expired", "rejected", "submit_failed", "done_for_day")
RATE = 0.04


@dataclass
class OptionsCycle:
    cycle_id: int | None
    at: datetime
    orders: list[OptionOrderIntent] = field(default_factory=list)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)  # option_key -> what it is for
    views: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    shadow_opened: list[dict[str, Any]] = field(default_factory=list)
    closed: list[dict[str, Any]] = field(default_factory=list)
    exits: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    no_trade: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        return jsonable({
            "at": self.at.isoformat(), "underlyings": self.views, "orders": [
                {"key": option_key(o), "family": o.family, "underlying": o.underlying, "qty": o.qty,
                 "limit": o.limit_price, "kind": o.kind, "opening": o.opening, "exploration": o.exploration}
                for o in self.orders],
            "candidates": self.candidates, "shadow_opened": self.shadow_opened, "closed": self.closed,
            "exits": self.exits, "comparisons": self.comparisons, "no_trade": self.no_trade, "notes": self.notes,
            "entry_filters": dict(self.skipped),
        })  # fmt: skip


def _legs_of(structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(structure.get("legs") or [])


def _sign(side: str) -> int:
    return 1 if side in ("long", "buy") else -1


def unit_value(
    legs: Sequence[Mapping[str, Any]], quotes: Mapping[str, OptionQuote]
) -> tuple[float | None, bool]:
    """Liquidation value of one unit at the mids, in dollars (a credit position is negative); ``stale``
    when a leg has no two-sided quote (its last price is used)."""
    total, stale = 0.0, False
    for leg in legs:
        q = quotes.get(leg["symbol"])
        mid = q.mid if q is not None else None
        if mid is None:
            stale = True
            mid = float(leg.get("last_mid") or leg.get("entry_mid") or 0.0)
        total += _sign(leg["side"]) * int(leg["ratio"]) * mid * 100
    return total, stale


def fill_value(
    legs: Sequence[Mapping[str, Any]], quotes: Mapping[str, OptionQuote], opening: bool
) -> float | None:
    """Dollars per unit at REALISTIC fills: paid to open (positive a debit) or received to close."""
    total = 0.0
    for leg in legs:
        q = quotes.get(leg["symbol"])
        if q is None or not q.two_sided:
            return None
        s = _sign(leg["side"])
        trade_sign = s if opening else -s
        px = leg_fill(trade_sign, q.bid, q.ask, ExecutionModel.REALISTIC)  # type: ignore[arg-type]
        total += trade_sign * int(leg["ratio"]) * px * 100
    return total if opening else -total


def leg_quote(q: OptionQuote, now: datetime, spot: float | None) -> OptionLegQuote:
    """What the risk book reads of a chain quote (IV, delta and vega computed here from the mid)."""
    iv = delta = vega = None
    mid = q.mid
    c = q.contract
    if mid is not None and spot:
        years = c.years(now)
        iv = implied_vol(c.kind, mid, spot, c.strike, years, RATE)
        if iv is not None and years > 0:
            v = bsm_greeks(c.kind, spot, c.strike, years, iv, RATE)
            delta, vega = v.delta, v.vega
    return OptionLegQuote(q.bid, q.ask, q.age(now), q.feed, q.open_interest, iv, delta, vega)


class OptionsBrain:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        db: Database,
        data: OptionsMarketDataProvider | None,
        market: Any,
        lab: OptionsLabService,
        reference: Any = None,
        refresh_orders: Callable[[list[str]], Awaitable[Any]] | None = None,
    ) -> None:
        self._s = settings
        # reads working orders' status from Alpaca into the order record (a lookup, never an order)
        self._refresh_orders = refresh_orders
        self._clock = clock
        self._db = db
        self.data = data
        self._market = market
        self._lab = lab
        self._reference = reference
        self.last: dict[str, Any] | None = None

    # ------------------------------------------------------------------ the cycle
    async def run(
        self,
        ctx: Any = None,
        cycle_id: int | None = None,
        *,
        consensus: Mapping[str, Any] | None = None,
        stock_proposals: Sequence[Any] = (),
        paper_allowed: bool = False,
    ) -> OptionsCycle:
        now = self._clock.now()
        cyc = OptionsCycle(cycle_id, now)
        s = self._s
        if not s.options_enabled:
            cyc.notes.append("options are switched off (QP_OPTIONS_ENABLED=false)")
            return cyc
        if self.data is None or not self.data.configured():
            cyc.notes.append(
                "no options market data (Alpaca keys not configured): nothing option-related decided"
            )
            return cyc
        market_open = ctx.market_open if ctx is not None else is_market_open(now)
        paper_allowed = paper_allowed and s.options_execution and market_open
        versions = await self._lab.eligible_versions()
        open_rows = await self._open_positions()
        unders = list(dict.fromkeys([*s.options_universe, *(r.underlying for r in open_rows)]))
        views = await self._perceive(unders, now, ctx)
        for u, v in views.items():
            cyc.views[u] = v.summary()
        if not versions:
            cyc.notes.append(("no strategy has passed its backtests (positive after realistic and pessimistic costs) yet"
                              if s.options_exploration
                              else "no strategy has passed validation (walk-forward, stress, baselines, the critic) yet")
                             + ": options are researched, not traded")  # fmt: skip

        # 1. what is already open: paper orders synced with Alpaca, marks, exits, expiration
        await self._manage(cyc, open_rows, views, ctx, now, market_open, paper_allowed)

        # 2. new candidates (only while the market is open and not in the last minutes of the session)
        local = now.astimezone(NEW_YORK)
        close_at = datetime.combine(local.date(), regular_close(local.date()), NEW_YORK)
        late = close_at - local < timedelta(minutes=max(s.trading_stop_minutes_before_close, 15))
        if not market_open or late:
            cyc.notes.append(
                "market closed" if not market_open else "too close to the close for new positions"
            )
        elif versions:
            await self._entries(
                cyc, versions, views, ctx, now, consensus or {}, stock_proposals, paper_allowed
            )
        self.last = cyc.summary()
        return cyc

    async def _perceive(self, unders: Sequence[str], now: datetime, ctx: Any) -> dict[str, UnderlyingView]:
        assert self.data is not None
        sem = asyncio.Semaphore(3)
        out: dict[str, UnderlyingView] = {}

        async def one(u: str) -> None:
            async with sem:
                earnings = None
                ev = getattr(ctx, "earnings", {}) if ctx is not None else {}
                if u in ev:
                    earnings = ev[u][0]
                closes = None
                if ctx is not None and u in getattr(ctx, "close", {}):
                    col = ctx.close[u].dropna()
                    closes = {ts.date() if hasattr(ts, "date") else ts: float(v) for ts, v in col.items()}
                try:
                    view = await asyncio.wait_for(
                        perceive(u, data=self.data, market=self._market, db=self._db, now=now,  # type: ignore[arg-type]
                                 min_dte=self._s.options_min_dte, max_dte=self._s.options_max_dte,
                                 next_earnings=earnings, closes=closes),
                        timeout=self._s.brain_research_timeout_seconds * 2,
                    )  # fmt: skip
                except TimeoutError:
                    view = UnderlyingView(u, now, problems=["option data timed out"])
                out[u] = view
                try:
                    await record(self._db, view, now)
                except Exception:
                    logger.exception("recording the option chain of %s failed", u)

        await asyncio.gather(*(one(u) for u in unders))
        return {u: out[u] for u in unders if u in out}

    # ------------------------------------------------------------------ positions
    async def _open_positions(self) -> list[OptionsPositionRow]:
        async with self._db.session() as s:
            return list(
                (await s.scalars(select(OptionsPositionRow).where(OptionsPositionRow.status.in_(LIVE)))).all()
            )

    async def _event(self, s: Any, pid: int, kind: str, message: str, now: datetime, **detail: Any) -> None:
        s.add(
            OptionsPositionEventRow(
                position_id=pid, at=now, kind=kind, message=message[:2000], detail=jsonable(detail)
            )
        )

    async def _manage(self, cyc: OptionsCycle, rows: Sequence[OptionsPositionRow], views: Mapping[str, UnderlyingView],
                      ctx: Any, now: datetime, market_open: bool, paper_allowed: bool) -> None:  # fmt: skip
        held = (
            dict(getattr(getattr(ctx, "account", None), "option_positions", {}) or {})
            if ctx is not None
            else None
        )
        waiting = [
            cid
            for r in rows
            if r.mode == "paper"
            for cid in (
                r.client_order_id
                if r.status == "pending"
                else r.exit_client_order_id
                if r.status == "closing"
                else None,
            )
            if cid
        ]
        if waiting and self._refresh_orders is not None:
            try:
                await self._refresh_orders(waiting)
            except Exception as exc:  # the record stays as it was; the next cycle tries again
                cyc.notes.append(f"working option orders could not be refreshed ({type(exc).__name__})")
        for row in rows:
            try:
                if row.mode == "paper":
                    await self._sync_paper(cyc, row, held, views, now)
                async with self._db.session() as s:
                    pos = await s.get(OptionsPositionRow, row.id)
                if pos is None or pos.status not in ("open",):
                    continue
                await self._mark_and_exit(cyc, pos, views, now, market_open, paper_allowed)
            except Exception:
                logger.exception("managing option position %s failed", row.id)
                cyc.notes.append(f"position {row.id}: management failed (logged); it stays as it was")

    async def _sync_paper(self, cyc: OptionsCycle, row: OptionsPositionRow, held: Mapping[str, Any] | None,
                          views: Mapping[str, UnderlyingView], now: datetime) -> None:  # fmt: skip
        """A paper position follows its orders at Alpaca: filled opens open, unfilled ones are dropped, filled
        closes close; contracts no longer held (expired, assigned, closed outside QuantPulse) close it."""
        async with self._db.session() as s:
            pos = await s.get(OptionsPositionRow, row.id)
            assert pos is not None
            if pos.status == "pending" and pos.client_order_id:
                o = await s.scalar(
                    select(BrokerOrderRow).where(BrokerOrderRow.client_order_id == pos.client_order_id)
                )
                if o is not None and o.status == "filled" and o.average_fill_price is not None:
                    pos.status = "open"
                    pos.entry_value = _net_dollars(
                        o.average_fill_price, pos.structure, pos.quantity, opening=True
                    )
                    await self._event(
                        s, pos.id, "filled", f"opened at {o.average_fill_price:+.2f} per share", now
                    )
                elif o is None or o.status in ENDED:
                    filled = int(o.filled_quantity or 0) if o is not None else 0
                    if o is not None and 0 < filled < pos.quantity and o.average_fill_price is not None:
                        # part filled, the rest ended (a DAY order at the close): the position is what was bought
                        ordered, share = pos.quantity, filled / pos.quantity
                        pos.status, pos.quantity = "open", filled
                        pos.max_loss = round(pos.max_loss * share, 2)
                        pos.max_profit = (
                            round(pos.max_profit * share, 2) if pos.max_profit is not None else None
                        )
                        pos.entry_value = _net_dollars(
                            o.average_fill_price, pos.structure, filled, opening=True
                        )
                        await self._event(s, pos.id, "partially_filled", f"{filled} of {ordered} filled at "
                                          f"{o.average_fill_price:+.2f} per share; the rest {o.status}", now)  # fmt: skip
                        return
                    pos.status, pos.closed_at = "closed", now
                    pos.exit_reason = f"the opening order never filled ({o.status if o else 'not found'})"
                    pos.realized_pnl = 0.0
                    await self._event(s, pos.id, "not_filled", pos.exit_reason, now)
                return
            if pos.status == "closing" and pos.exit_client_order_id:
                o = await s.scalar(
                    select(BrokerOrderRow).where(BrokerOrderRow.client_order_id == pos.exit_client_order_id)
                )
                if o is not None and o.status == "filled" and o.average_fill_price is not None:
                    exit_value = _net_dollars(
                        o.average_fill_price, pos.structure, pos.quantity, opening=False
                    )
                    await self._close(s, cyc, pos, exit_value, pos.exit_reason or "exit", now, views)
                elif o is None or o.status in ENDED:
                    filled = int(o.filled_quantity or 0) if o is not None else 0
                    if o is not None and 0 < filled < pos.quantity and o.average_fill_price is not None:
                        # part closed: book that part's P&L, keep the rest open at its share of the entry
                        share = filled / pos.quantity
                        exit_part = _net_dollars(o.average_fill_price, pos.structure, filled, opening=False)
                        pnl_part = round(exit_part - pos.entry_value * share, 2)
                        parts = [*(pos.structure.get("partial_exits") or []),
                                 {"at": now.isoformat(), "qty": filled, "exit_value": exit_part, "pnl": pnl_part}]  # fmt: skip
                        pos.structure = {**pos.structure, "partial_exits": parts}
                        pos.entry_value = round(pos.entry_value * (1 - share), 2)
                        pos.max_loss = round(pos.max_loss * (1 - share), 2)
                        pos.max_profit = (
                            round(pos.max_profit * (1 - share), 2) if pos.max_profit is not None else None
                        )
                        pos.quantity -= filled
                        pos.status, pos.exit_client_order_id = "open", None
                        await self._event(s, pos.id, "partly_closed", f"{filled} closed ({pnl_part:+,.2f}); "
                                          f"{pos.quantity} still open: the exit is tried again", now)  # fmt: skip
                        return
                    pos.status, pos.exit_client_order_id = "open", None
                    await self._event(s, pos.id, "exit_not_filled", f"the closing order did not fill "
                                      f"({o.status if o else 'not found'}): it is tried again", now)  # fmt: skip
                return
            if pos.status == "open" and held is not None:
                missing = [leg["symbol"] for leg in _legs_of(pos.structure) if leg["symbol"] not in held]
                if missing and len(missing) == len(_legs_of(pos.structure)):
                    await self._gone(s, cyc, pos, views, now)
                elif missing:
                    await self._partly_gone(s, cyc, pos, missing, views, now)
            elif pos.status == "assigned" and held is not None:
                if not any(leg["symbol"] in held for leg in _legs_of(pos.structure)):
                    pos.status, pos.closed_at = "closed", now
                    pos.exit_reason = (
                        "resolved outside QuantPulse after a leg was assigned: P&L unknown here "
                        "(see the account's activity)"
                    )
                    await self._event(s, pos.id, "resolved", pos.exit_reason, now)

    async def _partly_gone(self, s: Any, cyc: OptionsCycle, pos: OptionsPositionRow, missing: Sequence[str],
                           views: Mapping[str, UnderlyingView], now: datetime) -> None:  # fmt: skip
        """Some legs are gone while others remain: a short leg assigned (American options can be assigned any
        day) or a leg closed outside QuantPulse. What remains is the hedge of whatever the assignment delivered,
        so nothing is closed automatically — closing the surviving long leg alone could leave naked stock. The
        position is frozen (no exit orders, and it still blocks new trades on this underlying), the likely
        assignment is recorded as inferred, and a person is alerted to close the shares and the rest together."""
        view = views.get(pos.underlying)
        spot = view.spot if view is not None else None
        today = now.astimezone(NEW_YORK).date()
        inferred = []
        for leg in _legs_of(pos.structure):
            if leg["symbol"] not in missing or leg["side"] != "short" or spot is None:
                continue
            c = parse_occ(leg["symbol"])
            st = expiry.settlement(c, "short", int(leg["ratio"]) * pos.quantity, spot)
            if st["share_delivery"]:  # in the money: an assignment is the likely explanation
                inferred.append(leg["symbol"])
                s.add(OptionsAssignmentEventRow(position_id=pos.id, at=now, symbol=leg["symbol"],
                                                contracts=int(leg["ratio"]) * pos.quantity,
                                                share_delivery=int(st["share_delivery"]), cash_flow=float(st["cash_flow"]),
                                                detail=jsonable({**st, "inferred": True,
                                                                 "early": c.expiration > today})))  # fmt: skip
        pos.status = "assigned"
        pos.expiry_state = expiry.ExpiryState.ASSIGNED.value if inferred else pos.expiry_state
        cause = f"{', '.join(inferred)} most likely assigned" if inferred else "closed outside QuantPulse"
        message = (
            f"{', '.join(missing)} no longer held while the rest of the position is ({cause}). Frozen: "
            "QuantPulse sends no exit for it (the remaining legs hedge what was delivered); close the shares "
            "and the remaining legs together in the paper account."
        )
        await self._event(s, pos.id, "legs_gone", message, now, missing=list(missing), inferred=inferred)
        cyc.notes.append(f"position {pos.id}: {message}")

    async def _gone(self, s: Any, cyc: OptionsCycle, pos: OptionsPositionRow, views: Mapping[str, UnderlyingView],
                    now: datetime) -> None:  # fmt: skip
        """Contracts QuantPulse did not close are gone from the account: past expiration they were settled by
        the OCC (exercised or assigned in the money, worthless otherwise) — recorded as such, never exercised
        by QuantPulse; before expiration someone else closed them."""
        view = views.get(pos.underlying)
        spot = view.spot if view is not None else None
        today = now.astimezone(NEW_YORK).date()
        settled = []
        for leg in _legs_of(pos.structure):
            c = parse_occ(leg["symbol"])
            if c.expiration > today or spot is None:
                continue
            st = expiry.settlement(c, leg["side"], int(leg["ratio"]) * pos.quantity, spot)
            settled.append(st)
            if st["share_delivery"]:
                row_cls = OptionsExerciseEventRow if leg["side"] == "long" else OptionsAssignmentEventRow
                s.add(row_cls(position_id=pos.id, at=now, symbol=leg["symbol"], contracts=int(leg["ratio"]) * pos.quantity,
                              share_delivery=int(st["share_delivery"]), cash_flow=float(st["cash_flow"]), detail=jsonable(st)))  # fmt: skip
        if settled:
            value = sum(_sign(leg["side"]) * int(leg["ratio"]) * parse_occ(leg["symbol"]).intrinsic(spot or 0) * 100
                        for leg in _legs_of(pos.structure)) * pos.quantity  # fmt: skip
            await self._close(
                s, cyc, pos, value, "expiration (settled by the OCC; QuantPulse never exercises)", now, views
            )
            pos.expiry_state = expiry.ExpiryState.EXPIRED.value
        else:
            pos.status, pos.closed_at = "closed", now
            pos.exit_reason = "no longer held at Alpaca (closed outside QuantPulse): P&L unknown here"
            await self._event(s, pos.id, "gone", pos.exit_reason, now)

    async def _mark_and_exit(self, cyc: OptionsCycle, pos: OptionsPositionRow, views: Mapping[str, UnderlyingView],
                             now: datetime, market_open: bool, paper_allowed: bool) -> None:  # fmt: skip
        view = views.get(pos.underlying)
        if view is None or view.chain is None or view.spot is None:
            return
        legs = _legs_of(pos.structure)
        quotes = {q.symbol: q for q in view.chain.quotes if q.symbol in {leg["symbol"] for leg in legs}}
        if len(quotes) < len(legs) and self.data is not None:
            try:
                quotes.update(
                    await self.data.latest_quotes(
                        [leg["symbol"] for leg in legs if leg["symbol"] not in quotes]
                    )
                )
            except Exception as exc:
                cyc.notes.append(f"position {pos.id}: leg quotes unavailable ({type(exc).__name__})")
        value_unit, stale = unit_value(legs, quotes)
        contracts = [parse_occ(leg["symbol"]) for leg in legs]
        g = self._greeks(legs, quotes, view.spot, now)
        today = now.astimezone(NEW_YORK).date().isoformat()
        mark = {"day": today, "spot": view.spot, "iv": g.get("iv") or pos.entry_iv or 0.0,
                "value": (value_unit or 0.0) * pos.quantity, "days": (now - pos.opened_at).total_seconds() / 86400,
                **{k: (g.get(k) or 0.0) * pos.quantity for k in ("delta", "gamma", "theta", "vega")}}  # fmt: skip
        new_legs = [{**leg, "last_mid": quotes[leg["symbol"]].mid if leg["symbol"] in quotes and quotes[leg["symbol"]].mid
                     else leg.get("last_mid")} for leg in legs]  # fmt: skip
        state = expiry.assess(contracts, view.spot, now, expiry.ExpiryRules(
            near_days=7, risk_days=max(self._s.options_close_dte, 1)))  # fmt: skip
        g_row = from_dict(pos.structure.get("genome") or {}) if pos.structure.get("genome") else None
        async with self._db.session() as s:
            p = await s.get(OptionsPositionRow, pos.id)
            assert p is not None
            p.marks = jsonable([*[m for m in p.marks if m.get("day") != today], mark])
            p.structure = {**p.structure, "legs": new_legs}
            try:
                p.expiry_state = expiry.transition(expiry.ExpiryState(p.expiry_state), state.state).value
            except expiry.TransitionError:
                p.expiry_state = state.state.value
            reason = self._exit_reason(p, g_row, value_unit, stale, state, now)
            if reason is None or not market_open:
                return
            cyc.exits.append(
                {"position_id": p.id, "mode": p.mode, "underlying": p.underlying, "reason": reason}
            )
            if p.mode == "shadow":
                exit_unit = fill_value(legs, quotes, opening=False)
                if exit_unit is None:
                    await self._event(
                        s, p.id, "exit_waiting", f"{reason}: no two-sided quote to exit into", now
                    )
                    return
                fees = self._s.options_fee_per_contract * sum(int(leg["ratio"]) for leg in legs) * p.quantity
                await self._close(s, cyc, p, exit_unit * p.quantity - fees, reason, now, views)
            elif paper_allowed:
                intent = _closing_intent(p, quotes, view.spot, reason)
                if intent is None:
                    await self._event(
                        s, p.id, "exit_waiting", f"{reason}: a leg has no quote to price the close", now
                    )
                    return
                cyc.orders.append(intent)
                cyc.pending[option_key(intent)] = {"kind": "close", "position_id": p.id, "reason": reason}
                p.exit_reason = reason
            else:
                await self._event(
                    s, p.id, "exit_due", f"{reason}: the paper account cannot be traded now", now
                )

    def _exit_reason(self, p: OptionsPositionRow, g: Genome | None, value_unit: float | None, stale: bool,
                     state: expiry.ExpiryAssessment, now: datetime) -> str | None:  # fmt: skip
        """Exit rules: the strategy's own (the backtester's), then the protective overlay for live positions:
        never into expiration, the stop and take-profit limits on the maximum loss and profit."""
        if state.must_close or state.dte <= self._s.options_close_dte:
            return f"expiration: {state.reason}"
        if value_unit is None:
            return None
        held_days = (now - p.opened_at).days
        entry_unit = (
            p.entry_mid if p.entry_mid is not None else p.entry_value / max(p.quantity, 1) / 100
        ) * 100
        if g is not None and not stale:

            class _P:
                entry_mid = entry_unit

            why = _exit_reason(g, _P, value_unit, state.dte, held_days)  # type: ignore[arg-type]
            if why:
                return why
        pnl = value_unit * p.quantity - p.entry_value
        if p.max_loss and pnl <= -self._s.options_stop_loss_pct * p.max_loss:
            return f"stop: lost {-pnl:,.0f} of a {p.max_loss:,.0f} maximum loss"
        if p.max_profit and pnl >= self._s.options_take_profit_pct * p.max_profit:
            return f"take profit: made {pnl:,.0f} of a {p.max_profit:,.0f} maximum profit"
        return None

    def _greeks(self, legs: Sequence[Mapping[str, Any]], quotes: Mapping[str, OptionQuote], spot: float,
                now: datetime) -> dict[str, float | None]:  # fmt: skip
        out: dict[str, float | None] = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        ivs = []
        for leg in legs:
            q = quotes.get(leg["symbol"])
            if q is None or q.mid is None:
                return dict.fromkeys((*out, "iv"))
            c = q.contract
            iv = implied_vol(c.kind, q.mid, spot, c.strike, c.years(now), RATE)
            if iv is None:
                return dict.fromkeys((*out, "iv"))
            ivs.append(iv)
            v = bsm_greeks(c.kind, spot, c.strike, c.years(now), iv, RATE)
            n = _sign(leg["side"]) * int(leg["ratio"]) * 100
            for k in ("delta", "gamma", "theta", "vega"):
                out[k] = (out[k] or 0.0) + n * getattr(v, k)
        out["iv"] = sum(ivs) / len(ivs) if ivs else None
        return out

    async def _close(self, s: Any, cyc: OptionsCycle, pos: OptionsPositionRow, exit_value: float, reason: str,
                     now: datetime, views: Mapping[str, UnderlyingView]) -> None:  # fmt: skip
        """Close a position and learn from it: P&L, attribution, the critique, a counterfactual, the event."""
        entry_fees = float((pos.structure.get("fees") or {}).get("open") or 0.0)
        pos.status, pos.closed_at, pos.exit_value, pos.exit_reason = "closed", now, exit_value, reason[:2000]
        partial = sum(float(x.get("pnl") or 0) for x in (pos.structure.get("partial_exits") or []))
        pos.realized_pnl = round(exit_value - pos.entry_value - entry_fees + partial, 2)
        marks = [Mark(m["spot"], m.get("iv") or 0.0, m["value"], m["days"], m.get("delta", 0.0), m.get("gamma", 0.0),
                      m.get("theta", 0.0), m.get("vega", 0.0)) for m in (pos.marks or [])]  # fmt: skip
        attribution = None
        if len(marks) >= 2:
            execution = (exit_value - (marks[-1].value)) + (
                float(pos.entry_mid or 0) * 100 * pos.quantity - pos.entry_value
            )
            a = attribute_path(marks, execution=execution, fees=-entry_fees)
            attribution = {
                **a.as_dict(),
                "dominant": a.dominant(),
                "verdicts": verdicts(a, direction=pos.direction),
            }
        pos.attribution = jsonable(attribution or {})
        view = views.get(pos.underlying)
        trade = {
            "underlying": pos.underlying, "family": pos.family, "direction": pos.direction, "pnl": pos.realized_pnl,
            "max_loss": pos.max_loss, "qty": pos.quantity, "entry_date": pos.opened_at.date().isoformat(),
            "exit_date": now.date().isoformat(), "dte_entry": (pos.first_expiration - pos.opened_at.date()).days
            if pos.first_expiration else None, "underlying_return": (view.spot / pos.entry_underlying - 1)
            if view is not None and view.spot and pos.entry_underlying else None,
            "attribution": attribution, "features": (pos.structure.get("features") or {}), "exit_reason": reason,
        }  # fmt: skip
        pos.critique = jsonable(lessons.critique(trade))
        await self._event(s, pos.id, "closed", f"{reason}: {pos.realized_pnl:+,.2f} ({pos.mode})", now)
        s.add(OptionsLearningEventRow(at=now, kind="trade_graded", evidence=pos.mode, subject=str(pos.version_id or pos.family),
                                      dims=jsonable({"family": pos.family, "underlying": pos.underlying,
                                                     "regime": (pos.structure.get("features") or {}).get("regime"),
                                                     "iv_regime": (pos.structure.get("features") or {}).get("iv_regime")}),
                                      predicted=(pos.structure.get("expected") or {}).get("pop"),
                                      actual=1.0 if pos.realized_pnl > 0 else 0.0,
                                      payload=jsonable({"pnl": pos.realized_pnl, "max_loss": pos.max_loss,
                                                        "ror": pos.realized_pnl / max(pos.max_loss or 1, 1),
                                                        "exit_reason": reason, "critique": pos.critique})))  # fmt: skip
        s.add(OptionsFeatureObservationRow(day=pos.opened_at.date(), underlying=pos.underlying,
                                           features=jsonable(pos.structure.get("features") or {}),
                                           outcome=jsonable({"pnl": pos.realized_pnl, "ror": pos.realized_pnl / max(pos.max_loss or 1, 1),
                                                             "family": pos.family, "mode": pos.mode}), source=pos.mode))  # fmt: skip
        cf = self._counterfactual(pos, trade, view)
        if cf is not None:
            s.add(OptionsCounterfactualRow(position_id=pos.id, candidate_id=pos.candidate_id, created_at=now,
                                           alternative="shares", structure=jsonable(cf), pnl=cf["pnl"],
                                           pnl_on_risk=cf.get("pnl_on_risk"), chosen_pnl=pos.realized_pnl,
                                           better_than_chosen=cf["pnl"] > (pos.realized_pnl or 0), data_source="recorded"))  # fmt: skip
        cyc.closed.append({"position_id": pos.id, "mode": pos.mode, "underlying": pos.underlying, "family": pos.family,
                           "pnl": pos.realized_pnl, "reason": reason})  # fmt: skip

    def _counterfactual(self, pos: OptionsPositionRow, trade: Mapping[str, Any], view: UnderlyingView | None
                        ) -> dict[str, Any] | None:  # fmt: skip
        """The same view expressed with shares instead (same maximum loss budget): did the structure add value?
        Real prices only (the underlying's entry and exit)."""
        if view is None or not view.spot or not pos.entry_underlying:
            return None
        direction = pos.direction
        if direction not in ("bullish", "bearish"):
            return None
        shares = (pos.max_loss or 0) / max(pos.entry_underlying * 0.1, 1e-9)  # shares a 10% stop would allow
        move = view.spot - pos.entry_underlying
        pnl = (1 if direction == "bullish" else -1) * move * shares
        return {"alternative": "shares", "shares": round(shares, 2), "pnl": round(pnl, 2),
                "pnl_on_risk": round(pnl / max(pos.max_loss or 1, 1), 4),
                "note": "shares sized so a 10% stop risks the option's maximum loss"}  # fmt: skip

    # ------------------------------------------------------------------ entries
    async def _entries(self, cyc: OptionsCycle, versions: Sequence[Mapping[str, Any]], views: Mapping[str, UnderlyingView],
                       ctx: Any, now: datetime, consensus: Mapping[str, Any], stock_proposals: Sequence[Any],
                       paper_allowed: bool) -> None:  # fmt: skip
        s = self._s
        open_rows = await self._open_positions()
        # the paper and shadow books are separate: shadow evidence never blocks (or stands in for) a paper trade.
        # The paper book is the account's: one position per underlying across every strategy. Shadow trades are
        # each strategy's own forward test, so each strategy has its own shadow book (one per underlying):
        # one strategy's shadow position never keeps another from building its record.
        paper_book: dict[str, list[int]] = defaultdict(list)
        shadow_books: dict[int | None, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for r in open_rows:
            if r.mode == "paper":
                paper_book[r.underlying].append(r.id)
            else:
                shadow_books[r.version_id][r.underlying].append(r.id)
        shadow_open = sum(len(ids) for b in shadow_books.values() for ids in b.values())
        weights = await self._weights()
        decays = await self._decays()
        rng = random.Random(now.date().toordinal())
        scored: list[dict[str, Any]] = []
        for v in versions:
            g = from_dict(v["genome"])
            for u, view in views.items():
                if not view.usable:
                    cyc.skipped["no usable chain or features"] += 1
                    continue
                f = view.features
                assert f is not None
                assert view.chain is not None
                assert view.spot is not None
                why = _passes(g, f, rng) or _chain_filters(g, view.chain.quotes, view.spot, now)
                if why:
                    cyc.skipped[why] += 1
                    continue
                spec = Spec(g.family, max(g.dte_min, s.options_min_dte), min(g.dte_max, s.options_max_dte), g.delta_target,
                            g.width_pct, g.wing_pct, g.max_spread_pct, g.min_open_interest)  # fmt: skip
                pick = _pick(build(spec, view.chain.quotes, view.spot, now), g)
                if pick is None:
                    cyc.skipped["no liquid structure"] += 1
                    continue
                evaluate(pick, view.spot, now, fee_per_contract=s.options_fee_per_contract)
                terminal = empirical_distribution(
                    view.spot, [view.closes[d] for d in sorted(view.closes)], pick.dte
                )
                if terminal is not None:
                    emp = Candidate(pick.structure, pick.quotes, pick.expiration, pick.dte)
                    evaluate(emp, view.spot, now, terminal=terminal, distribution="empirical",
                             fee_per_contract=s.options_fee_per_contract)  # fmt: skip
                    pick.metrics["empirical"] = {
                        k: emp.metrics.get(k) for k in ("expected_pnl", "pop", "expected_on_risk")
                    }
                score(pick)
                paper_mode = paper_allowed and (
                    v["stage"] in ("PAPER_ACTIVE", "PROVEN") or s.options_exploration
                )
                intent = (
                    self._intent(pick, g, v, view, ctx, now, exploration=v["stage"] in EXPLORING)
                    if paper_mode
                    else None
                )
                risk = self._preview(intent, ctx, view, now) if intent is not None else None
                cc = A.CandidateContext(view=view, version=v, genome=g, cand=pick, now=now,
                                        stock_view=_stock_view(consensus.get(u)),
                                        book=paper_book if intent is not None else shadow_books[v["version_id"]],
                                        weight=weights.get((v["key"], view.regime or "", g.family, view.vol_regime or "")),
                                        decay=decays.get(v["version_id"]), risk=risk, paper=intent is not None)  # fmt: skip
                verdict = A.deliberate(cc)
                scored.append({"v": v, "g": g, "view": view, "cand": pick, "intent": intent, "risk": risk,
                               "verdict": verdict, "cc": cc})  # fmt: skip
        # the best first; a veto, a non-positive validated edge or a weak verdict never trades
        for item in scored:
            item["rank"] = item["verdict"]["score"] + (
                s.options_priority_weight if item["intent"] is not None else 0.0
            )
        scored.sort(key=lambda x: x["rank"], reverse=True)
        # this cycle's new positions: paper by underlying, shadow by (strategy, underlying)
        chosen_paper: set[str] = set()
        chosen_shadow: set[tuple[int | None, str]] = set()
        n_shadow = n_paper = 0
        max_paper = s.brain_max_new_positions_per_cycle
        for item in scored:
            v, verdict, u = item["v"], item["verdict"], item["view"].underlying
            own = shadow_books[v["version_id"]]
            paper = item["intent"] is not None
            gate = None
            if verdict["vetoes"]:
                gate = verdict["vetoes"][0]["agent"]
            elif (v.get("expected_ror") or 0) <= 0:
                gate = "no validated edge"
            elif verdict["score"] <= 0.05:
                gate = "weak verdict"
            elif paper and (u in chosen_paper or paper_book.get(u)):
                gate = "one position per underlying"
            elif not paper and ((v["version_id"], u) in chosen_shadow or own.get(u)):
                gate = "this strategy already holds this underlying"
            elif n_shadow >= SHADOW_PER_CYCLE:
                gate = "enough new positions this cycle"
            elif not paper and shadow_open >= SHADOW_MAX_OPEN:
                gate = "the shadow book is full"
            comparison = self._compare(item, consensus.get(u), stock_proposals)
            cyc.comparisons.append(comparison)
            cand_id, thesis_id = await self._record_candidate(cyc, item, gate, comparison, now)
            if gate is not None:
                await self._missed(item, cand_id, gate, now)
                continue
            if paper:
                chosen_paper.add(u)
            n_shadow += 1
            if not own.get(u) and (v["version_id"], u) not in chosen_shadow and shadow_open < SHADOW_MAX_OPEN:
                chosen_shadow.add((v["version_id"], u))
                shadow_open += 1
                await self._open_shadow(cyc, item, cand_id, thesis_id, now)
            intent = item["intent"]
            if intent is not None and n_paper < max_paper and (item["risk"] or {}).get("approved"):
                n_paper += 1
                cyc.orders.append(intent)
                cyc.pending[option_key(intent)] = {"kind": "open", "candidate_id": cand_id, "thesis_id": thesis_id,
                                                   "version_id": v["version_id"], "item": _entry_record(item)}  # fmt: skip
                if comparison.get("prefer_option"):
                    for p in stock_proposals:  # the view is expressed through the option: no stock entry too
                        if getattr(p, "subject", None) == u and getattr(
                            getattr(p, "action", None), "value", ""
                        ) in ("buy", "increase"):
                            p.blocked_by.append(
                                f"expressed through options ({intent.family}): {comparison['verdict']}"
                            )
        if not n_shadow:
            counts = Counter(i["verdict"]["vetoes"][0]["agent"] for i in scored if i["verdict"]["vetoes"])
            reasons = [f"{n} vetoed by {a}" for a, n in counts.most_common(4)] + [
                f"{n} × {why}" for why, n in cyc.skipped.most_common(4)]  # fmt: skip
            cyc.no_trade = A.no_trade(reasons).as_dict()

    def _intent(self, cand: Candidate, g: Genome, v: Mapping[str, Any], view: UnderlyingView, ctx: Any,
                now: datetime, *, exploration: bool) -> OptionOrderIntent | None:  # fmt: skip
        """The paper order for a candidate: a REALISTIC-level net limit, sized by the strategy's risk budget and
        the protected limits (exploration: one unit, capped lower)."""
        from quantpulse.options.fills import limit_price

        legs_q = [
            (leg.sign, leg.ratio, q) for leg, q in zip(cand.structure.option_legs, cand.quotes, strict=True)
        ]
        limit = limit_price(1, legs_q)
        loss_unit = cand.metrics.get("max_loss")
        if limit is None or loss_unit is None or not math.isfinite(loss_unit) or loss_unit <= 0:
            return None
        equity = (
            getattr(getattr(getattr(ctx, "account", None), "account", None), "equity", None)
            or self._s.brain_book_capital
        )
        s = self._s
        cap = min(
            s.options_max_loss_per_trade, s.options_max_loss_pct_per_trade * equity, g.risk_per_trade * equity
        )
        if exploration:
            if loss_unit > s.options_exploration_max_loss:
                return None
            qty = 1
        else:
            qty = min(int(cap // loss_unit), s.options_max_contracts)
            # sized to use at most half of the book's delta and vega limits (the risk engine checks the whole book)
            gk = cand.metrics.get("greeks") or {}
            unit_delta = abs(float(gk.get("delta") or 0.0)) * (view.spot or 0.0)
            unit_vega = abs(float(gk.get("vega") or 0.0))
            if unit_delta > 0:
                qty = min(qty, int(0.5 * s.options_max_delta_pct * equity // unit_delta))
            if unit_vega > 0:
                qty = min(qty, int(0.5 * s.options_max_vega_pct * equity // unit_vega))
            if qty < 1:
                return None
        legs = tuple(OptionLegIntent(q.symbol, "buy" if leg.sign > 0 else "sell", leg.ratio,
                                     "buy_to_open" if leg.sign > 0 else "sell_to_open")
                     for leg, q in zip(cand.structure.option_legs, cand.quotes, strict=True))  # fmt: skip
        return OptionOrderIntent(view.underlying, g.family, legs, qty, limit, "entry",
                                 f"{v['key']}: {g.describe()[:200]}", True, view.spot or 0.0, exploration=exploration,
                                 score=None, strategy_key=v["key"])  # fmt: skip

    def _preview(
        self, o: OptionOrderIntent, ctx: Any, view: UnderlyingView, now: datetime
    ) -> dict[str, Any] | None:
        acct = getattr(ctx, "account", None)
        if (
            acct is None
            or not getattr(acct, "available", False)
            or acct.account is None
            or view.chain is None
        ):
            return {"approved": False, "summary": "the paper account could not be read"}
        quotes = {q.symbol: leg_quote(q, now, view.spot) for q in view.chain.quotes
                  if q.symbol in {x.symbol for x in o.legs} | set(acct.option_positions)}  # fmt: skip
        book = RiskBook(RiskLimits.from_settings(self._s), acct.account, {**acct.positions, **acct.option_positions},
                        acct.open_orders, bool(getattr(ctx, "market_open", True)), bool(getattr(ctx, "kill_switch", False)),
                        {}, option_quotes=quotes, now=now)  # fmt: skip
        d = book.evaluate_option(o, spot=view.spot)
        return {"approved": d.approved, "summary": d.summary, "checks": [c.name for c in d.failures]}

    def _compare(self, item: Mapping[str, Any], cons: Any, stock_proposals: Sequence[Any]) -> dict[str, Any]:
        """Options versus shares for the same underlying, on the Brain's own scale: the option's verdict plus
        the priority weight against the stock consensus. The weight can tip a close call toward the option;
        it never makes a failing option pass (vetoes and edge are checked first)."""
        u = item["view"].underlying
        sv = _stock_view(cons)
        stock_score = (sv or {}).get("score_signed")
        opt = item["verdict"]["score"]
        w = self._s.options_priority_weight
        proposal = next((p for p in stock_proposals if getattr(p, "subject", None) == u), None)
        prefer = stock_score is None or opt + w >= abs(stock_score)
        verdict = (f"option {opt:+.2f} + weight {w:.2f} vs stock {stock_score:+.2f}: "
                   f"{'the option expresses the view' if prefer else 'the shares express it better'}"
                   if stock_score is not None else f"option {opt:+.2f}; no stock view on {u}")  # fmt: skip
        return {"underlying": u, "option_score": opt, "priority_weight": w, "stock_score": stock_score,
                "stock_proposal": getattr(getattr(proposal, "action", None), "value", None), "prefer_option": prefer,
                "verdict": verdict}  # fmt: skip

    async def _record_candidate(self, cyc: OptionsCycle, item: Mapping[str, Any], gate: str | None,
                                comparison: Mapping[str, Any], now: datetime) -> tuple[int, int | None]:  # fmt: skip
        v, g, view, cand, verdict = item["v"], item["g"], item["view"], item["cand"], item["verdict"]
        f = view.features
        mode = "paper" if item["intent"] is not None else "shadow"
        status = "rejected" if gate else ("proposed" if mode == "paper" else "shadow_opened")
        cc = item["cc"]
        th = A.thesis(cc, verdict)
        db_ = A.debate(cc, verdict)
        text = A.explain(th, verdict, mode=mode if not gate else "rejected", comparison=comparison)
        cyc.candidates.append({"underlying": view.underlying, "strategy": v["key"], "family": g.family, "status": status,
                               "gate": gate, "score": verdict["score"], "mode": mode, "explanation": text})  # fmt: skip
        async with self._db.session() as s:
            row = OptionsTradeCandidateRow(
                cycle_key=f"brain-{cyc.cycle_id}" if cyc.cycle_id else f"options-{now:%Y%m%dT%H%M}",
                created_at=now, underlying=view.underlying, version_id=v["version_id"], family=g.family,
                structure_key=cand.structure.key()[:200], structure=jsonable(cand.structure.summary()),
                score=verdict["score"], features=jsonable(f.as_dict() if f else {}), regime=view.regime,
                dte=cand.dte, iv_rank=f.iv_rank if f else None, mode=mode, status=status, gate=gate,
                reject_reason="; ".join(r for x in verdict["vetoes"] for r in x["reasons"])[:2000] if gate else None,
                data_quality=jsonable(view.quality),
                audit=jsonable({"metrics": cand.metrics, "verdict": verdict, "risk": item["risk"],
                                "comparison": comparison, "version": {k: v.get(k) for k in ("key", "stage", "expected_ror", "edge_basis")},
                                "quotes": [{"symbol": q.symbol, "bid": q.bid, "ask": q.ask, "age": q.age(now),
                                            "feed": q.feed} for q in cand.quotes]}),
            )  # fmt: skip
            s.add(row)
            await s.flush()
            thesis_id = None
            if not gate:
                t = OptionsTradeThesisRow(candidate_id=row.id, created_at=now, underlying=view.underlying,
                                          direction=FAMILIES[g.family].direction, market_regime=view.regime,
                                          iv_regime=view.vol_regime, thesis=text[:4000], body=jsonable(th),
                                          debate=jsonable(db_), explanation=text[:4000], confidence=verdict["confidence"])  # fmt: skip
                s.add(t)
                await s.flush()
                thesis_id = t.id
            return row.id, thesis_id

    async def _open_shadow(self, cyc: OptionsCycle, item: Mapping[str, Any], cand_id: int, thesis_id: int | None,
                           now: datetime) -> None:  # fmt: skip
        """Every chosen candidate is also traded in the shadow book at REALISTIC fills on live quotes: the
        evidence PAPER_ACTIVE needs, kept apart from paper results."""
        rec = _entry_record(item)
        legs = rec["structure"]["legs"]
        quotes = {q.symbol: q for q in item["cand"].quotes}
        entry = fill_value(legs, quotes, opening=True)
        if entry is None:
            return
        qty = item["intent"].qty if item["intent"] is not None else 1
        fees = self._s.options_fee_per_contract * sum(int(leg["ratio"]) for leg in legs) * qty
        async with self._db.session() as s:
            p = OptionsPositionRow(candidate_id=cand_id, thesis_id=thesis_id, version_id=item["v"]["version_id"],
                                   underlying=item["view"].underlying, family=item["g"].family, direction=rec["direction"],
                                   mode="shadow", structure=jsonable({**rec["structure"], "fees": {"open": fees}}),
                                   quantity=qty, status="open", expiry_state="OPEN", first_expiration=rec["first_expiration"],
                                   opened_at=now, entry_value=entry * qty, entry_mid=rec["entry_mid"],
                                   entry_underlying=rec["spot"], entry_iv=rec["iv"], entry_greeks=jsonable(rec["greeks"]),
                                   max_loss=rec["max_loss"] * qty, max_profit=(rec["max_profit"] * qty) if rec["max_profit"] else None,
                                   marks=[])  # fmt: skip
            s.add(p)
            await s.flush()
            await self._event(s, p.id, "opened", f"shadow {item['g'].family} × {qty} at {entry / 100:+.2f} per share "
                              "(REALISTIC fills on live quotes)", now)  # fmt: skip
            cyc.shadow_opened.append(
                {"position_id": p.id, "underlying": p.underlying, "family": p.family, "qty": qty}
            )

    async def _missed(self, item: Mapping[str, Any], cand_id: int, gate: str, now: datetime) -> None:
        cand = item["cand"]
        async with self._db.session() as s:
            s.add(OptionsMissedOpportunityRow(candidate_id=cand_id, created_at=now, underlying=item["view"].underlying,
                                              family=item["g"].family, reject_reason=gate, gate=gate,
                                              strategy_confidence=item["verdict"]["confidence"],
                                              grade_after=now.date() + timedelta(days=max(min(cand.dte, 21), 1)),
                                              details=jsonable({"structure": _entry_record(item)["structure"],
                                                                "spot": item["view"].spot,
                                                                "would_have_passed_risk": (item["risk"] or {}).get("approved")})))  # fmt: skip

    # ------------------------------------------------------------------ after the trading service
    async def after_execution(
        self, cyc: OptionsCycle, trades: Sequence[Mapping[str, Any]], *, reason: str | None = None
    ) -> None:
        """What happened to each option order: positions opened or closing, the execution ledger, candidates."""
        now = self._clock.now()
        by_key = {t.get("option_key"): t for t in trades if t.get("asset_class") == "us_option"}
        async with self._db.session() as s:
            for key, what in cyc.pending.items():
                t = by_key.get(key)
                status = t.get("status") if t else None
                sent = bool(t and t.get("alpaca_order_id"))
                if what["kind"] == "open":
                    cand = await s.get(OptionsTradeCandidateRow, what["candidate_id"])
                    if cand is not None:
                        cand.status = "submitted" if sent else (status or "not_sent")
                        if not sent:
                            cand.reject_reason = (
                                t.get("error") or t.get("risk") if t else reason
                            ) or "not sent"
                    if not sent or t is None:
                        continue
                    rec = what["item"]
                    fill = t.get("filled_avg_price")
                    qty = int(t.get("qty") or 1)
                    filled = status == "filled" and fill is not None
                    entry_value = (
                        _net_dollars(fill, rec["structure"], qty, opening=True)
                        if filled
                        else (t.get("limit_price") or 0) * 100 * qty
                    )
                    p = OptionsPositionRow(candidate_id=what["candidate_id"], thesis_id=what["thesis_id"],
                                           version_id=what["version_id"], underlying=rec["underlying"], family=rec["family"],
                                           direction=rec["direction"], mode="paper",
                                           structure=jsonable({**rec["structure"], "exploration": t.get("exploration"),
                                                               "fees": {"open": 0.0}}),
                                           quantity=qty, status="open" if filled else "pending", expiry_state="OPEN",
                                           first_expiration=rec["first_expiration"], opened_at=now, entry_value=entry_value,
                                           entry_mid=rec["entry_mid"], entry_underlying=rec["spot"], entry_iv=rec["iv"],
                                           entry_greeks=jsonable(rec["greeks"]), max_loss=rec["max_loss"] * qty,
                                           max_profit=(rec["max_profit"] * qty) if rec["max_profit"] else None,
                                           marks=[], client_order_id=t.get("client_order_id"))  # fmt: skip
                    s.add(p)
                    await s.flush()
                    await self._event(
                        s, p.id, "submitted", f"paper order {t.get('client_order_id')} {status}", now
                    )
                    self._ledger(s, p.id, what["candidate_id"], "open", t, rec, now)
                else:
                    pos = await s.get(OptionsPositionRow, what["position_id"])
                    if pos is None:
                        continue
                    if not sent or t is None:
                        await self._event(
                            s,
                            pos.id,
                            "exit_not_sent",
                            f"{what['reason']}: {(t or {}).get('risk') or reason}",
                            now,
                        )
                        continue
                    pos.exit_client_order_id = t.get("client_order_id")
                    pos.status = "closing"
                    self._ledger(s, pos.id, pos.candidate_id, "close", t, {"structure": pos.structure}, now)
                    if status == "filled" and t.get("filled_avg_price") is not None:
                        await self._close(s, cyc, pos, _net_dollars(t["filled_avg_price"], pos.structure, pos.quantity,
                                                                    opening=False), what["reason"], now, {})  # fmt: skip

    def _ledger(self, s: Any, pid: int, cand_id: int | None, action: str, t: Mapping[str, Any], rec: Mapping[str, Any],
                now: datetime) -> None:  # fmt: skip
        legs = _legs_of(rec["structure"])
        mid = sum(
            _sign(x["side"]) * int(x["ratio"]) * float(x.get("last_mid") or x.get("entry_mid") or 0)
            for x in legs
        )
        fill = t.get("filled_avg_price")
        slip = None if fill is None else (fill - mid if action == "open" else mid - fill)
        s.add(OptionsExecutionLedgerRow(position_id=pid, candidate_id=cand_id, client_order_id=t.get("client_order_id"),
                                        action=action, legs=jsonable(t.get("legs") or []), decision_at=now,
                                        submitted_at=_dt(t.get("submitted_at")), filled_at=now if t.get("status") == "filled" else None,
                                        decision_price=t.get("limit_price"), mid=round(mid, 4), limit_price=t.get("limit_price"),
                                        fill_price=fill, expected_price=t.get("limit_price"),
                                        slippage_dollars=None if slip is None else round(slip * 100 * float(t.get("qty") or 1), 2),
                                        latency_ms=t.get("submit_latency_ms"), status=str(t.get("status"))[:24]))  # fmt: skip

    # ------------------------------------------------------------------ learning inputs
    async def _weights(self) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        async with self._db.session() as s:
            rows = (await s.scalars(select(OptionsStrategyWeightRow))).all()
        return {(r.strategy_key, r.regime, r.structure, r.vol_state): {"weight": r.weight, "mean": r.mean, "n": r.n}
                for r in rows}  # fmt: skip

    async def _decays(self) -> dict[int, str]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(select(OptionsStrategyDecayRow).order_by(OptionsStrategyDecayRow.at))
            ).all()
        return {r.version_id: r.status for r in rows}

    async def learn(self) -> dict[str, Any]:
        """After the close: strategy weights by context (shrunk, recency-weighted) from shadow and paper trades,
        lessons that replicate, and missed opportunities graded (model-valued, labelled)."""
        now = self._clock.now()
        today = now.astimezone(NEW_YORK).date()
        async with self._db.session() as s:
            closed = (await s.scalars(select(OptionsPositionRow).where(OptionsPositionRow.status == "closed",
                                                                       OptionsPositionRow.realized_pnl.is_not(None)))).all()  # fmt: skip
            trades: list[dict[str, Any]] = [{"pnl": r.realized_pnl, "max_loss": r.max_loss, "family": r.family,
                       "regime": (r.structure.get("features") or {}).get("regime") or "",
                       "iv_regime": (r.structure.get("features") or {}).get("iv_regime") or "",
                       "exit_date": r.closed_at.date().isoformat() if r.closed_at else None,
                       "key": str(r.structure.get("strategy") or r.version_id), "mode": r.mode,
                       "critique": r.critique, "created_at": r.opened_at.isoformat()} for r in closed]  # fmt: skip
            updated = 0
            by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for t in trades:
                by_key[t["key"]].append(t)
            for key, ts in by_key.items():
                for (regime, fam, uclass, vol), w in learning.weights_by_context(ts, today).items():
                    row = await s.scalar(select(OptionsStrategyWeightRow).where(
                        OptionsStrategyWeightRow.strategy_key == key[:64], OptionsStrategyWeightRow.regime == regime,
                        OptionsStrategyWeightRow.structure == fam, OptionsStrategyWeightRow.underlying_class == uclass,
                        OptionsStrategyWeightRow.vol_state == vol))  # fmt: skip
                    if row is None:
                        row = OptionsStrategyWeightRow(strategy_key=key[:64], regime=regime, structure=fam,
                                                       underlying_class=uclass, vol_state=vol, weight=0, mean=0, sd=0, n=0,
                                                       updated_at=now)  # fmt: skip
                        s.add(row)
                    row.weight, row.mean, row.sd, row.n = w.weight, w.mean, w.sd, round(w.n)
                    row.evidence, row.updated_at = jsonable(w.as_dict()), now
                    updated += 1
            new_lessons = lessons.lessons_from(
                [t["critique"] for t in trades if t.get("critique")], today=today
            )
            for le in new_lessons:
                s.add(OptionsLessonRow(created_at=now, memory=le.get("memory", "OptionsLessonMemory"),
                                       observation=le.get("observation", ""), hypothesis=le.get("hypothesis", ""),
                                       evidence=jsonable(le.get("evidence", {})), confidence=le.get("confidence", 0.0),
                                       sample_size=le.get("sample_size", 0), applicability=jsonable(le.get("applicability", {})),
                                       relevance=1.0, status=le.get("status", "candidate"), source="trade"))  # fmt: skip
            graded = await self._grade_missed(s, today, now)
        meta = learning.meta_learn(trades)
        return {"weights": updated, "lessons": len(new_lessons), "missed_graded": graded, "meta": meta,
                "trades": {"shadow": sum(t["mode"] == "shadow" for t in trades), "paper": sum(t["mode"] == "paper" for t in trades)}}  # fmt: skip

    async def _grade_missed(self, s: Any, today: date, now: datetime) -> int:
        rows = (await s.scalars(select(OptionsMissedOpportunityRow).where(OptionsMissedOpportunityRow.graded_at.is_(None),
                                                                           OptionsMissedOpportunityRow.grade_after <= today))).all()  # fmt: skip
        n = 0
        for r in rows:
            spot = None
            try:
                from quantpulse.brain.options.perception import closes_of

                closes = await closes_of(self._market, r.underlying)
                spot = closes[max(d for d in closes if d <= r.grade_after)] if closes else None
            except Exception:
                spot = None
            legs = _legs_of((r.details or {}).get("structure") or {})
            outcome = None
            if spot is not None and legs:
                st = _structure(r.underlying, r.family, legs)
                when = datetime.combine(r.grade_after, datetime.min.time(), NEW_YORK)
                iv = float(((r.details or {}).get("structure") or {}).get("iv") or 0.3)
                outcome = round(st.pnl_at(spot, when, iv), 2) if st is not None else None
            r.outcome_pnl = outcome
            r.classification = lessons.classify_missed(outcome, rejected_for=r.reject_reason,
                                                       would_have_passed_risk=(r.details or {}).get("would_have_passed_risk"))  # fmt: skip
            r.graded_at = now
            r.details = jsonable(
                {**(r.details or {}), "valuation": "model (BSM at the entry IV) on the real underlying close"}
            )
            n += 1
        return n

    # ------------------------------------------------------------------ read models
    async def positions(
        self, *, mode: str | None = None, status: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            q = select(OptionsPositionRow).order_by(OptionsPositionRow.opened_at.desc()).limit(limit)
            if mode:
                q = q.where(OptionsPositionRow.mode == mode)
            if status:
                q = q.where(OptionsPositionRow.status == status)
            rows = (await s.scalars(q)).all()
        return [_position_out(r) for r in rows]

    async def candidates(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(OptionsTradeCandidateRow)
                    .order_by(OptionsTradeCandidateRow.created_at.desc())
                    .limit(limit)
                )
            ).all()
            theses = {t.candidate_id: t for t in (await s.scalars(select(OptionsTradeThesisRow).where(
                OptionsTradeThesisRow.candidate_id.in_([r.id for r in rows])))).all()} if rows else {}  # fmt: skip
        return [{"id": r.id, "at": r.created_at.isoformat(), "underlying": r.underlying, "family": r.family,
                 "version_id": r.version_id, "mode": r.mode, "status": r.status, "gate": r.gate, "score": r.score,
                 "dte": r.dte, "iv_rank": r.iv_rank, "regime": r.regime, "reject_reason": r.reject_reason,
                 "structure": r.structure, "explanation": theses[r.id].explanation if r.id in theses else None,
                 "thesis": theses[r.id].body if r.id in theses else None, "debate": theses[r.id].debate if r.id in theses else None,
                 "agents": (r.audit or {}).get("verdict", {}).get("opinions"), "comparison": (r.audit or {}).get("comparison")}
                for r in rows]  # fmt: skip

    async def performance(self) -> dict[str, Any]:
        """Shadow and paper kept apart (research is model-priced and reported by the lab)."""
        from quantpulse.options.lab.metrics import trade_stats

        async with self._db.session() as s:
            rows = (await s.scalars(select(OptionsPositionRow).where(OptionsPositionRow.status == "closed",
                                                                     OptionsPositionRow.realized_pnl.is_not(None)))).all()  # fmt: skip
        out: dict[str, Any] = {}
        for mode in ("shadow", "paper"):
            ts = [r for r in rows if r.mode == mode]
            out[mode] = {**trade_stats([r.realized_pnl or 0 for r in ts], [r.max_loss or 0 for r in ts]),
                         "by_family": {f: len([r for r in ts if r.family == f]) for f in sorted({r.family for r in ts})},
                         "attribution": _sum_attr(ts)}  # fmt: skip
        out["note"] = ("shadow = simulated on live quotes at REALISTIC fills; paper = real Alpaca paper orders; "
                       "never mixed")  # fmt: skip
        return jsonable(out)

    async def counterfactuals(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(OptionsCounterfactualRow)
                    .order_by(OptionsCounterfactualRow.created_at.desc())
                    .limit(limit)
                )
            ).all()
        return [{"position_id": r.position_id, "alternative": r.alternative, "pnl": r.pnl, "chosen_pnl": r.chosen_pnl,
                 "better_than_chosen": r.better_than_chosen, "detail": r.structure, "data": r.data_source,
                 "at": r.created_at.isoformat()} for r in rows]  # fmt: skip

    async def missed(self, limit: int = 100) -> dict[str, Any]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(OptionsMissedOpportunityRow)
                    .order_by(OptionsMissedOpportunityRow.created_at.desc())
                    .limit(limit)
                )
            ).all()
        items = [{"id": r.id, "underlying": r.underlying, "family": r.family, "gate": r.gate, "reason": r.reject_reason,
                  "grade_after": r.grade_after.isoformat(), "outcome_pnl": r.outcome_pnl, "classification": r.classification,
                  "at": r.created_at.isoformat()} for r in rows]  # fmt: skip
        return {
            "items": items,
            "summary": lessons.missed_summary([{"classification": i["classification"]} for i in items]),
        }

    async def greeks(self) -> dict[str, Any]:
        """The option book's Greeks from the latest marks (paper and shadow apart)."""
        async with self._db.session() as s:
            rows = (
                await s.scalars(select(OptionsPositionRow).where(OptionsPositionRow.status == "open"))
            ).all()
        out: dict[str, Any] = {}
        for mode in ("paper", "shadow"):
            tot = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
            per = []
            for r in rows:
                if r.mode != mode or not r.marks:
                    continue
                m = r.marks[-1]
                for k in tot:
                    tot[k] += float(m.get(k) or 0.0)
                per.append({"position_id": r.id, "underlying": r.underlying, "family": r.family,
                            **{k: m.get(k) for k in tot}, "value": m.get("value"), "day": m.get("day")})  # fmt: skip
            out[mode] = {"total": {k: round(v, 2) for k, v in tot.items()}, "positions": per}
        return out


# --------------------------------------------------------------------------- helpers
def _stock_view(cons: Any) -> dict[str, Any] | None:
    if cons is None:
        return None
    stance = getattr(getattr(cons, "stance", None), "value", None) or (
        cons.get("stance") if isinstance(cons, dict) else None
    )
    score_ = getattr(cons, "score", None) if not isinstance(cons, dict) else cons.get("score")
    conf = getattr(cons, "confidence", None) if not isinstance(cons, dict) else cons.get("confidence")
    if stance is None:
        return None
    signed = (score_ or 0.0) * (conf or 0.0)
    return {"stance": stance, "score": score_, "confidence": conf, "score_signed": signed}


def _entry_record(item: Mapping[str, Any]) -> dict[str, Any]:
    cand: Candidate = item["cand"]
    g: Genome = item["g"]
    legs = []
    for leg, q in zip(cand.structure.option_legs, cand.quotes, strict=True):
        c = leg.contract
        assert c is not None
        legs.append({"symbol": q.symbol, "side": leg.side, "ratio": leg.ratio, "kind": c.kind, "strike": c.strike,
                     "expiration": c.expiration.isoformat(), "entry_mid": q.mid, "last_mid": q.mid})  # fmt: skip
    m = cand.metrics
    view: UnderlyingView = item["view"]
    return {
        "underlying": view.underlying,
        "family": g.family,
        "direction": FAMILIES[g.family].direction,
        "spot": view.spot or 0.0,
        "iv": m.get("iv"),
        "greeks": m.get("greeks") or {},
        "entry_mid": cand.structure.debit() / 100,
        "max_loss": float(m.get("max_loss") or cand.structure.max_loss()),
        "max_profit": m.get("max_profit"),
        "first_expiration": cand.expiration,
        "structure": {"family": g.family, "legs": legs, "genome": g.canonical(), "strategy": item["v"]["key"],
                      "features": {**(view.features.as_dict() if view.features else {}), "regime": view.regime,
                                   "iv_regime": view.vol_regime}, "iv": m.get("iv"),
                      "expected": {k: m.get(k) for k in ("expected_on_risk", "pop", "expected_pnl")}},
    }  # fmt: skip


def _net_dollars(price: float | None, structure: Mapping[str, Any], qty: int, *, opening: bool) -> float:
    """A fill's net price per share → dollars: paid to open (positive a debit), received on close."""
    if price is None:
        return 0.0
    legs = _legs_of(structure)
    if len(legs) == 1:
        s = _sign(legs[0]["side"])
        # single leg: a positive price; a long leg is paid for when opened and paid out when closed, a short leg
        # the other way round — in both directions the leg's sign times the price
        return s * price * 100 * qty
    # multi-leg: the net price is signed (positive a debit); a close's debit is money paid
    return (price if opening else -price) * 100 * qty


def _closing_intent(
    p: OptionsPositionRow, quotes: Mapping[str, OptionQuote], spot: float, reason: str
) -> OptionOrderIntent | None:
    legs = _legs_of(p.structure)
    intents = []
    natural = 0.0
    for leg in legs:
        q = quotes.get(leg["symbol"])
        side = "sell" if leg["side"] == "long" else "buy"
        px = (q.bid if side == "sell" else q.ask) if q is not None else None
        if px is None or px <= 0:
            if side == "sell":
                px = 0.01  # a worthless long leg: offered at a cent (it may simply expire)
            else:
                return None
        natural += (1 if side == "buy" else -1) * int(leg["ratio"]) * px
        intents.append(OptionLegIntent(leg["symbol"], side, int(leg["ratio"]),
                                       "sell_to_close" if side == "sell" else "buy_to_close"))  # fmt: skip
    limit = round(natural, 2) if len(legs) > 1 else round(abs(natural), 2)
    return OptionOrderIntent(p.underlying, p.family, tuple(intents), p.quantity, limit, "exit", reason[:300], False, spot,
                             strategy_key=str(p.structure.get("strategy") or ""))  # fmt: skip


def _structure(underlying: str, family: str, legs: Sequence[Mapping[str, Any]]) -> Structure | None:
    try:
        built = [Leg("long" if x["side"] == "long" else "short", int(x["ratio"]), float(x.get("entry_mid") or 0.0),
                     parse_occ(x["symbol"])) for x in legs]  # fmt: skip
        return Structure(family, underlying, tuple(built))
    except (ContractError, ValueError, KeyError):
        return None


def _dt(x: Any) -> datetime | None:
    if isinstance(x, datetime):
        return x
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _sum_attr(rows: Sequence[OptionsPositionRow]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for r in rows:
        for k, v in (r.attribution or {}).items():
            if isinstance(v, int | float):
                out[k] += float(v)
    return {k: round(v, 2) for k, v in out.items()}


def _position_out(r: OptionsPositionRow) -> dict[str, Any]:
    last = r.marks[-1] if r.marks else {}
    unreal = (float(last.get("value") or 0) - r.entry_value) if r.status == "open" and last else None
    return {"id": r.id, "mode": r.mode, "status": r.status, "underlying": r.underlying, "family": r.family,
            "direction": r.direction, "quantity": r.quantity, "legs": _legs_of(r.structure),
            "strategy": r.structure.get("strategy"), "exploration": bool(r.structure.get("exploration")),
            "opened_at": r.opened_at.isoformat(), "first_expiration": r.first_expiration.isoformat() if r.first_expiration else None,
            "expiry_state": r.expiry_state, "entry_value": r.entry_value, "max_loss": r.max_loss, "max_profit": r.max_profit,
            "mark": last, "unrealized_pnl": None if unreal is None else round(unreal, 2),
            "closed_at": r.closed_at.isoformat() if r.closed_at else None, "exit_reason": r.exit_reason,
            "realized_pnl": r.realized_pnl, "attribution": r.attribution, "critique": r.critique,
            "client_order_id": r.client_order_id, "exit_client_order_id": r.exit_client_order_id}  # fmt: skip


def _contract(symbol: str) -> OptionContract | None:
    try:
        return parse_occ(symbol)
    except ContractError:
        return None
