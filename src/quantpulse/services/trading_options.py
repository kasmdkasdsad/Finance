"""Option orders inside the trading service — the same service, the same cycle, the same risk book and last
gate as every stock order; nothing here can reach Alpaca any other way.

:class:`OptionsExecution` is mixed into :class:`~quantpulse.services.trading.TradingService`. It adds only
what options need on top of the existing path:

* **fresh execution quotes** for every leg, read just before the order (never the Brain's snapshot): bid and
  ask from the options feed, open interest from the contracts list, and the implied volatility, delta and
  vega computed here from the mid and the underlying's live price (deterministic arithmetic, never a model's
  say-so). A leg whose quote is model-priced, recorded, stale or one-sided never reaches an order;
* **closing intents** for held contracts — grouped by underlying and expiration, one multi-leg order where
  Alpaca allows it (two to four legs), otherwise short legs bought back before long legs are sold, so a
  close never leaves a naked short behind;
* :meth:`OptionsExecution._risk_and_submit_option` — the risk book's verdict
  (:meth:`~quantpulse.services.trading_risk.RiskBook.evaluate_option`), then the same
  :meth:`~quantpulse.services.trading.TradingService.pre_submit_blockers` last gate, then the order manager's
  exactly-once :meth:`~quantpulse.services.order_manager.OrderManager.submit_option`.

Exercise is never requested: QuantPulse closes positions before expiration (``QP_OPTIONS_CLOSE_DTE``) and
has no exercise call at all.
"""

from __future__ import annotations

import logging
import time as _time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from quantpulse.db import repositories as repo
from quantpulse.options.contracts import ContractError, parse_occ
from quantpulse.options.data import OptionsMarketDataProvider
from quantpulse.options.pricing import greeks as bsm_greeks
from quantpulse.options.pricing import implied_vol
from quantpulse.providers.alpaca_trading import BrokerPosition
from quantpulse.schemas.trading import ProposedTradeOut, RiskCheckOut
from quantpulse.services.order_manager import STRATEGY, Submission, option_client_order_id, trade_stage
from quantpulse.services.trading_risk import (
    OptionLegIntent,
    OptionLegQuote,
    OptionOrderIntent,
    RiskBook,
    RiskDecision,
)

if TYPE_CHECKING:
    from quantpulse.config import Settings
    from quantpulse.core.clock import Clock
    from quantpulse.db.session import Database
    from quantpulse.services.order_manager import OrderManager

logger = logging.getLogger(__name__)
RATE = 0.04  # the risk-free rate assumed for the Greeks the risk book reads (its effect on delta is tiny)


def option_trade_out(o: OptionOrderIntent, decision: RiskDecision, mode: str, cid: str) -> ProposedTradeOut:
    return ProposedTradeOut(
        symbol=o.symbol,
        side=o.side,
        qty=float(o.qty),
        est_price=o.est_price,
        notional=o.notional,
        kind=o.kind,
        reason=o.reason[:300],
        current_weight=0.0,
        target_weight=0.0,
        score=o.score,
        approved=decision.approved,
        risk=decision.summary,
        checks=[RiskCheckOut(name=c.name, passed=c.passed, detail=c.detail) for c in decision.checks],
        order_type="limit",
        limit_price=o.limit_price,
        client_order_id=cid if decision.approved and mode == "paper" else None,
        status="risk_rejected"
        if not decision.approved
        else ("dry_run" if mode != "paper" else "not_submitted"),
        stage="risk_approved" if decision.approved else "risk_rejected",
        asset_class="us_option",
        family=o.family,
        option_key=option_key(o),
        legs=[
            {"symbol": x.symbol, "side": x.side, "ratio": x.ratio, "position_intent": x.position_intent}
            for x in o.legs
        ],
        exploration=o.exploration,
    )


def option_key(o: OptionOrderIntent) -> str:
    """How the Brain finds its option decision again in a cycle's trades: the legs, the quantity and the
    direction (open or close)."""
    legs = ",".join(f"{x.side}:{x.ratio}:{x.symbol}" for x in sorted(o.legs, key=lambda x: x.symbol))
    return f"{'open' if o.opening else 'close'}|{o.qty}|{legs}"


def closing_intents(
    positions: Mapping[str, BrokerPosition],
    quotes: Mapping[str, OptionLegQuote],
    spots: Mapping[str, float],
    *,
    kind: str,
    reason: str,
    intent: str = "strategy",
    only: set[str] | None = None,
) -> list[OptionOrderIntent]:
    """Orders that close held contracts at their natural prices (asks paid to buy back, bids received to
    sell). Legs of one underlying and expiration close together as one order when there are two to four of
    them; otherwise one by one, short legs first — a close can never leave a naked short behind."""
    groups: dict[tuple[str, str], list[BrokerPosition]] = defaultdict(list)
    for sym, p in positions.items():
        if only is not None and sym not in only:
            continue
        try:
            c = parse_occ(sym)
        except ContractError:
            continue
        groups[(c.underlying, c.expiration.isoformat())].append(p)
    out: list[OptionOrderIntent] = []
    for (u, _exp), held in sorted(groups.items()):
        held = [p for p in held if abs(p.qty) >= 1]
        if not held:
            continue
        units = min(int(abs(p.qty)) for p in held)
        if 2 <= len(held) <= 4 and all(int(abs(p.qty)) % units == 0 for p in held):
            legs = tuple(
                OptionLegIntent(p.symbol, "sell" if p.qty > 0 else "buy", int(abs(p.qty)) // units,
                                "sell_to_close" if p.qty > 0 else "buy_to_close")
                for p in held
            )  # fmt: skip
            out.append(_close(u, legs, units, quotes, spots, kind, reason, intent))
            continue
        for p in sorted(held, key=lambda p: p.qty):  # shorts (negative) first
            leg = OptionLegIntent(p.symbol, "sell" if p.qty > 0 else "buy", 1,
                                  "sell_to_close" if p.qty > 0 else "buy_to_close")  # fmt: skip
            out.append(_close(u, (leg,), int(abs(p.qty)), quotes, spots, kind, reason, intent))
    return out


def _close(
    underlying: str,
    legs: tuple[OptionLegIntent, ...],
    qty: int,
    quotes: Mapping[str, OptionLegQuote],
    spots: Mapping[str, float],
    kind: str,
    reason: str,
    intent: str,
) -> OptionOrderIntent:
    natural = 0.0
    for x in legs:
        q = quotes.get(x.symbol)
        px = (q.ask if x.side == "buy" else q.bid) if q is not None else None
        natural += (1 if x.side == "buy" else -1) * x.ratio * (px or 0.0)
    if len(legs) == 1 and legs[0].side == "sell":
        natural = max(abs(natural), 0.01)  # a single sell: its (positive) price
    elif len(legs) == 1:
        natural = max(natural, 0.01)
    family = "close"
    return OptionOrderIntent(underlying, family, legs, qty, round(natural, 2), kind, reason, False,
                             spots.get(underlying, 0.0), intent=intent)  # fmt: skip


class OptionsExecution:
    """The option half of the trading service (mixed into it; see the module docstring)."""

    _s: Settings
    _db: Database
    _clock: Clock
    orders: OrderManager
    options_data: OptionsMarketDataProvider | None

    async def pre_submit_blockers(
        self, owner: str, *, flatten: bool = False
    ) -> list[str]:  # pragma: no cover - TradingService's
        raise NotImplementedError

    # ------------------------------------------------------------------ fresh quotes
    async def option_leg_quotes(
        self, symbols: Sequence[str], spots: Mapping[str, float], now: datetime
    ) -> dict[str, OptionLegQuote]:
        """Execution quotes for ``symbols``: the options feed's latest bid/ask, open interest from the
        contracts list, and the IV, delta and vega computed here (see the module docstring)."""
        symbols = sorted(set(symbols))
        if not symbols or self.options_data is None or not self.options_data.configured():
            return {}
        raw = await self.options_data.latest_quotes(symbols)
        oi: dict[str, float | None] = {}
        by_expiry: dict[tuple[str, Any], list[float]] = defaultdict(list)
        for sym in symbols:
            try:
                c = parse_occ(sym)
            except ContractError:
                continue
            by_expiry[(c.underlying, c.expiration)].append(c.strike)
        for (u, exp), strikes in by_expiry.items():
            try:
                infos = await self.options_data.contracts(u, expiration_from=exp, expiration_to=exp,
                                                          strike_from=min(strikes), strike_to=max(strikes))  # fmt: skip
            except Exception as exc:  # open interest stays unknown: the risk book fails such legs closed
                logger.warning("open interest for %s %s unavailable: %s", u, exp, exc)
                continue
            for info in infos:
                oi[info.contract.symbol] = info.open_interest
        out: dict[str, OptionLegQuote] = {}
        for sym in symbols:
            q = raw.get(sym)
            if q is None:
                continue
            c = q.contract
            spot = spots.get(c.underlying)
            iv = delta = vega = None
            mid = q.mid
            if mid is not None and spot and spot > 0:
                years = c.years(now)
                iv = implied_vol(c.kind, mid, spot, c.strike, years, RATE)
                if iv is not None and years > 0:
                    v = bsm_greeks(c.kind, spot, c.strike, years, iv, RATE)
                    delta, vega = v.delta, v.vega
            out[sym] = OptionLegQuote(
                bid=q.bid,
                ask=q.ask,
                age_seconds=q.age(now),
                feed=q.feed,
                open_interest=oi.get(sym, q.open_interest),
                iv=iv,
                delta=delta,
                vega=vega,
            )
        return out

    # ------------------------------------------------------------------ risk, the last gate, the order
    async def _risk_and_submit_option(
        self,
        o: OptionOrderIntent,
        book: RiskBook,
        slot: str,
        mode: str,
        cycle_id: int | None,
        *,
        strategy: str = STRATEGY,
    ) -> tuple[ProposedTradeOut, Submission | None]:
        decision = book.evaluate_option(o)
        cid = option_client_order_id(slot, o)
        async with self._db.session() as s:
            now = self._clock.now()
            legs = "; ".join(f"{x.position_intent.replace('_', ' ')} {x.ratio}× {x.symbol}" for x in o.legs)
            await repo.add_trading_event(s, "trade_proposed", f"{o.family} {o.qty}× {o.underlying} ({o.kind}"
                                         f"{', exploration' if o.exploration else ''}): {legs} @ {o.limit_price:+.2f}",
                                         now, cycle_id=cycle_id, symbol=o.symbol,
                                         details={"notional": round(o.notional, 2), "score": o.score,
                                                  "family": o.family, "exploration": o.exploration})  # fmt: skip
            await repo.add_trading_event(s, "risk_approved" if decision.approved else "risk_rejected",
                                         f"{o.symbol} {o.side}: {decision.summary}", now, cycle_id=cycle_id,
                                         symbol=o.symbol, details={"checks": [c.name for c in decision.failures]})  # fmt: skip
        base = option_trade_out(o, decision, mode, cid)
        if not decision.approved or mode != "paper":
            if decision.approved:
                book.commit(o)
            return base, None
        book.commit(o)
        owner = "brain" if strategy != STRATEGY else "strategy"
        late = await self.pre_submit_blockers(owner, flatten=o.intent == "flatten")
        if o.opening and not (self._s.options_enabled and self._s.options_execution):
            late.append("option execution is switched off (QP_OPTIONS_ENABLED / QP_OPTIONS_EXECUTION)")
        if late:
            async with self._db.session() as s:
                await repo.add_trading_event(s, "order_blocked_at_submit", f"{o.symbol} not sent: " + "; ".join(late),
                                             self._clock.now(), cycle_id=cycle_id, symbol=o.symbol,
                                             details={"blockers": late})  # fmt: skip
            return base.model_copy(update={"status": "blocked_at_submit", "client_order_id": None,
                                           "error": "; ".join(late)}), None  # fmt: skip
        started = _time.perf_counter()
        sub = await self.orders.submit_option(o, cid=cid, cycle_id=cycle_id, strategy=strategy)
        latency_ms = round((_time.perf_counter() - started) * 1000, 1)
        order = sub.order
        return (
            base.model_copy(
                update={
                    "status": sub.status,
                    "stage": trade_stage(True, sub.status),
                    "alpaca_order_id": sub.alpaca_order_id,
                    "filled_qty": order.filled_qty if order is not None else None,
                    "filled_avg_price": order.filled_avg_price if order is not None else None,
                    "submitted_at": order.submitted_at if order is not None else None,
                    "error": sub.error,
                    "submit_latency_ms": latency_ms,
                }
            ),
            sub,
        )
