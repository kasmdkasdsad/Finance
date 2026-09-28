"""Perception: build the cycle's :class:`~quantpulse.brain.context.BrainContext` from existing services.

Reused, not recomputed:

* :class:`~quantpulse.services.trading_data.TradingDataLoader` — the liquid universe, warehouse-first daily
  bars, validated live quotes (IEX vs SIP spreads, stale/off-market bid/asks), the stock model snapshot
  (live z-scores and point-in-time fundamentals), the VIX, implied volatility and earnings dates;
* :func:`~quantpulse.domain.trading_signals.raw_signals` (inside :func:`~quantpulse.brain.indicators.compute_indicators`);
* :func:`~quantpulse.domain.trading_regime.classify` — the same regime classifier the strategy uses;
* the Alpaca paper account, through the read-only :class:`~quantpulse.brain.context.BrokerView`;
* the trading service's kill switch (read only) and the risk limits from settings.

Every failure is recorded (``provider_errors``) and the cycle continues with less information: a broker
outage leaves research running but makes every action non-executable.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, Session, is_market_open, session_at
from quantpulse.domain import trading_regime as regime_mod
from quantpulse.domain import trading_signals as ts
from quantpulse.domain.fundamental_factors import FUNDAMENTAL_FEATURES
from quantpulse.providers.alpaca_trading import BrokerError
from quantpulse.schemas.common import DataStatus
from quantpulse.services.market import MarketService
from quantpulse.services.model import ModelService, ModelSnapshot
from quantpulse.services.options import OptionsService
from quantpulse.services.reference import ReferenceService
from quantpulse.services.trading import TradingService
from quantpulse.services.trading_data import LiveQuote, TradingDataLoader, TradingInputs
from quantpulse.services.trading_risk import RiskLimits

from .book import PaperBook
from .context import BrainContext, BrokerView, PortfolioState, brain_session
from .data_health import closed_reason, diagnose, feed_report
from .indicators import compute_indicators, market_statistics
from .opportunities import Opportunity, scan_focus, scan_universe
from .research_data import earnings_events, option_metrics
from .types import BrainMode, BrainSession

logger = logging.getLogger(__name__)
FOCUS_SCREEN = ("mom_3m", "rel_strength", "persistence", "px_vs_sma50", "mom_accel")


def choose_focus(
    indicators: pd.DataFrame,
    held: Sequence[str],
    requested: Sequence[str],
    quotes: dict[str, LiveQuote],
    market_open: bool,
    n: int,
    exclude: Sequence[str] = (),
    opportunities: Sequence[Opportunity] = (),
    max_opportunities: int = 0,
) -> tuple[list[str], dict[str, str]]:
    """Which symbols to study closely this cycle: requested ones, every holding, the lead symbols of the
    strongest detected opportunities (up to ``max_opportunities``), and the ``n`` strongest of a cheap
    cross-sectional pre-screen (momentum, relative strength, persistence, trend, acceleration). The rest of
    the universe is not analysed symbol by symbol (cost control)."""
    reasons: dict[str, str] = {}
    for s in requested:
        reasons.setdefault(s, "requested")
    for s in held:
        reasons.setdefault(s, "held position")
    added = 0
    for o in opportunities:
        if added >= max_opportunities:
            break
        lead = o.lead
        if lead is None or lead in exclude or (market_open and lead not in quotes):
            continue
        if lead not in reasons:
            reasons[lead] = f"opportunity: {o.headline}"
            added += 1
    cols = [c for c in FOCUS_SCREEN if c in indicators.columns]
    pool = [s for s in indicators.index if s not in reasons and s not in exclude]
    if market_open:
        pool = [s for s in pool if s in quotes]  # only tradeable names during the session
    if cols and pool:
        frame = indicators.loc[pool, cols].apply(pd.to_numeric, errors="coerce")
        z = (frame - frame.mean()) / frame.std().replace(0, pd.NA)
        screen = z.clip(-3, 3).mean(axis=1, skipna=True).dropna().sort_values(ascending=False)
        for s in list(screen.index[:n]):
            reasons[str(s)] = f"pre-screen rank {list(screen.index).index(s) + 1} (score {screen[s]:+.2f})"
    return list(reasons), reasons


class Perception:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        data: TradingDataLoader,
        broker: BrokerView,
        trading: TradingService,
        reference: ReferenceService | None = None,
        model: ModelService | None = None,
        options: OptionsService | None = None,
        market: MarketService | None = None,
        book: PaperBook | None = None,
    ) -> None:
        self._s = settings
        self._clock = clock
        self._data = data
        self._broker = broker
        self._trading = trading
        self._reference = reference
        self._model = model
        self._options = options
        self._market = market
        self._book = book  # the Brain's paper book: the portfolio its decisions manage

    async def _portfolio(self, errors: dict[str, str]) -> PortfolioState:
        if not self._broker.configured():
            errors["broker"] = "Alpaca paper keys are not configured"
            return PortfolioState(available=False, error=errors["broker"])
        try:
            account = await self._broker.account()
            positions = await self._broker.positions()
            orders = await self._broker.open_orders()
        except BrokerError as exc:
            errors["broker"] = str(exc)
            return PortfolioState(available=False, error=str(exc))
        return PortfolioState(
            available=True,
            account=account,
            positions={p.symbol: p for p in positions},
            open_orders=[o for o in orders if o.is_open],
        )

    async def _market_open(self, now: datetime, errors: dict[str, str]) -> tuple[bool, str, float | None]:
        """Whether the market is open (Alpaca's clock is authoritative), and how far this computer's clock
        is from Alpaca's (seconds, positive when ours is ahead; ``None`` without a broker clock)."""
        if self._broker.configured():
            try:
                before = self._clock.now()
                clock = await self._broker.clock()
                after = self._clock.now()
                local = before + (after - before) / 2  # the request's midpoint: network time cancels out
                return clock.is_open, "alpaca", (local - clock.timestamp).total_seconds()
            except BrokerError as exc:
                errors["clock"] = str(exc)
        return is_market_open(now), "calendar", None

    async def _sectors(self, errors: dict[str, str]) -> dict[str, str]:
        sectors = dict.fromkeys(self._s.trading_etfs, "ETF")
        if self._reference is None:
            return sectors
        try:
            membership = (await self._reference.membership()).value
            sectors.update(membership.gics)
        except Exception as exc:  # optional context
            errors["sectors"] = f"{type(exc).__name__}: {exc}"
        return sectors

    async def _stock_model(self, inputs: TradingInputs, errors: dict[str, str]) -> ModelSnapshot | None:
        """The stock model's latest completed run (the trading loader's, or asked for directly when the
        strategy's weights did not need it). Never waits for training; synthetic runs are not evidence."""
        if inputs.model is not None:
            return inputs.model
        if self._model is None or not self._s.brain_use_stock_model:
            return None
        try:
            snap = await asyncio.wait_for(
                self._model.trading_snapshot(wait=0), self._s.brain_research_timeout_seconds
            )
        except Exception as exc:  # training, unavailable or slow: the model agents abstain
            errors["stock_model"] = f"{type(exc).__name__}: {exc}"[:300]
            return None
        if snap.data_status is DataStatus.SYNTHETIC:
            errors["stock_model"] = "the stock model only has synthetic prices"
            return None
        return snap

    async def _research(
        self, focus: Sequence[str], inputs: TradingInputs, close: pd.DataFrame, errors: dict[str, str]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Option-chain metrics and earnings events for the focus stocks (bounded; failures recorded)."""
        stocks = [s for s in focus if s not in self._s.trading_etfs]
        today = self._clock.now().astimezone(NEW_YORK).date()
        timeout = self._s.brain_research_timeout_seconds
        jobs: dict[str, Any] = {}
        if self._options is not None and self._s.brain_options_analysis and self._s.enable_live_data:
            spots = {s: inputs.quotes[s].price for s in stocks if s in inputs.quotes}
            jobs["options"] = option_metrics(self._options, list(spots), spots, today, timeout)
        if self._reference is not None and self._s.brain_catalyst_analysis:
            jobs["events"] = earnings_events(self._reference, stocks, close, inputs.benchmark, today, timeout)
        got = dict(zip(jobs, await asyncio.gather(*jobs.values()), strict=True))
        for kind, (found, failed) in got.items():
            for sym, why in failed.items():
                errors[f"{kind}:{sym}"] = why[:200]
            label = "live option chains" if kind == "options" else "earnings calendars"
            self._notes.append(
                f"{label}: {len(found)} of {len(stocks)} focus stocks (only real data is used)"
            )
        return got.get("options", ({}, {}))[0], got.get("events", ({}, {}))[0]

    def _regime(self, inputs: TradingInputs) -> regime_mod.MarketRegime:
        """The strategy's own regime classification (same inputs as the trading service)."""
        spy, qqq = inputs.benchmark, inputs.qqq
        if inputs.session_open:
            day = pd.Timestamp(inputs.as_of.astimezone(NEW_YORK).date())
            bq = inputs.quotes.get(self._s.benchmark_symbol)
            if bq is not None:
                spy = pd.concat([spy, pd.Series([bq.price], index=[day])])
            qq = inputs.quotes.get("QQQ")
            if qq is not None and qqq is not None:
                qqq = pd.concat([qqq, pd.Series([qq.price], index=[day])])
        stocks = [c for c in inputs.close.columns if c not in self._s.trading_etfs]
        b50, b200 = regime_mod.breadth(inputs.close[stocks]) if stocks else (None, None)
        return regime_mod.classify(spy, qqq, breadth_50=b50, breadth_200=b200, vix=inputs.vix)

    async def perceive(
        self,
        mode: BrainMode,
        requested: Sequence[str] = (),
        previous_regime: str | None = None,
        *,
        pre_screen: float = 1.0,
        opportunity_budget: float = 1.0,
    ) -> BrainContext:
        self._notes: list[str] = []
        now = self._clock.now()
        errors: dict[str, str] = {}
        account = await self._portfolio(errors)  # the Alpaca paper account (read here; orders go via trading)
        market_open, clock_source, skew = await self._market_open(now, errors)
        # the portfolio the decisions manage: the Alpaca paper account when the Brain owns it, else its book
        owns = mode is BrainMode.PAPER_EXECUTION
        book = await self._book.load() if self._book is not None and not owns else None
        held = (
            [s for s, p in book.positions.items() if p.qty > 0]
            if book is not None
            else [s for s, p in account.positions.items() if p.qty > 0]
        )
        requested = [s.strip().upper() for s in requested if s.strip()]
        inputs = await self._data.load(list(dict.fromkeys([*held, *requested])))

        def last_price(symbol: str) -> float | None:
            q = inputs.quotes.get(symbol)
            if q is not None:
                return q.price
            col = inputs.close[symbol].dropna() if symbol in inputs.close.columns else None
            return float(col.iloc[-1]) if col is not None and len(col) else None

        portfolio = PaperBook.portfolio(book, last_price) if book is not None else account

        live_row = inputs.session_open and market_open and bool(inputs.quotes)
        close, high, low, volume = inputs.close, inputs.high, inputs.low, inputs.volume
        bench = inputs.benchmark
        if live_row:
            day = pd.Timestamp(now.astimezone(NEW_YORK).date())
            quoted = [c for c in close.columns if c in inputs.quotes]
            bars = {s: inputs.quotes[s].bar() for s in quoted}
            close, high, low, volume = ts.append_live_row(close, high, low, volume, bars, day)
            bq = inputs.quotes.get(self._s.benchmark_symbol)
            bench = pd.concat([bench, pd.Series([bq.price if bq else float("nan")], index=[day])])
        indicators = compute_indicators(
            close,
            high,
            low,
            volume,
            bench,
            live_row=live_row,
            vwap={s: q.vwap for s, q in inputs.quotes.items() if q.vwap} if live_row else None,
            session_fraction=inputs.session_fraction,
        )
        sectors = await self._sectors(errors)
        model = await self._stock_model(inputs, errors)
        regime = self._regime(inputs)
        exclude = [self._s.benchmark_symbol, "QQQ", *self._s.trading_etfs]
        opportunities = scan_universe(
            indicators,
            close,
            market_open=market_open,
            sectors=sectors,
            held=held,
            features=model.features if model is not None else None,
            regime=regime.label,
            previous_regime=previous_regime,
            trend_score=regime.trend_score,
            exclude=exclude,
            session_fraction=inputs.session_fraction,
            portfolio=portfolio,  # the Brain's book: its holdings' alerts and portfolio-level risk
            limits=RiskLimits.from_settings(self._s),
            vix=inputs.vix,
        )
        focus, reasons = choose_focus(
            indicators,
            held,
            requested,
            inputs.quotes,
            market_open,
            round(self._s.brain_focus_candidates * pre_screen),
            exclude=exclude,
            opportunities=opportunities,
            max_opportunities=round(self._s.brain_max_opportunities * opportunity_budget),
        )
        await self._data.enrich(inputs, focus)  # implied vol and earnings dates for the focus set only
        if inputs.implied_vol:
            iv = pd.Series(inputs.implied_vol, dtype=float)
            indicators["implied_vol"] = iv.reindex(indicators.index)
            rv = pd.to_numeric(indicators.get("rv63"), errors="coerce")
            indicators["iv_premium"] = indicators["implied_vol"] / rv.where(rv > 0) - 1

        options, events = await self._research(focus, inputs, inputs.close, errors)
        for sym, (nxt, source) in inputs.earnings.items():  # the trading loader's calendar, when it has one
            events.setdefault(
                sym,
                {
                    "next": nxt.isoformat(),
                    "next_source": source,
                    "days_to_next": (nxt - now.astimezone(NEW_YORK).date()).days,
                },
            )
        opportunities = sorted(
            [*opportunities, *scan_focus(focus, options, events)], key=lambda o: -o.strength
        )[: self._s.brain_max_opportunities_recorded]

        kill = await self._trading.kill_switch()
        # why a Brain order would not reach Alpaca right now (the executor asks again before sending)
        blockers = await self._trading.submit_blockers(kill, owner="brain")
        closed = None
        if not market_open:
            closed = closed_reason(now) or (
                "unscheduled closure (Alpaca says closed)"
                if session_at(now) is Session.REGULAR
                else "after hours"
            )
        feeds = self._market.feed_status() if self._market is not None else []
        refused = next((f["stock_feed"] for f in feeds if f.get("stock_feed_error")), None)
        diagnoses = {
            s: diagnose(
                s,
                quote=inputs.quotes.get(s),
                quality=inputs.quality.get(s),
                missing_reason=inputs.missing_quotes.get(s),
                market_open=market_open,
                closed=closed,
                price_status=inputs.price_status,
                fresh_seconds=self._s.brain_fresh_quote_seconds,
                max_age_seconds=self._s.trading_max_quote_age_seconds,
                refused_feed=refused,
            )
            for s in dict.fromkeys([*inputs.universe, *focus])
        }
        states = {s: d.state for s, d in diagnoses.items()}
        feed = feed_report(
            diagnoses,
            market_open=market_open,
            closed=closed,
            feeds=feeds,
            skew_seconds=skew,
            max_age_seconds=self._s.trading_max_quote_age_seconds,
        )
        session = brain_session(now)
        if session is BrainSession.OPEN and not market_open:
            session = BrainSession.HOLIDAY  # Alpaca says closed on a calendar trading day
        stocks = [c for c in inputs.close.columns if c not in self._s.trading_etfs]
        ctx = BrainContext(
            as_of=now,
            session=session,
            market_open=market_open,
            clock_source=clock_source,
            mode=mode,
            universe=list(inputs.universe),
            close=inputs.close,
            high=inputs.high,
            low=inputs.low,
            volume=inputs.volume,
            benchmark=inputs.benchmark,
            benchmark_symbol=self._s.benchmark_symbol,
            qqq=inputs.qqq,
            price_status=inputs.price_status,
            quotes=dict(inputs.quotes),
            quality=dict(inputs.quality),
            missing_quotes=dict(inputs.missing_quotes),
            indicators=indicators,
            market_stats=market_statistics(
                inputs.close[stocks] if stocks else inputs.close, inputs.benchmark
            ),
            regime=regime,
            vix=inputs.vix,
            implied_vol=dict(inputs.implied_vol),
            earnings=dict(inputs.earnings),
            model_z=inputs.model_z,
            fundamentals=(
                model.features[[c for c in FUNDAMENTAL_FEATURES if c in model.features.columns]]
                if model is not None and not model.features.empty
                else None
            ),
            sectors=sectors,
            portfolio=portfolio,
            account=account,
            data_states=states,
            limits=RiskLimits.from_settings(self._s),
            kill_switch=kill.active,
            trading_blockers=blockers,
            data_health=diagnoses,
            feed=feed,
            model=model,
            options=options,
            events=events,
            opportunities=opportunities,
            focus=focus,
            focus_reasons=reasons,
            notes=list(inputs.notes) + ([inputs.model_note] if inputs.model_note else []) + self._notes,
            provider_errors=errors,
        )
        if inputs.price_status is DataStatus.SYNTHETIC:
            ctx.notes.append("prices are SYNTHETIC (no live data): nothing is executable")
        return ctx
