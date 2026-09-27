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

import logging
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, is_market_open
from quantpulse.domain import trading_regime as regime_mod
from quantpulse.domain import trading_signals as ts
from quantpulse.providers.alpaca_trading import BrokerError
from quantpulse.schemas.common import DataStatus
from quantpulse.services.reference import ReferenceService
from quantpulse.services.trading import TradingService
from quantpulse.services.trading_data import LiveQuote, QuoteQuality, TradingDataLoader, TradingInputs
from quantpulse.services.trading_risk import RiskLimits

from .context import BrainContext, BrokerView, PortfolioState, brain_session
from .indicators import compute_indicators, market_statistics
from .types import BrainMode, BrainSession, DataState

logger = logging.getLogger(__name__)
FOCUS_SCREEN = ("mom_3m", "rel_strength", "persistence", "px_vs_sma50", "mom_accel")


def data_state(
    symbol: str,
    *,
    quote: LiveQuote | None,
    quality: QuoteQuality | None,
    missing_reason: str | None,
    market_open: bool,
    price_status: DataStatus,
    fresh_seconds: float,
    max_age_seconds: float,
) -> DataState:
    """How far a symbol's market data can be trusted right now (see :class:`DataState`)."""
    if price_status is DataStatus.SYNTHETIC:
        return DataState.UNAVAILABLE  # simulated prices are never data
    if not market_open:
        return DataState.MARKET_CLOSED
    if quote is None:
        reason = (missing_reason or "").lower()
        return DataState.PROVIDER_ERROR if ("fail" in reason or "error" in reason) else DataState.UNAVAILABLE
    age = quote.age_seconds
    if quote.quote_age_seconds is not None:
        age = max(age, quote.quote_age_seconds) if quality is not None and quality.usable_bid_ask else age
    if age <= fresh_seconds:
        return DataState.FRESH
    if age <= max_age_seconds:
        return DataState.LIVE
    return DataState.STALE


def choose_focus(
    indicators: pd.DataFrame,
    held: Sequence[str],
    requested: Sequence[str],
    quotes: dict[str, LiveQuote],
    market_open: bool,
    n: int,
    exclude: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Which symbols to study closely this cycle: requested ones, every holding, and the ``n`` strongest
    of a cheap cross-sectional pre-screen (momentum, relative strength, persistence, trend, acceleration).
    The rest of the universe is not analysed symbol by symbol (cost control)."""
    reasons: dict[str, str] = {}
    for s in requested:
        reasons.setdefault(s, "requested")
    for s in held:
        reasons.setdefault(s, "held position")
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
    ) -> None:
        self._s = settings
        self._clock = clock
        self._data = data
        self._broker = broker
        self._trading = trading
        self._reference = reference

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

    async def _market_open(self, now: datetime, errors: dict[str, str]) -> tuple[bool, str]:
        if self._broker.configured():
            try:
                return (await self._broker.clock()).is_open, "alpaca"
            except BrokerError as exc:
                errors["clock"] = str(exc)
        return is_market_open(now), "calendar"

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

    async def perceive(self, mode: BrainMode, requested: Sequence[str] = ()) -> BrainContext:
        now = self._clock.now()
        errors: dict[str, str] = {}
        portfolio = await self._portfolio(errors)
        market_open, clock_source = await self._market_open(now, errors)
        held = [s for s, p in portfolio.positions.items() if p.qty > 0]
        requested = [s.strip().upper() for s in requested if s.strip()]
        inputs = await self._data.load(list(dict.fromkeys([*held, *requested])))

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
        focus, reasons = choose_focus(
            indicators,
            held,
            requested,
            inputs.quotes,
            market_open,
            self._s.brain_focus_candidates,
            exclude=[self._s.benchmark_symbol, "QQQ"],
        )
        await self._data.enrich(inputs, focus)  # implied vol and earnings dates for the focus set only
        if inputs.implied_vol:
            iv = pd.Series(inputs.implied_vol, dtype=float)
            indicators["implied_vol"] = iv.reindex(indicators.index)
            rv = pd.to_numeric(indicators.get("rv63"), errors="coerce")
            indicators["iv_premium"] = indicators["implied_vol"] / rv.where(rv > 0) - 1

        kill = await self._trading.kill_switch()
        states = {
            s: data_state(
                s,
                quote=inputs.quotes.get(s),
                quality=inputs.quality.get(s),
                missing_reason=inputs.missing_quotes.get(s),
                market_open=market_open,
                price_status=inputs.price_status,
                fresh_seconds=self._s.brain_fresh_quote_seconds,
                max_age_seconds=self._s.trading_max_quote_age_seconds,
            )
            for s in dict.fromkeys([*inputs.universe, *focus])
        }
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
            regime=self._regime(inputs),
            vix=inputs.vix,
            implied_vol=dict(inputs.implied_vol),
            earnings=dict(inputs.earnings),
            model_z=inputs.model_z,
            fundamentals=inputs.fundamentals(["earnings_yield", "fcf_yield", "book_to_market", "roe"]),
            sectors=await self._sectors(errors),
            portfolio=portfolio,
            data_states=states,
            limits=RiskLimits.from_settings(self._s),
            kill_switch=kill.active,
            focus=focus,
            focus_reasons=reasons,
            notes=list(inputs.notes) + ([inputs.model_note] if inputs.model_note else []),
            provider_errors=errors,
        )
        if inputs.price_status is DataStatus.SYNTHETIC:
            ctx.notes.append("prices are SYNTHETIC (no live data): nothing is executable")
        return ctx
