"""Market inputs for one paper-trading cycle, gathered from QuantPulse's existing data services.

* **Universe** — ``QP_TRADING_UNIVERSE=auto``: the stock model's universe (today's S&P 500 members when
  Alpaca data is configured, the picks list otherwise) plus the liquid ETFs in ``QP_TRADING_ETFS``,
  narrowed to the ``QP_TRADING_UNIVERSE_SIZE`` names with the highest 20-day dollar volume. Current
  holdings are always included.
* **Prices** — daily bars from the warehouse-first universe panel (only missing sessions are downloaded),
  plus today's live snapshot (price, volume, VWAP, bid/ask) for every candidate.
* **Quote quality** — every quote is checked before its spread is believed (:func:`assess_quote`): a
  one-sided, crossed, stale or off-market bid/ask is not a spread. Alpaca's free IEX feed is a single
  exchange whose book can be far wider than the market, so the consolidated (SIP) quote is used for the
  spread when the subscription allows it (possibly 15 minutes delayed). A price far from the last close
  or history that disagrees with the vendor's previous close blocks new entries (bad tick, split or a
  mis-mapped symbol). Nothing here loosens a limit: an unmeasurable spread fails the liquidity check.
* **Stock model** — the latest walk-forward run's live z-scores, and its point-in-time fundamentals and
  earnings-reaction features (never waits long: the previous run is used while a new one trains).
* **VIX** (when a live source has it), **implied volatility** for the leading candidates, and **earnings
  dates** for the leading candidates.

Every piece records where it came from; nothing synthetic is ever passed off as live.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta

import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.market_calendar import NEW_YORK, Session, is_trading_day, regular_close, session_at
from quantpulse.domain.trading_signals import LiveBar
from quantpulse.schemas.common import DataStatus
from quantpulse.schemas.market import Bar, Quote
from quantpulse.schemas.options import OptionChain
from quantpulse.services.market import MarketService
from quantpulse.services.model import ModelService, ModelSnapshot
from quantpulse.services.options import OptionsService
from quantpulse.services.reference import ReferenceService

logger = logging.getLogger(__name__)

HISTORY_DAYS = 420  # ~290 sessions: 12-1 momentum, 200-day averages and a year of beta
LIQUIDITY_WINDOW = 20
MIN_BARS = 64
IV_TIMEOUT = 20.0
EARNINGS_TIMEOUT = 20.0
VIX_SYMBOL = "^VIX"
DELAYED_SIP_SECONDS = 15 * 60  # Alpaca's delayed SIP feed lags by 15 minutes
OFF_MARKET_PCT = (
    0.03  # a real-time bid/ask midpoint this far from the last trade does not describe the market
)
OFF_MARKET_DELAYED_PCT = 0.10  # the same for a 15-minute-old consolidated quote
PRICE_JUMP_PCT = 0.25  # live price this far from the last close: bad tick, split or wrong symbol
HISTORY_MISMATCH_PCT = 0.15  # stored close vs the vendor's previous close: split or mis-mapped history
FEED_LABELS = {"iex": "IEX", "sip": "SIP", "delayed_sip": "SIP (15-min delayed)"}
Progress = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    return None


def spread_of(bid: float | None, ask: float | None) -> float | None:
    """Quoted spread in basis points of the midpoint (``None`` when one-sided or crossed)."""
    if bid and ask and ask >= bid > 0:
        return (ask - bid) / (0.5 * (ask + bid)) * 10_000
    return None


@dataclass(frozen=True, slots=True)
class LiveQuote:
    symbol: str
    price: float
    bid: float | None
    ask: float | None
    vwap: float | None
    volume: float | None
    day_high: float | None
    day_low: float | None
    day_open: float | None
    timestamp: datetime  # of the last trade (the price)
    provider: str
    age_seconds: float  # of the last trade
    quote_time: datetime | None = None  # of the bid/ask
    feed: str | None = None  # vendor feed: "iex" is one exchange, "sip" all of them
    previous_close: float | None = None  # the vendor's previous session close
    as_of: datetime | None = None
    # consolidated (all-exchange) quote, when the vendor offers one (possibly 15 minutes delayed)
    nbbo_bid: float | None = None
    nbbo_ask: float | None = None
    nbbo_time: datetime | None = None
    nbbo_feed: str | None = None
    history_close: float | None = None  # the last completed close in QuantPulse's price history

    @property
    def venue(self) -> str:
        return FEED_LABELS.get(self.feed or "", self.provider)

    @property
    def quote_age_seconds(self) -> float | None:
        if self.quote_time is None or self.as_of is None:
            return None
        return max((self.as_of - self.quote_time).total_seconds(), 0.0)

    @property
    def nbbo_age_seconds(self) -> float | None:
        if self.nbbo_time is None or self.as_of is None:
            return None
        return max((self.as_of - self.nbbo_time).total_seconds(), 0.0)

    @property
    def venue_spread_bps(self) -> float | None:
        return spread_of(self.bid, self.ask)

    @property
    def nbbo_spread_bps(self) -> float | None:
        return spread_of(self.nbbo_bid, self.nbbo_ask)

    @property
    def spread_bps(self) -> float | None:
        """The spread as quoted by the primary feed (unvalidated; see :func:`assess_quote`)."""
        return self.venue_spread_bps

    def bar(self) -> LiveBar:
        return LiveBar(
            price=self.price,
            volume=self.volume,
            vwap=self.vwap,
            high=self.day_high,
            low=self.day_low,
            open=self.day_open,
        )


@dataclass(frozen=True, slots=True)
class QuoteQuality:
    """What a quote can be trusted for."""

    spread_bps: float | None  # the spread risk checks use (None: it cannot be measured reliably)
    spread_source: str  # "SIP", "SIP (15-min delayed)", "IEX only", … or "unavailable"
    problems: tuple[str, ...] = ()  # parts of the quote that were not believed, and why
    entry_blocks: tuple[str, ...] = ()  # reasons no new position may be opened or added (exits still go)

    @property
    def usable_bid_ask(self) -> bool:
        return not any(p.startswith("primary:") for p in self.problems)


def assess_quote(q: LiveQuote, max_age_seconds: float) -> QuoteQuality:
    """Validate a quote before its spread is used: one-sided, crossed, stale or off-market bid/asks are
    discarded (never read as a spread), the consolidated quote is preferred over a single exchange, and
    prices inconsistent with the price history block new entries."""
    problems: list[str] = []
    blocks: list[str] = []

    venue: float | None = None
    label = q.venue
    if not q.bid or not q.ask:
        missing = "bid" if not q.bid else "ask"
        problems.append(f"primary: {label} quote is one-sided (no {missing})")
    elif q.bid > q.ask:
        problems.append(f"primary: {label} quote is crossed (bid ${q.bid:,.2f} > ask ${q.ask:,.2f})")
    elif q.quote_age_seconds is not None and q.quote_age_seconds > max_age_seconds:
        problems.append(
            f"primary: {label} bid/ask is {q.quote_age_seconds:,.0f}s old (limit {max_age_seconds:,.0f}s)"
        )
    elif abs(0.5 * (q.bid + q.ask) / q.price - 1) > OFF_MARKET_PCT:
        off = 0.5 * (q.bid + q.ask) / q.price - 1
        problems.append(
            f"primary: {label} bid/ask midpoint is {off:+.1%} from the last trade ${q.price:,.2f}: not the market"
        )
    else:
        venue = q.venue_spread_bps

    nbbo: float | None = None
    if q.nbbo_feed:
        nlabel = FEED_LABELS.get(q.nbbo_feed, q.nbbo_feed)
        delayed = q.nbbo_feed == "delayed_sip"
        limit = max_age_seconds + (DELAYED_SIP_SECONDS if delayed else 0)
        age = q.nbbo_age_seconds
        spread = q.nbbo_spread_bps
        if spread is None:
            problems.append(f"consolidated: {nlabel} quote is one-sided or crossed")
        elif age is not None and age > limit:
            problems.append(f"consolidated: {nlabel} quote is {age:,.0f}s old (limit {limit:,.0f}s)")
        elif abs(0.5 * ((q.nbbo_bid or 0) + (q.nbbo_ask or 0)) / q.price - 1) > (
            OFF_MARKET_DELAYED_PCT if delayed else OFF_MARKET_PCT
        ):
            problems.append(f"consolidated: {nlabel} midpoint is far from the last trade ${q.price:,.2f}")
        else:
            nbbo = spread

    if nbbo is not None:
        spread_bps, source = nbbo, FEED_LABELS.get(q.nbbo_feed or "", "consolidated")
    elif venue is not None:
        spread_bps = venue
        source = f"{label} only" if q.feed in (None, "iex") else label
    else:
        spread_bps, source = None, "unavailable"

    ref = q.history_close or q.previous_close
    if ref and ref > 0 and abs(q.price / ref - 1) > PRICE_JUMP_PCT:
        blocks.append(
            f"price ${q.price:,.2f} is {q.price / ref - 1:+.0%} from the last close ${ref:,.2f}: "
            "possible bad tick, split or wrong symbol"
        )
    if (
        q.history_close
        and q.previous_close
        and abs(q.previous_close / q.history_close - 1) > HISTORY_MISMATCH_PCT
    ):
        blocks.append(
            f"stored history (last close ${q.history_close:,.2f}) disagrees with {q.provider}'s previous "
            f"close ${q.previous_close:,.2f}: possible split or mis-mapped symbol"
        )
    return QuoteQuality(spread_bps, source, tuple(problems), tuple(blocks))


@dataclass
class TradingInputs:
    as_of: datetime
    session_open: bool
    session_fraction: float | None
    universe: list[str]
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    qqq: pd.Series | None
    price_status: DataStatus
    quotes: dict[str, LiveQuote]
    missing_quotes: dict[str, str]
    quality: dict[str, QuoteQuality] = field(default_factory=dict)
    model: ModelSnapshot | None = None
    model_note: str | None = None
    vix: float | None = None
    implied_vol: dict[str, float] = field(default_factory=dict)
    earnings: dict[str, tuple[date, str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def model_z(self) -> dict[str, float]:
        if self.model is None:
            return {}
        return {s: float(v.z) for s, v in self.model.live.items() if math.isfinite(v.z)}

    def fundamentals(self, names: Sequence[str]) -> pd.DataFrame | None:
        if self.model is None or self.model.features.empty:
            return None
        cols = [n for n in names if n in self.model.features.columns]
        return self.model.features[cols] if cols else None


def session_fraction(moment: datetime) -> float | None:
    """Share of today's regular session already traded (``None`` outside the session)."""
    if session_at(moment) is not Session.REGULAR:
        return None
    local = moment.astimezone(NEW_YORK)
    start = datetime.combine(local.date(), time(9, 30), NEW_YORK)
    end = datetime.combine(local.date(), regular_close(local.date()), NEW_YORK)
    return min(max((local - start) / (end - start), 0.0), 1.0)


def atm_implied_vol(chain: OptionChain, spot: float, today: date, target_days: int = 30) -> float | None:
    """At-the-money implied volatility of the listed expiry nearest ``target_days`` (at least a week out)."""
    expiries = sorted({c.expiration for c in chain.contracts if (c.expiration - today).days >= 7})
    if not expiries or spot <= 0:
        return None
    expiry = min(expiries, key=lambda e: abs((e - today).days - target_days))
    near = [c for c in chain.contracts if c.expiration == expiry and c.implied_volatility]
    if not near:
        return None
    strike = min({c.strike for c in near}, key=lambda k: abs(k - spot))
    ivs = [float(c.implied_volatility or 0) for c in near if c.strike == strike]
    ivs = [v for v in ivs if 0.01 < v < 5]
    return sum(ivs) / len(ivs) if ivs else None


def _wide(frames: dict[str, pd.DataFrame], column: str, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({s: f[column].reindex(index) for s, f in frames.items()}, index=index)


class TradingDataLoader:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        market: MarketService,
        model: ModelService,
        options: OptionsService,
        reference: ReferenceService,
    ) -> None:
        self._s = settings
        self._clock = clock
        self._market = market
        self._model = model
        self._options = options
        self._reference = reference

    async def candidates(self) -> list[str]:
        """Symbols considered before the liquidity cut."""
        s = self._s
        if s.trading_universe != "auto":
            base = s.trading_universe.split(",")
        else:
            spec = await self._model.universe(None)
            base = list(spec.membership.current) if spec.membership is not None else list(spec.symbols)
        return list(dict.fromkeys([*s.trading_etfs, *base]))

    async def load(self, held: Sequence[str], progress: Progress = _noop) -> TradingInputs:
        s = self._s
        now = self._clock.now()
        bench = s.benchmark_symbol
        progress(0.02, "building the trading universe")
        pool = list(dict.fromkeys([*await self.candidates(), *held, bench, "QQQ"]))
        progress(0.05, f"loading daily prices for {len(pool)} symbols")
        panel = await self._market.daily_panel(
            pool, HISTORY_DAYS, progress=lambda f, st: progress(0.05 + 0.45 * f, st)
        )
        if bench not in panel.frames:
            raise DomainError(f"no daily prices for the benchmark {bench}: cannot rank the universe")
        today = pd.Timestamp(now.astimezone(NEW_YORK).date())
        index = panel.frames[bench].index
        index = index[index < today]  # completed sessions only; today's comes from the live snapshot
        frames = {
            sym: f for sym, f in panel.frames.items() if f["close"].reindex(index).notna().sum() >= MIN_BARS
        }
        skipped = dict(panel.missing)
        for sym in panel.frames:
            if sym not in frames:
                skipped[sym] = f"fewer than {MIN_BARS} daily bars"
        close = _wide(frames, "close", index)
        volume = _wide(frames, "volume", index)
        adv = (close * volume).iloc[-LIQUIDITY_WINDOW:].mean().sort_values(ascending=False)
        keep = [sym for sym in adv.index[: s.trading_universe_size] if sym in frames]
        universe = list(dict.fromkeys([*keep, *[h for h in held if h in frames]]))
        for sym in adv.index[s.trading_universe_size :]:
            if sym not in universe:
                skipped.setdefault(sym, "outside the most liquid names this cycle")
        price_status = (
            DataStatus.worst([panel.status(sym) for sym in universe]) if universe else DataStatus.SYNTHETIC
        )

        progress(0.55, f"fetching live quotes for {len(universe)} symbols")
        history_close = {
            sym: float(f["close"].reindex(index).dropna().iloc[-1])
            for sym, f in frames.items()
            if f["close"].reindex(index).notna().any()
        }
        quotes, missing_quotes = await self._live(
            list(dict.fromkeys([*universe, bench, "QQQ"])), consolidated=True, history_close=history_close
        )
        is_open = session_at(now) is Session.REGULAR and is_trading_day(now.astimezone(NEW_YORK).date())
        inputs = TradingInputs(
            as_of=now,
            session_open=is_open,
            session_fraction=session_fraction(now),
            universe=universe,
            close=close[universe] if universe else close,
            high=_wide(frames, "high", index)[universe] if universe else close,
            low=_wide(frames, "low", index)[universe] if universe else close,
            volume=volume[universe] if universe else volume,
            benchmark=frames[bench]["close"].reindex(index) if bench in frames else pd.Series(dtype=float),
            qqq=frames["QQQ"]["close"].reindex(index) if "QQQ" in frames else None,
            price_status=price_status,
            quotes=quotes,
            missing_quotes=missing_quotes,
            quality={sym: assess_quote(q, s.trading_max_quote_age_seconds) for sym, q in quotes.items()},
            skipped=skipped,
        )
        consolidated = sum(1 for q in quotes.values() if q.nbbo_feed)
        if quotes and any(q.feed == "iex" for q in quotes.values()):
            inputs.notes.append(
                "Live quotes are Alpaca IEX (one exchange): "
                + (
                    f"spreads measured on the consolidated SIP quote for {consolidated} symbol(s)"
                    if consolidated
                    else "no consolidated quote is available on this subscription, so spreads are IEX's own "
                    "book (often wider than the market): names with an unmeasurable or wide spread are not bought"
                )
            )
        inputs.notes.append(
            f"{len(universe)} of {len(pool)} symbols tradable after the liquidity cut; live quotes for "
            f"{len([u for u in universe if u in quotes])}"
        )

        progress(0.65, "reading the stock model")
        weights = s.trading_signal_weights
        if weights.get("model", 0) > 0 or weights.get("fundamental", 0) > 0:
            try:
                snap = await self._model.trading_snapshot(wait=s.trading_model_wait_seconds)
                real = price_status is not DataStatus.SYNTHETIC
                if real and snap.data_status is DataStatus.SYNTHETIC:
                    inputs.model_note = (
                        "the stock model only has synthetic prices: model and fundamentals skipped"
                    )
                else:
                    inputs.model = snap
                    inputs.model_note = f"stock model ({snap.label}) as of the {snap.as_of} close"
            except DomainError as exc:  # still training for the first time, or not enough history
                inputs.model_note = (
                    f"stock model unavailable this cycle ({exc}): its components count as neutral"
                )

        progress(0.72, "checking the VIX")
        inputs.vix = await self._vix()
        return inputs

    async def enrich(self, inputs: TradingInputs, leaders: Sequence[str]) -> None:
        """Implied volatility and earnings dates for the leading candidates (bounded in time)."""
        s = self._s
        today = self._clock.now().astimezone(NEW_YORK).date()

        async def iv(sym: str) -> None:
            q = inputs.quotes.get(sym)
            if q is None:
                return
            chain_r, _ = await self._options.chain(sym, max_expirations=4)
            if chain_r.status not in (DataStatus.LIVE, DataStatus.CACHED):
                return
            value = atm_implied_vol(chain_r.value, q.price, today)
            if value is not None:
                inputs.implied_vol[sym] = value

        async def earnings(sym: str) -> None:
            nxt, source = await self._reference.next_earnings(sym)
            if nxt is not None and source is not None and nxt >= today:
                inputs.earnings[sym] = (nxt, source)

        async def bounded(
            jobs: list[asyncio.Future[None] | asyncio.Task[None]], what: str, timeout: float
        ) -> None:
            done, pending = await asyncio.wait(jobs, timeout=timeout) if jobs else (set(), set())
            for task in pending:
                task.cancel()
            failed = sum(1 for t in done if t.exception() is not None)
            if pending or failed:
                inputs.notes.append(f"{what}: {len(pending)} timed out, {failed} failed (used what arrived)")

        stocks = [sym for sym in leaders if sym not in s.trading_etfs]
        if s.trading_use_implied_vol and s.enable_live_data:
            await bounded(
                [asyncio.ensure_future(iv(sym)) for sym in leaders], "implied volatility", IV_TIMEOUT
            )
        if s.trading_earnings_blackout_days > 0:
            await bounded(
                [asyncio.ensure_future(earnings(sym)) for sym in stocks], "earnings dates", EARNINGS_TIMEOUT
            )

    async def live_quotes(
        self,
        symbols: Sequence[str],
        *,
        consolidated: bool = True,
        history_close: dict[str, float] | None = None,
    ) -> dict[str, LiveQuote]:
        """Live quotes for ``symbols``, with the consolidated (SIP) bid/ask attached where the primary feed
        is a single exchange and a vendor offers it."""
        quotes, _ = await self._live(symbols, consolidated=consolidated, history_close=history_close)
        return quotes

    async def _live(
        self, symbols: Sequence[str], *, consolidated: bool, history_close: dict[str, float] | None
    ) -> tuple[dict[str, LiveQuote], dict[str, str]]:
        if not symbols:
            return {}, {}
        got = await self._market.live_quotes(symbols)
        now = self._clock.now()
        out = {
            sym: _live_quote(sym, q, got.providers.get(sym, "?"), now, (history_close or {}).get(sym))
            for sym, q in got.quotes.items()
        }
        need = [sym for sym, q in out.items() if q.feed != "sip"] if consolidated else []
        if need:
            nbbo = await self._market.consolidated_quotes(need)
            for sym, c in nbbo.items():
                if sym in out:
                    out[sym] = replace(
                        out[sym], nbbo_bid=c.bid, nbbo_ask=c.ask, nbbo_time=c.timestamp, nbbo_feed=c.feed
                    )
        return out, dict(got.missing)

    async def daily_history(self, symbol: str, days: int = 15) -> list[Bar]:
        """Recent daily bars for one symbol (empty when only synthetic prices exist)."""
        r = await self._market.history(symbol, "1d", days)
        if r.status is DataStatus.SYNTHETIC:
            return []
        return list(r.value.bars)

    async def _vix(self) -> float | None:
        if not self._s.enable_live_data:
            return None
        try:
            r = await self._market.history(VIX_SYMBOL, "1d", 30)
        except Exception as exc:  # the VIX is optional
            logger.info("VIX unavailable: %s", exc)
            return None
        if r.status is DataStatus.SYNTHETIC or not r.value.bars:
            return None
        last = r.value.bars[-1]
        if self._clock.now() - last.timestamp > timedelta(days=5):
            return None
        return float(last.close)


def _live_quote(
    symbol: str, q: Quote, provider: str, now: datetime, history_close: float | None = None
) -> LiveQuote:
    return LiveQuote(
        symbol=symbol,
        price=q.price,
        bid=q.bid,
        ask=q.ask,
        vwap=q.vwap,
        volume=q.volume,
        day_high=q.day_high,
        day_low=q.day_low,
        day_open=q.day_open,
        timestamp=q.timestamp,
        provider=provider,
        age_seconds=max((now - q.timestamp).total_seconds(), 0.0),
        quote_time=q.quote_timestamp,
        feed=q.feed,
        previous_close=q.previous_close,
        as_of=now,
        history_close=history_close,
    )
