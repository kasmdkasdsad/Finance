"""Market data service: quotes, OHLCV history and dividend yields through the data gateway.

Besides per-symbol requests it maintains **daily price panels** for whole universes (the S&P 500 and its
former members): bars live in the warehouse, and a panel request only downloads what is missing — the
full window for new symbols, the last few sessions for the rest — through a multi-symbol endpoint when
the vendor has one. A restart or a second model run therefore costs one database read, not 600 requests.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.core.market_calendar import (
    NEW_YORK,
    Session,
    is_trading_day,
    previous_trading_day,
    regular_close,
    session_at,
)
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.providers import synthetic
from quantpulse.providers.yahoo import YahooFinance
from quantpulse.schemas.common import DataStatus, Provenance
from quantpulse.schemas.market import INTRADAY_INTERVALS, Interval, PriceHistory, Quote

logger = logging.getLogger(__name__)

# One daily-history window shared by picks, forecasts, the stock model and reports, so each symbol is
# fetched (and cached) once instead of once per feature.
STANDARD_HISTORY_DAYS = 1825

COVERAGE_KEY = "bars-coverage:1d"
TAIL_OVERLAP = timedelta(days=10)  # re-read this much before the last stored bar (vendor revisions)
SETTLE = timedelta(minutes=20)  # daily bars are final this long after the close
RETRY_MISSING = timedelta(days=7)  # how often a symbol the vendors did not know is tried again
PANEL_CONCURRENCY = 4
BATCH_CHUNK = 50
REBASE_TOLERANCE = 1e-6  # relative close difference that means the vendor re-adjusted the history
FINISHED_AFTER = timedelta(days=20)  # no bars this long before the freshest symbol: the listing ended
PERSIST_CHUNK = 25


class MarketProvider(Protocol):
    name: str

    def configured(self) -> bool: ...

    async def quote(self, symbol: str) -> Quote: ...

    async def history(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> PriceHistory: ...


class BatchHistoryProvider(Protocol):
    name: str

    def configured(self) -> bool: ...

    async def histories(
        self, symbols: Sequence[str], interval: Interval, start: datetime, end: datetime
    ) -> dict[str, PriceHistory]: ...


class BatchQuoteProvider(Protocol):
    name: str

    def configured(self) -> bool: ...

    async def quotes(self, symbols: Sequence[str]) -> dict[str, Quote]: ...


QUOTE_BATCH = 100  # symbols per multi-symbol snapshot request
QUOTE_FALLBACKS = 30  # per-symbol lookups allowed for names the batch source did not return


@dataclass
class LiveQuotes:
    """Quotes fetched from a live vendor just now — never cached, archived or synthetic values."""

    quotes: dict[str, Quote]
    providers: dict[str, str]
    fetched_at: datetime
    missing: dict[str, str] = field(default_factory=dict)


def history_frame(h: PriceHistory) -> pd.DataFrame:
    """OHLCV indexed by New York session date (naive midnight timestamps)."""
    idx = pd.DatetimeIndex([b.timestamp for b in h.bars])
    return _ohlcv_frame(
        idx,
        [b.open for b in h.bars],
        [b.high for b in h.bars],
        [b.low for b in h.bars],
        [b.close for b in h.bars],
        [b.volume for b in h.bars],
    )


def _ohlcv_frame(
    idx: pd.DatetimeIndex, o: object, h: object, lo: object, c: object, v: object
) -> pd.DataFrame:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    days = idx.tz_convert("America/New_York").normalize().tz_localize(None)
    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=days, dtype=float)
    return df[~df.index.duplicated(keep="last")].sort_index()


@dataclass
class DailyPanel:
    """Daily OHLCV for a universe, one frame per symbol, with the provenance of each symbol's prices."""

    frames: dict[str, pd.DataFrame]
    provenance: dict[str, Provenance]
    missing: dict[str, str] = field(default_factory=dict)  # symbol -> why it has no prices

    def status(self, symbol: str) -> DataStatus:
        p = self.provenance.get(symbol)
        return p.status if p else DataStatus.SYNTHETIC

    def summary(self) -> dict[str, Provenance]:
        """One provenance entry per (status, provider) group, for composite responses."""
        groups: dict[tuple[DataStatus, str], list[Provenance]] = {}
        for p in self.provenance.values():
            groups.setdefault((p.status, p.provider), []).append(p)
        out: dict[str, Provenance] = {}
        for (status, provider), items in sorted(groups.items(), key=lambda kv: (kv[0][0].rank, kv[0][1])):
            latest = max(items, key=lambda p: p.as_of)
            out[f"prices:{status.value}:{provider}"] = latest.model_copy(
                update={
                    "message": f"{len(items)} symbols" + (f" — {latest.message}" if latest.message else "")
                }
            )
        return out


class MarketService:
    def __init__(
        self,
        settings: Settings,
        gateway: DataGateway,
        db: Database,
        clock: Clock,
        providers: Sequence[MarketProvider],
        yahoo: YahooFinance,
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._providers = list(providers)
        self._yahoo = yahoo
        self._history_sem = asyncio.Semaphore(6)

    @property
    def providers(self) -> list[MarketProvider]:
        return self._providers

    def _sources(self, fetch_name: str, *args: object) -> list[Source]:
        sources: list[Source] = []
        for p in self._providers:
            fetch = functools.partial(getattr(p, fetch_name), *args)
            sources.append(Source(p.name, fetch, configured=p.configured()))
        return sources

    async def quote(self, symbol: str, *, force_refresh: bool = False) -> Resolved[Quote]:
        async def persist(q: Quote, provider: str) -> None:
            async with self._db.session() as s:
                await repo.insert_quote(s, q, provider)

        async def archive() -> tuple[Quote, datetime, str] | None:
            async with self._db.session() as s:
                return await repo.latest_quote(s, symbol)

        return await self._gw.resolve(
            f"quote:{symbol}",
            self._sources("quote", symbol),
            lambda: synthetic.synthetic_quote(symbol, self._clock.now()),
            self._settings.ttl_quote,
            as_of=lambda q: q.timestamp,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
        )

    async def quotes(
        self, symbols: Sequence[str], *, force_refresh: bool = False
    ) -> dict[str, Resolved[Quote]]:
        results = await asyncio.gather(*(self.quote(s, force_refresh=force_refresh) for s in symbols))
        return dict(zip(symbols, results, strict=True))

    async def live_quotes(self, symbols: Sequence[str]) -> LiveQuotes:
        """Fresh quotes for many symbols, for trading: multi-symbol snapshots from a vendor that offers them
        (Alpaca), then per-symbol lookups for the few it missed. Only answers a live vendor gave *now* are
        returned — a symbol with nothing better than a cached, stale or synthetic price is listed in
        ``missing`` instead."""
        symbols = list(dict.fromkeys(symbols))
        now = self._clock.now()
        out = LiveQuotes({}, {}, now)
        if not self._settings.enable_live_data:
            out.missing = dict.fromkeys(symbols, "live data disabled (QP_ENABLE_LIVE_DATA=false)")
            return out
        todo = list(symbols)
        for p in self._providers:
            if not todo or not (hasattr(p, "quotes") and p.configured()):
                continue
            batcher: BatchQuoteProvider = p  # type: ignore[assignment]
            for i in range(0, len(todo), QUOTE_BATCH):
                chunk = todo[i : i + QUOTE_BATCH]
                try:
                    got = await batcher.quotes(chunk)
                except Exception as exc:  # a failed batch falls through to the next source
                    logger.warning("batch quotes from %s failed: %s", p.name, exc)
                    continue
                for s, q in got.items():
                    if s in chunk:
                        out.quotes[s], out.providers[s] = q, p.name
            todo = [s for s in todo if s not in out.quotes]

        async def one(symbol: str) -> None:
            r = await self.quote(symbol, force_refresh=True)
            if r.status is DataStatus.LIVE:
                out.quotes[symbol], out.providers[symbol] = r.value, r.provenance.provider
            else:
                out.missing[symbol] = f"no live quote ({r.status.value} from {r.provenance.provider})"

        sem = asyncio.Semaphore(PANEL_CONCURRENCY)

        async def limited(symbol: str) -> None:
            async with sem:
                await one(symbol)

        await asyncio.gather(*(limited(s) for s in todo[:QUOTE_FALLBACKS]))
        for s in todo[QUOTE_FALLBACKS:]:
            out.missing[s] = "not returned by the batch quote source"
        return out

    async def consolidated_quotes(self, symbols: Sequence[str]) -> dict[str, Any]:
        """All-exchange (SIP) bid/ask for ``symbols`` from a vendor that offers it (Alpaca), possibly
        15 minutes delayed; empty when no configured vendor can provide it. Never raises."""
        if not symbols or not self._settings.enable_live_data:
            return {}
        for p in self._providers:
            fetch = getattr(p, "consolidated_quotes", None)
            if fetch is None or not p.configured():
                continue
            try:
                return dict(await fetch(list(symbols)))
            except Exception as exc:  # optional data: the single-venue quote is used instead
                logger.warning("consolidated quotes from %s failed: %s", p.name, exc)
        return {}

    async def history(
        self, symbol: str, interval: Interval = "1d", lookback_days: int = 365, *, force_refresh: bool = False
    ) -> Resolved[PriceHistory]:
        end = self._clock.now()
        start = end - timedelta(days=lookback_days)
        intraday = interval in INTRADAY_INTERVALS
        ttl = self._settings.ttl_bars_intraday if intraday else self._settings.ttl_bars_daily

        async def persist(h: PriceHistory, provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.upsert_bars(s, h, provider)
                await repo.record_ingestion(s, "bars", f"{symbol}:{interval}", provider, rows)

        async def archive() -> tuple[PriceHistory, datetime, str] | None:
            async with self._db.session() as s:
                found = await repo.load_bars(s, symbol, interval, since=start)
            if found is None or len(found[0].bars) < 5:
                return None
            return found

        async with self._history_sem:
            return await self._gw.resolve(
                f"bars:{symbol}:{interval}:{lookback_days}",
                self._sources("history", symbol, interval, start, end),
                lambda: synthetic.synthetic_history(symbol, interval, start, end, self._clock.now()),
                ttl,
                as_of=lambda h: h.bars[-1].timestamp if h.bars else None,
                archive=archive,
                on_live=persist,
                force_refresh=force_refresh,
            )

    # ------------------------------------------------------------------ universe panels
    def _fresh_after(self, now: datetime) -> datetime:
        """A fetch made after this moment already has every final daily bar."""
        if session_at(now) is Session.REGULAR:
            return now - timedelta(seconds=self._settings.ttl_bars_daily)
        local = now.astimezone(NEW_YORK)
        day = local.date()
        if not (is_trading_day(day) and local.time() >= regular_close(day)):
            day = previous_trading_day(day)
        settled = datetime.combine(day, regular_close(day), NEW_YORK) + SETTLE
        if settled > now:  # just after the close: bars are still settling
            return now - timedelta(seconds=self._settings.ttl_bars_daily)
        return settled

    @property
    def has_bulk_history(self) -> bool:
        """Whether a configured vendor can download many symbols' histories per request."""
        return self._settings.enable_live_data and self._batch_provider() is not None

    def _batch_provider(self) -> BatchHistoryProvider | None:
        for p in self._providers:
            if hasattr(p, "histories") and p.configured():
                return p  # type: ignore[return-value]
        return None

    async def daily_panel(
        self,
        symbols: Sequence[str],
        lookback_days: int = STANDARD_HISTORY_DAYS,
        *,
        progress: Callable[[float, str], None] | None = None,
        synthetic_only: bool = False,
    ) -> DailyPanel:
        """Daily bars for many symbols: warehouse first, then only the missing pieces from the vendors.

        Synthetic prices are only produced when live data is disabled (or ``synthetic_only`` is asked
        for); with live data on, a symbol no vendor knows is reported in ``missing`` instead of being
        invented."""
        symbols = list(dict.fromkeys(symbols))
        now = self._clock.now()
        start = now - timedelta(days=lookback_days)

        def report(fraction: float, stage: str) -> None:
            if progress is not None:
                progress(min(max(fraction, 0.0), 1.0), stage)

        if synthetic_only or not self._settings.enable_live_data:
            reason = (
                "live prices unavailable for this universe"
                if self._settings.enable_live_data
                else "live data disabled (QP_ENABLE_LIVE_DATA=false)"
            )
            return await self._synthetic_panel(symbols, start, now, report, reason)

        report(0.0, "reading stored prices")
        async with self._db.session() as s:
            blob = await repo.get_blob(s, COVERAGE_KEY)
            rows = await repo.bar_frame(s, symbols, "1d", start - TAIL_OVERLAP)
        coverage: dict[str, dict[str, str]] = dict(blob[0]) if blob else {}
        last_stored: dict[str, datetime] = {}
        stored_provider: dict[str, str] = {}
        for r in rows:  # ordered by (symbol, ts)
            last_stored[r[0]] = r[1]
            stored_provider[r[0]] = r[7]
        newest = max(last_stored.values(), default=None)

        plan = _plan(symbols, coverage, last_stored, newest, start, now, self._fresh_after(now))
        fetched: dict[str, tuple[PriceHistory, str]] = {}
        failed: dict[str, str] = {}  # the download failed: nothing learned
        empty: dict[str, str] = {}  # every vendor answered, none had bars
        batch = self._batch_provider()
        single = [p for p in self._providers if batch is None or p.name != batch.name]
        progress_total = max(1, len(plan.full) + len(plan.tail))
        progress_done = 0

        def tick(n: int) -> None:
            nonlocal progress_done
            progress_done += n
            report(
                0.05 + 0.75 * min(1.0, progress_done / progress_total),
                f"downloading prices ({min(progress_done, progress_total)}/{progress_total})",
            )

        async def via_batch(group: list[str], since: datetime) -> None:
            assert batch is not None
            for i in range(0, len(group), BATCH_CHUNK):
                chunk = group[i : i + BATCH_CHUNK]
                call = await self._gw.try_live(
                    Source(batch.name, functools.partial(batch.histories, chunk, "1d", since, now))
                )
                got = call.value or {}
                for sym in chunk:
                    h = got.get(sym)
                    if h is not None and h.bars:
                        fetched[sym] = (h, batch.name)
                        failed.pop(sym, None)
                    elif call.value is not None or call.no_data:
                        empty[sym] = f"{batch.name}: no bars"
                    else:
                        failed[sym] = f"{batch.name}: {call.attempt.error}"
                tick(len(chunk))

        async def via_single(group: list[str], since_of: Callable[[str], datetime]) -> None:
            sem = asyncio.Semaphore(PANEL_CONCURRENCY)

            async def one(sym: str) -> None:
                answered = sym in empty  # a vendor already said it has no bars for this symbol
                reasons = [x for x in (failed.pop(sym, None), empty.pop(sym, None)) if x]
                async with sem:
                    for p in single:
                        call = await self._gw.try_live(
                            Source(
                                p.name,
                                functools.partial(p.history, sym, "1d", since_of(sym), now),
                                configured=p.configured(),
                            )
                        )
                        if call.value is not None and call.value.bars:
                            fetched[sym] = (call.value, p.name)
                            break
                        answered = answered or call.no_data
                        reasons.append(f"{p.name}: {call.attempt.error or 'no bars'}")
                if sym not in fetched:
                    (empty if answered else failed)[sym] = "; ".join(reasons) or "no vendor configured"
                tick(1)

            await asyncio.gather(*(one(s) for s in group))

        def full_since(_: str) -> datetime:
            return start

        def tail_since(sym: str) -> datetime:
            return last_stored[sym] - TAIL_OVERLAP

        if batch is not None:
            if plan.full:
                await via_batch(plan.full, start)
            if plan.tail:
                await via_batch(plan.tail, min(last_stored[s] for s in plan.tail) - TAIL_OVERLAP)
            # The per-symbol vendors cover what the bulk source could not. Tails the bulk source answered
            # with "no bars" belong to finished listings and are not retried one by one.
            leftovers = [s for s in plan.full if s not in fetched]
            if leftovers and single:
                await via_single(leftovers, full_since)
        elif single:
            await via_single(plan.full, full_since)
            await via_single(plan.tail, tail_since)

        # A split or dividend re-bases a vendor's whole adjusted history. A tail whose overlap with the
        # stored bars no longer matches is therefore downloaded again in full, so old and new bars
        # never mix two price bases.
        stored_close = {
            (r[0], r[1]): r[5]
            for r in rows
            if r[0] in last_stored and r[1] >= last_stored[r[0]] - TAIL_OVERLAP
        }
        rebased = [
            sym for sym in plan.tail if sym in fetched and _rebased(fetched[sym][0], sym, stored_close)
        ]
        if rebased:
            report(0.75, f"re-downloading {len(rebased)} re-adjusted histories")
            for sym in rebased:
                del fetched[sym]
            if batch is not None:
                await via_batch(rebased, start)
                missed = [s for s in rebased if s not in fetched]
                if missed and single:
                    await via_single(missed, full_since)
            elif single:
                await via_single(rebased, full_since)
            for sym in rebased:
                if sym not in fetched:  # keep serving the stored (old-basis) history, flagged stale
                    failed[sym] = failed.get(sym) or "history was re-adjusted but the full download failed"

        report(0.8, "storing prices")
        items = list(fetched.items())
        for i in range(0, len(items), PERSIST_CHUNK):
            async with self._db.session() as s:
                for sym, (h, provider) in items[i : i + PERSIST_CHUNK]:
                    n = await repo.upsert_bars(s, h, provider)
                    await repo.record_ingestion(s, "bars", f"{sym}:1d:panel", provider, n)
        stamp = now.isoformat()
        for sym in [*plan.full, *plan.tail]:
            if sym in failed:
                continue  # a failed download proves nothing: the same plan runs again next time
            prior = coverage.get(sym, {})
            requested = (
                start.date().isoformat() if sym in plan.full else prior.get("start", start.date().isoformat())
            )
            coverage[sym] = {"start": requested, "checked": stamp}
        async with self._db.session() as s:
            await repo.put_blob(s, COVERAGE_KEY, coverage, "panel")
            if fetched:
                rows = await repo.bar_frame(s, symbols, "1d", start - TAIL_OVERLAP)

        report(0.9, "assembling the panel")
        frames = await asyncio.to_thread(_frames_from_rows, rows, start)
        provenance: dict[str, Provenance] = {}
        missing: dict[str, str] = {}
        for sym in symbols:
            f = frames.get(sym)
            if f is None or f.empty:
                missing[sym] = failed.get(sym) or empty.get(sym) or "no price history in the window"
                frames.pop(sym, None)
                continue
            last = datetime.combine(f.index[-1].date(), regular_close(f.index[-1].date()), NEW_YORK)
            if sym in fetched:
                provenance[sym] = Provenance(
                    status=DataStatus.LIVE, provider=fetched[sym][1], as_of=last, fetched_at=now
                )
            elif sym in failed:
                provenance[sym] = Provenance(
                    status=DataStatus.STALE,
                    provider=f"warehouse:{stored_provider.get(sym, '?')}",
                    as_of=last,
                    fetched_at=now,
                    message=f"refresh failed, serving stored bars ({failed[sym]})",
                )
            else:
                checked = coverage.get(sym, {}).get("checked")
                finished = sym in empty or sym in plan.finished
                provenance[sym] = Provenance(
                    status=DataStatus.CACHED,
                    provider=stored_provider.get(sym, "warehouse"),
                    as_of=last,
                    fetched_at=datetime.fromisoformat(checked) if checked else now,
                    message=f"no bars since {last.date()} (delisted, acquired or renamed)"
                    if finished
                    else None,
                )
        report(1.0, "prices ready")
        return DailyPanel(frames=frames, provenance=provenance, missing=missing)

    async def _synthetic_panel(
        self,
        symbols: list[str],
        start: datetime,
        now: datetime,
        report: Callable[[float, str], None],
        reason: str,
    ) -> DailyPanel:
        report(0.0, "generating synthetic prices")

        def generate() -> dict[str, pd.DataFrame]:
            return {s: history_frame(synthetic.synthetic_history(s, "1d", start, now, now)) for s in symbols}

        frames = await asyncio.to_thread(generate)
        prov = Provenance(
            status=DataStatus.SYNTHETIC,
            provider="synthetic",
            as_of=now,
            fetched_at=now,
            message=reason,
        )
        report(1.0, "prices ready")
        return DailyPanel(frames=frames, provenance=dict.fromkeys(symbols, prov))

    async def dividend_yield(self, symbol: str) -> Resolved[float]:
        async def fetch() -> float:
            value = await self._yahoo.dividend_yield(symbol)
            return value or 0.0

        return await self._gw.resolve(
            f"divyield:{symbol}",
            [Source(self._yahoo.name, fetch)],
            lambda: synthetic.synthetic_quote(symbol, self._clock.now()).dividend_yield or 0.0,
            self._settings.ttl_fundamentals,
        )


@dataclass
class _Plan:
    full: list[str]  # download the whole window
    tail: list[str]  # download the last few sessions
    finished: set[str]  # stored history ended long ago: nothing new will come


def _plan(
    symbols: Sequence[str],
    coverage: dict[str, dict[str, str]],
    last_stored: dict[str, datetime],
    newest: datetime | None,
    start: datetime,
    now: datetime,
    fresh_after: datetime,
) -> _Plan:
    full: list[str] = []
    tail: list[str] = []
    finished: set[str] = set()
    for sym in symbols:
        c = coverage.get(sym)
        if c is None or date.fromisoformat(c["start"]) > (start + timedelta(days=5)).date():
            full.append(sym)  # never downloaded for (all of) this window
            continue
        checked = datetime.fromisoformat(c["checked"])
        if sym not in last_stored:
            if now - checked > RETRY_MISSING:
                full.append(sym)  # nobody had it last time; ask again now and then
            continue
        if newest is not None and last_stored[sym] < newest - FINISHED_AFTER:
            finished.add(sym)
            continue
        if checked < fresh_after:
            tail.append(sym)
    return _Plan(full, tail, finished)


def _rebased(h: PriceHistory, symbol: str, stored_close: dict[tuple[str, datetime], float]) -> bool:
    for b in h.bars:
        old = stored_close.get((symbol, b.timestamp))
        if old is not None and abs(b.close - old) > REBASE_TOLERANCE * max(1.0, abs(old)):
            return True
    return False


def _frames_from_rows(rows: list[tuple[object, ...]], start: datetime) -> dict[str, pd.DataFrame]:
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=list(repo.BAR_COLUMNS))
    ts = pd.to_datetime(df["ts"], utc=True)
    df = df[ts >= pd.Timestamp(start)]
    ts = ts[ts >= pd.Timestamp(start)]
    out: dict[str, pd.DataFrame] = {}
    for sym, idx in df.groupby("symbol", sort=False).indices.items():
        part = df.iloc[idx]
        out[str(sym)] = _ohlcv_frame(
            pd.DatetimeIndex(ts.iloc[idx]),
            part["open"].to_numpy(dtype=float),
            part["high"].to_numpy(dtype=float),
            part["low"].to_numpy(dtype=float),
            part["close"].to_numpy(dtype=float),
            np.nan_to_num(part["volume"].to_numpy(dtype=float)),
        )
    return out
