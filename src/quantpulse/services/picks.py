"""Daily stock picks: factor screen over a configurable universe with a 1–10 rating."""

from __future__ import annotations

import asyncio

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, next_trading_day, regular_close
from quantpulse.domain import screener
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus
from quantpulse.schemas.picks import DailyPicks, FactorScores, PicksEmailResult, StockPick
from quantpulse.services.market import MarketService
from quantpulse.services.notifications import EmailNotifier, SyntheticDataRefused, render_picks_email

HISTORY_DAYS = 400  # > 252 trading days for 12-1 momentum and the 200-day SMA
METHODOLOGY = (
    "Cross-sectional factor screen on daily closes: 12-1 month momentum (20%), 3-month momentum (10%), "
    "trend vs 50/200-day averages (25%), 6-month return/volatility (25%), low 3-month volatility (10%) and "
    "short-term RSI pullback (10%). Factors are z-scored across the universe; the composite is mapped to "
    "a 1-10 rating via round(1 + 9·Φ(z)). Ratings are relative to the screened universe."
)


class PicksService:
    def __init__(self, settings: Settings, clock: Clock, market: MarketService) -> None:
        self._settings = settings
        self._clock = clock
        self._market = market

    async def daily(
        self, top_n: int | None = None, *, force_refresh: bool = False
    ) -> CompositeEnvelope[DailyPicks]:
        universe = list(self._settings.picks_universe)
        top_n = top_n or self._settings.picks_top_n
        sem = asyncio.Semaphore(4)

        async def load(symbol: str):
            async with sem:
                hist = await self._market.history(symbol, "1d", HISTORY_DAYS, force_refresh=force_refresh)
                quote = await self._market.quote(symbol, force_refresh=force_refresh)
            return hist, quote

        loaded = await asyncio.gather(*(load(s) for s in universe))
        raw: dict[str, screener.RawFactors] = {}
        skipped: dict[str, str] = {}
        info = {}
        sources = {}
        for symbol, (hist_r, quote_r) in zip(universe, loaded, strict=True):
            status = DataStatus.worst([hist_r.status, quote_r.status])
            sources[symbol] = (
                hist_r.provenance if hist_r.status.rank >= quote_r.status.rank else quote_r.provenance
            )
            closes = list(hist_r.value.closes)
            # Use the live price as today's close when it is newer than the last daily bar.
            if hist_r.value.bars and quote_r.value.timestamp > hist_r.value.bars[-1].timestamp:
                closes.append(quote_r.value.price)
            try:
                raw[symbol] = screener.compute_factors(closes)
            except ValueError as exc:
                skipped[symbol] = str(exc)
                continue
            info[symbol] = (quote_r, status)

        results = screener.screen(raw)
        picks: list[StockPick] = []
        for rank, r in enumerate(results, start=1):
            quote_r, status = info[r.symbol]
            q = quote_r.value
            f = r.factors
            picks.append(
                StockPick(
                    rank=rank,
                    symbol=r.symbol,
                    name=q.name,
                    price=q.price,
                    change_percent=q.change_percent,
                    rating=r.rating,
                    score=round(r.composite, 4),
                    factors=FactorScores(
                        momentum_12_1=f.momentum_12_1,
                        momentum_3m=f.momentum_3m,
                        trend=f.trend,
                        risk_adjusted=f.risk_adjusted,
                        volatility_3m=f.volatility_3m,
                        rsi_14=f.rsi_14,
                    ),
                    factor_z={k: round(v, 3) for k, v in r.z.items()},
                    drivers=r.drivers,
                    data_status=status,
                    provider=quote_r.provenance.provider,
                )
            )
        now = self._clock.now()
        local = now.astimezone(NEW_YORK)
        # Picks are for the session that has not closed yet: today before the close, else the next session.
        if is_trading_day(local.date()) and local.time() < regular_close(local.date()):
            trading_day = local.date()
        else:
            trading_day = next_trading_day(local.date())
        top = picks[:top_n]
        overall = DataStatus.worst([p.data_status for p in top]) if top else DataStatus.SYNTHETIC
        data = DailyPicks(
            as_of=now,
            trading_day=trading_day.isoformat(),
            universe_size=len(universe),
            screened=len(picks),
            picks=top,
            top_pick=top[0] if top else None,
            data_status=overall,
            methodology=METHODOLOGY,
            skipped=skipped,
        )
        return CompositeEnvelope(data=data, meta=CompositeMeta.from_sources(sources, now))

    async def email_digest(
        self,
        notifier: EmailNotifier,
        recipients: list[str] | None = None,
        top_n: int | None = None,
        allow_synthetic: bool | None = None,
    ) -> PicksEmailResult:
        """Compute today's picks and email them. Refuses to mail synthetic numbers unless explicitly allowed."""
        allow = self._settings.picks_allow_synthetic_email if allow_synthetic is None else allow_synthetic
        to = list(recipients or self._settings.picks_recipients)
        envelope = await self.daily(top_n)
        picks = envelope.data
        if picks.data_status is DataStatus.SYNTHETIC and not allow:
            raise SyntheticDataRefused(
                "live market data is unavailable, so today's picks are synthetic; not emailing them "
                "(pass allow_synthetic=true to send a clearly-labelled synthetic digest)"
            )
        subject, text, body_html = render_picks_email(picks)
        await notifier.send(to, subject, text, body_html)
        return PicksEmailResult(
            sent_to=to,
            subject=subject,
            data_status=picks.data_status,
            top_pick=picks.top_pick.symbol if picks.top_pick else None,
        )
