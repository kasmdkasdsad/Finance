"""Daily stock picks: factor screen over a configurable universe with a 1–10 rating."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.gateway import Resolved
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, next_trading_day, regular_close
from quantpulse.domain import screener
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus, Provenance
from quantpulse.schemas.market import PriceHistory, Quote
from quantpulse.schemas.model import LiveScore
from quantpulse.schemas.picks import DailyPicks, FactorScores, PicksEmailResult, PicksMethod, StockPick
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService
from quantpulse.services.notifications import EmailNotifier, SyntheticDataRefused, render_picks_email

if TYPE_CHECKING:
    from quantpulse.services.forecast import ForecastService
    from quantpulse.services.model import ModelService

HISTORY_DAYS = STANDARD_HISTORY_DAYS  # > 252 trading days for 12-1 momentum and the 200-day SMA
FACTOR_RULE = (
    "Cross-sectional factor screen on daily closes: 12-1 month momentum (20%), 3-month momentum (10%), "
    "trend vs 50/200-day averages (25%), 6-month return/volatility (25%), low 3-month volatility (10%) and "
    "short-term RSI pullback (10%). Factors are z-scored across the universe."
)
MODEL_RULE = (
    "A ridge-regression stock model on 18 price, volatility and volume features, retrained monthly "
    "walk-forward and scored only on data it never saw, predicts each stock's 21-day return relative to the "
    "others; probabilities are calibrated from its out-of-sample record."
)
RATING_RULE = (
    "The score is mapped to a 1-10 rating via round(1 + 9·Φ(z)); ratings are relative to the universe."
)
METHODOLOGY = {
    "factors": f"{FACTOR_RULE} {RATING_RULE}",
    "model": f"{MODEL_RULE} {RATING_RULE}",
    "blend": f"Average of two standardised scores. (1) {FACTOR_RULE} (2) {MODEL_RULE} {RATING_RULE}",
}


def closes_with_quote(history: PriceHistory, quote: Quote) -> list[float]:
    """Daily closes with the live price as today's close.

    A quote from a later New York session than the last bar is appended; a newer quote from the same
    session (a provider that already publishes today's partial bar) replaces that bar's close.
    """
    closes = list(history.closes)
    if not history.bars or quote.timestamp <= history.bars[-1].timestamp:
        return closes
    last_day = history.bars[-1].timestamp.astimezone(NEW_YORK).date()
    if quote.timestamp.astimezone(NEW_YORK).date() > last_day:
        closes.append(quote.price)
    else:
        closes[-1] = quote.price
    return closes


class PicksService:
    def __init__(self, settings: Settings, clock: Clock, market: MarketService) -> None:
        self._settings = settings
        self._clock = clock
        self._market = market
        # Optional collaborators wired by the container (the picks work without them).
        self.model: ModelService | None = None
        self.forecast: ForecastService | None = None

    async def daily(
        self,
        top_n: int | None = None,
        *,
        force_refresh: bool = False,
        method: PicksMethod = "auto",
        with_forecast: bool = True,
    ) -> CompositeEnvelope[DailyPicks]:
        """Rank the universe. ``method``: ``factors`` (the hand-set rule), ``model`` (the walk-forward stock
        model), ``blend`` (average of both z-scores) or ``auto`` (blend only if the model has shown
        out-of-sample skill, otherwise factors)."""
        universe = list(self._settings.picks_universe)
        top_n = top_n or self._settings.picks_top_n
        notes: list[str] = []
        live: dict[str, LiveScore] = {}
        report = None
        if method != "factors" and self.model is not None:
            try:
                live, report = await self.model.live_scores()
            except DomainError as exc:
                notes.append(f"Stock model unavailable ({exc}); ranked by the factor rule.")
        used: str = "factors"
        if report is not None:
            if method == "auto":
                used = "blend" if report.has_skill else "factors"
                if not report.has_skill:
                    notes.append(
                        "The stock model has not shown reliable out-of-sample skill, so picks are ranked by the "
                        "factor rule; the model's probabilities are shown for reference."
                    )
            else:
                used = method
        elif method in ("model", "blend"):
            notes.append("No stock model is available; ranked by the factor rule.")
        if used in ("model", "blend") and len(live) > len(universe):
            # The model covers more stocks than the picks list (e.g. the whole S&P 500): its best-ranked names
            # compete too, so the day's top pick can come from anywhere in the model's universe.
            ranked = sorted(live.values(), key=lambda x: x.rank)
            extra = [x.symbol for x in ranked[: 2 * top_n] if x.symbol not in universe]
            if extra:
                shown = ", ".join(extra[:8]) + ("…" if len(extra) > 8 else "")
                notes.append(
                    f"The stock model ranks {len(live)} stocks; its top-ranked names outside the picks list "
                    f"({shown}) are screened too."
                )
                universe += extra
        sem = asyncio.Semaphore(4)

        async def load(symbol: str) -> tuple[Resolved[PriceHistory], Resolved[Quote]]:
            async with sem:
                hist = await self._market.history(symbol, "1d", HISTORY_DAYS, force_refresh=force_refresh)
                quote = await self._market.quote(symbol, force_refresh=force_refresh)
            return hist, quote

        loaded = await asyncio.gather(*(load(s) for s in universe))
        raw: dict[str, screener.RawFactors] = {}
        skipped: dict[str, str] = {}
        info: dict[str, tuple[Resolved[Quote], DataStatus]] = {}
        sources: dict[str, Provenance] = {}
        for symbol, (hist_r, quote_r) in zip(universe, loaded, strict=True):
            status = DataStatus.worst([hist_r.status, quote_r.status])
            sources[symbol] = (
                hist_r.provenance if hist_r.status.rank >= quote_r.status.rank else quote_r.provenance
            )
            try:
                raw[symbol] = screener.compute_factors(closes_with_quote(hist_r.value, quote_r.value))
            except ValueError as exc:
                skipped[symbol] = str(exc)
                continue
            info[symbol] = (quote_r, status)

        results = {r.symbol: r for r in screener.screen(raw)}

        def combined(symbol: str) -> float:
            factor = results[symbol].standardized
            score = live.get(symbol)
            if used == "model" and score is not None:
                return score.z
            if used == "blend" and score is not None:
                return 0.5 * factor + 0.5 * score.z
            return factor

        scores = {s: combined(s) for s in results}
        mean = sum(scores.values()) / len(scores) if scores else 0.0
        sd = (sum((v - mean) ** 2 for v in scores.values()) / len(scores)) ** 0.5 if scores else 0.0
        order = sorted(scores, key=lambda s: (-scores[s], s))
        picks: list[StockPick] = []
        for rank, symbol in enumerate(order, start=1):
            r = results[symbol]
            quote_r, status = info[symbol]
            q = quote_r.value
            f = r.factors
            score = live.get(symbol)
            z = (scores[symbol] - mean) / sd if sd > 0 else 0.0
            picks.append(
                StockPick(
                    rank=rank,
                    symbol=symbol,
                    name=q.name,
                    price=q.price,
                    change_percent=q.change_percent,
                    rating=screener.rating_from_z(z),
                    score=round(scores[symbol], 4),
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
                    factor_rating=r.rating,
                    model_rank=score.rank if score else None,
                    sector=score.sector_label if score else None,
                    prob_outperform=score.prob_outperform if score else None,
                    expected_excess_return=score.expected_excess_return if score else None,
                )
            )
        top = picks[:top_n]
        if with_forecast and self.forecast is not None and top:
            top = await self._with_forecasts(top, notes)

        now = self._clock.now()
        local = now.astimezone(NEW_YORK)
        # Picks are for the session that has not closed yet: today before the close, else the next session.
        if is_trading_day(local.date()) and local.time() < regular_close(local.date()):
            trading_day = local.date()
        else:
            trading_day = next_trading_day(local.date())
        overall = DataStatus.worst([p.data_status for p in top]) if top else DataStatus.SYNTHETIC
        data = DailyPicks(
            as_of=now,
            trading_day=trading_day.isoformat(),
            universe_size=len(universe),
            screened=len(picks),
            picks=top,
            top_pick=top[0] if top else None,
            data_status=overall,
            methodology=METHODOLOGY[used],
            skipped=skipped,
            method=used,
            requested_method=method,
            benchmark=self._settings.benchmark_symbol,
            model_verdict=report.verdict if report else None,
            model_has_skill=report.has_skill if report else None,
            notes=notes,
        )
        return CompositeEnvelope(data=data, meta=CompositeMeta.from_sources(sources, now))

    async def _with_forecasts(self, picks: list[StockPick], notes: list[str]) -> list[StockPick]:
        assert self.forecast is not None
        forecast = self.forecast
        sem = asyncio.Semaphore(4)

        async def one(p: StockPick) -> StockPick:
            async with sem:
                try:
                    env = await forecast.forecast(p.symbol, (21,), include_options=False)
                except DomainError:
                    return p
            h = env.data.horizons[0]
            e = env.data.earnings
            return p.model_copy(
                update={
                    "prob_up_21d": h.prob_up,
                    "low_21d": h.band.p05,
                    "high_21d": h.band.p95,
                    "earnings_date": e.next_date if e else None,
                    "earnings_in_sessions": e.sessions_ahead if e else None,
                    "typical_earnings_move": e.typical_move if e else None,
                }
            )

        out = await asyncio.gather(*(one(p) for p in picks))
        if any(p.low_21d is None for p in out):
            notes.append("Some 21-day price ranges could not be computed.")
        soon = [p.symbol for p in out if p.earnings_in_sessions is not None and p.earnings_in_sessions <= 21]
        if soon:
            notes.append(
                f"Earnings are due within the 21-day horizon for {', '.join(soon)}: expect a bigger move either way "
                "(the ranges include the jump)."
            )
        return list(out)

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
