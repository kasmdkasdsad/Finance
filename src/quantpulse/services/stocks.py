"""Stock intelligence report: everything the platform knows about one ticker, on one page."""

from __future__ import annotations

import asyncio
import logging
import math

import numpy as np

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError, NotFoundError
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.domain import screener
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus, Provenance
from quantpulse.schemas.forecast import StockForecast
from quantpulse.schemas.fundamentals import DCFRequest
from quantpulse.schemas.stocks import (
    ModelView,
    PricePoint,
    StockReport,
    Technicals,
    TrackRecord,
    ValuationView,
)
from quantpulse.services.forecast import ForecastService
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService
from quantpulse.services.model import ModelService
from quantpulse.services.picks import closes_with_quote
from quantpulse.services.predictions import PredictionService
from quantpulse.services.valuation import ValuationService

logger = logging.getLogger(__name__)
REPORT_HISTORY_DAYS = STANDARD_HISTORY_DAYS


def _ret(c: np.ndarray, n: int) -> float | None:
    return float(c[-1] / c[-1 - n] - 1) if c.size > n else None


def _sma(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(c.size, np.nan)
    if c.size >= n:
        csum = np.cumsum(np.concatenate([[0.0], c]))
        out[n - 1 :] = (csum[n:] - csum[:-n]) / n
    return out


def _last(x: np.ndarray) -> float | None:
    return None if x.size == 0 or not math.isfinite(float(x[-1])) else float(x[-1])


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


class StockReportService:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        market: MarketService,
        forecast: ForecastService,
        model: ModelService,
        valuation: ValuationService,
        predictions: PredictionService,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._market = market
        self._forecast = forecast
        self._model = model
        self._valuation = valuation
        self._predictions = predictions

    async def report(
        self,
        symbol: str,
        *,
        target: float | None = None,
        include_model: bool = True,
        include_valuation: bool = True,
        include_options: bool = True,
    ) -> CompositeEnvelope[StockReport]:
        notes: list[str] = []
        hist_r, quote_r, forecast_env = await asyncio.gather(
            self._market.history(symbol, "1d", REPORT_HISTORY_DAYS),
            self._market.quote(symbol),
            self._forecast.forecast(symbol, (5, 21, 63), target=target, include_options=include_options),
        )
        if not hist_r.value.bars:
            raise NotFoundError(f"no price history for {symbol}")
        sources: dict[str, Provenance] = dict(forecast_env.meta.sources)
        sources["report_history"] = hist_r.provenance
        fc = forecast_env.data

        async def model_view() -> ModelView | None:
            if not include_model:
                return None
            try:
                score, rep = await self._model.score_symbol(symbol)
            except DomainError as exc:
                notes.append(f"Stock model unavailable: {exc}")
                return None
            if score is None:
                notes.append(
                    "The stock model could not score this symbol (under a year of usable live history)."
                )
                return None
            return ModelView(
                in_universe=symbol in rep.symbols,
                rank=score.rank,
                universe_size=len(rep.symbols) + (0 if symbol in rep.symbols else 1),
                z=score.z,
                rating=score.rating,
                prob_outperform=score.prob_outperform,
                expected_excess_return=score.expected_excess_return,
                horizon=rep.horizon,
                benchmark=rep.benchmark,
                has_skill=rep.has_skill,
                verdict=rep.verdict,
                as_of=rep.as_of,
                data_status=rep.data_status,
            )

        async def valuation_view() -> ValuationView | None:
            if not include_valuation:
                return None
            try:
                env = await self._valuation.dcf(symbol, DCFRequest(monte_carlo=None))
            except DomainError as exc:
                notes.append(f"No DCF: {exc}")
                return None
            v = env.data
            sources.update({f"dcf:{k}": p for k, p in env.meta.sources.items()})
            return ValuationView(
                value_per_share=v.dcf.value_per_share,
                current_price=v.dcf.current_price,
                upside=v.dcf.upside,
                wacc=v.wacc.wacc,
                terminal_value_share=v.dcf.terminal_value_share,
                warnings=v.warnings,
                data_status=env.meta.status,
            )

        model, valuation, card = await asyncio.gather(
            model_view(), valuation_view(), self._predictions.scorecard(symbol)
        )

        bars = hist_r.value.bars
        closes = np.asarray(closes_with_quote(hist_r.value, quote_r.value), dtype=float)
        sma20, sma50, sma200 = _sma(closes, 20), _sma(closes, 50), _sma(closes, 200)
        year = closes[-252:]
        log_r = np.diff(np.log(closes))
        peak = np.maximum.accumulate(year)
        technicals = Technicals(
            price=quote_r.value.price,
            change_percent=quote_r.value.change_percent,
            sma20=_last(sma20),
            sma50=_last(sma50),
            sma200=_last(sma200),
            rsi_14=screener.rsi(closes[-screener.RSI_WINDOW :]),
            high_52w=float(year.max()),
            low_52w=float(year.min()),
            from_high_52w=float(closes[-1] / year.max() - 1),
            return_1m=_ret(closes, 21),
            return_3m=_ret(closes, 63),
            return_6m=_ret(closes, 126),
            return_1y=_ret(closes, 252),
            volatility_3m=float(log_r[-63:].std(ddof=1) * math.sqrt(252)) if log_r.size >= 63 else None,
            max_drawdown_1y=float(np.min(year / peak - 1)),
            beta=fc.drift.beta,
            avg_volume_20d=float(np.mean([b.volume for b in bars[-20:]])) if bars else None,
        )
        dates = [b.timestamp.astimezone(NEW_YORK).date() for b in bars]
        n = len(bars)
        chart = [
            PricePoint(
                date=dates[i],
                close=bars[i].close,
                sma50=None if math.isnan(sma50[i]) else float(sma50[i]),
                sma200=None if math.isnan(sma200[i]) else float(sma200[i]),
            )
            for i in range(max(0, n - 252), n)
        ]

        forecast_scores = [s for s in card.sources if s.source == "forecast"]
        model_scores = [s for s in card.sources if s.source == "model"]
        track = TrackRecord(
            resolved=sum(s.resolved for s in card.sources),
            open=sum(s.open for s in card.sources),
            forecast_brier=next((s.brier for s in forecast_scores if s.horizon_days == 21), None),
            forecast_coverage_90=next((s.coverage_90 for s in forecast_scores if s.horizon_days == 21), None),
            model_hit_rate=model_scores[0].hit_rate if model_scores else None,
            recent=card.recent[:10],
        )
        summary = self._summary(symbol, fc, technicals, model, valuation, track)
        status = DataStatus.worst([hist_r.status, quote_r.status, fc.data_status])
        report = StockReport(
            symbol=symbol,
            name=quote_r.value.name,
            as_of=self._clock.now(),
            technicals=technicals,
            chart=chart,
            forecast=fc,
            model=model,
            valuation=valuation,
            track_record=track,
            summary=summary,
            notes=[*fc.notes, *notes],
            data_status=status,
        )
        return CompositeEnvelope(data=report, meta=CompositeMeta.from_sources(sources, report.as_of))

    def _summary(
        self,
        symbol: str,
        fc: StockForecast,
        t: Technicals,
        model: ModelView | None,
        valuation: ValuationView | None,
        track: TrackRecord,
    ) -> list[str]:
        out: list[str] = []
        month = next((h for h in fc.horizons if h.days == 21), fc.horizons[0])
        lo, hi = month.band.p05, month.band.p95
        out.append(
            f"Next {month.days} trading days (to {month.target_date}): 90% of simulated outcomes fall between "
            f"${lo:,.2f} and ${hi:,.2f} ({_pct(lo / fc.spot - 1)} to {_pct(hi / fc.spot - 1)}); "
            f"chance of finishing higher {month.prob_up:.0%}."
        )
        if month.implied is not None:
            imp = month.implied
            out.append(
                f"Options expiring {imp.expiration} price a ±{imp.move_1sd:.1%} one-standard-deviation move "
                f"(ATM implied volatility {imp.atm_iv:.0%}) versus {fc.volatility.forecast_vol_annual_21d:.0%} "
                "forecast by the volatility model."
            )
        if t.sma200 is not None:
            side = "above" if t.price > t.sma200 else "below"
            rsi = "" if t.rsi_14 is None else f"; RSI(14) {t.rsi_14:.0f}"
            if t.rsi_14 is not None and t.rsi_14 >= 70:
                rsi += " (overbought zone)"
            elif t.rsi_14 is not None and t.rsi_14 <= 30:
                rsi += " (oversold zone)"
            out.append(f"Trend: {side} its 200-day average (${t.sma200:,.2f}){rsi}.")
        if model is not None:
            skill = (
                "the model has shown out-of-sample skill"
                if model.has_skill
                else "the model has not yet shown reliable skill"
            )
            out.append(
                f"Stock model: ranked {model.rank} of {model.universe_size}; probability of beating "
                f"{model.benchmark} over {model.horizon} trading days {model.prob_outperform:.0%} ({skill})."
            )
        if valuation is not None and valuation.upside is not None:
            out.append(
                f"DCF fair value ${valuation.value_per_share:,.2f} ({_pct(valuation.upside)} vs the price) at a "
                f"{valuation.wacc:.1%} WACC; {valuation.terminal_value_share:.0%} of the value is the terminal value, "
                "so small assumption changes move it a lot."
            )
        if track.resolved:
            parts = [f"{track.resolved} graded predictions for {symbol}"]
            if track.forecast_coverage_90 is not None:
                parts.append(f"{track.forecast_coverage_90:.0%} landed inside the 90% range (target 90%)")
            out.append("Track record: " + "; ".join(parts) + ".")
        else:
            out.append(
                f"Track record: no graded predictions for {symbol} yet; the ledger grades them as they come due."
            )
        return out
