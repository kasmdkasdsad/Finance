"""Per-stock probabilistic forecasts: volatility model + CAPM drift + options-implied view + calibration."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from quantpulse.config import Settings
from quantpulse.core.cache import TTLCache
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.market_calendar import NEW_YORK, upcoming_sessions
from quantpulse.quant import forecasting as fc
from quantpulse.quant import implied
from quantpulse.quant import volatility as vol
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus, Provenance
from quantpulse.schemas.forecast import (
    Band,
    CalibrationOut,
    ConePoint,
    DriftOut,
    HorizonOut,
    ImpliedView,
    StockForecast,
    VolModelOut,
)
from quantpulse.schemas.market import PriceHistory
from quantpulse.schemas.options import Smile, VolSurface
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService
from quantpulse.services.options import OptionsService
from quantpulse.services.picks import closes_with_quote
from quantpulse.services.portfolio import closes_frame
from quantpulse.services.rates import RatesService

if TYPE_CHECKING:
    from quantpulse.services.model import ModelService

logger = logging.getLogger(__name__)

FORECAST_HISTORY_DAYS = STANDARD_HISTORY_DAYS
CALIBRATION_HISTORY_DAYS = STANDARD_HISTORY_DAYS
BETA_WINDOW = 504
CALIBRATION_TTL = 6 * 3600.0


def blume_beta(history: PriceHistory, benchmark: PriceHistory) -> float | None:
    """Two-year daily beta, Blume-adjusted towards 1 (0.67·β + 0.33), or ``None`` if data is short."""
    frame = closes_frame({"s": history, "b": benchmark})
    rets = frame.pct_change().dropna().iloc[-BETA_WINDOW:]
    if len(rets) < 60 or float(rets["b"].var()) == 0:
        return None
    raw = float(rets["s"].cov(rets["b"]) / rets["b"].var())
    return 0.67 * raw + 0.33


def _band(values: dict[float, float] | Sequence[float]) -> Band:
    if isinstance(values, dict):
        return Band(p05=values[0.05], p25=values[0.25], p50=values[0.5], p75=values[0.75], p95=values[0.95])
    return Band(p05=values[0], p25=values[1], p50=values[2], p75=values[3], p95=values[4])


def implied_view(
    smile: Smile, spot: float, target: float | None, status: DataStatus, now: datetime
) -> ImpliedView | None:
    pts = [p for p in smile.points if p.iv > 0]
    if smile.atm_iv is None or len(pts) < 3 or smile.years <= 0:
        return None
    fn = implied.smile_function([p.log_moneyness for p in pts], [p.iv for p in pts])
    dist = implied.risk_neutral_distribution(smile.forward, smile.years, smile.rate, fn, smile.atm_iv)
    move, abs_move = implied.implied_move(smile.atm_iv, smile.years)
    return ImpliedView(
        expiration=smile.expiration,
        days_to_expiry=smile.years * 365.0,
        atm_iv=smile.atm_iv,
        move_1sd=move,
        expected_abs_move=abs_move,
        prob_up=dist.prob_above(spot),
        prob_above_target=dist.prob_above(target) if target else None,
        band=_band([dist.quantile(q) for q in fc.QUANTILES]),
        data_status=status,
    )


def nearest_smile(surface: VolSurface, calendar_days: float) -> Smile | None:
    """The expiry closest to ``calendar_days`` (within ±50% or a week), with a usable ATM volatility."""
    candidates = [s for s in surface.smiles if s.atm_iv is not None and len(s.points) >= 3]
    if not candidates:
        return None
    best = min(candidates, key=lambda s: abs(s.years * 365.0 - calendar_days))
    if abs(best.years * 365.0 - calendar_days) > max(7.0, 0.5 * calendar_days):
        return None
    return best


class ForecastService:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        cache: TTLCache,
        market: MarketService,
        rates: RatesService,
        options: OptionsService,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._cache = cache
        self._market = market
        self._rates = rates
        self._options = options
        self.model: ModelService | None = None  # wired by the container (optional model tilt)

    async def forecast(
        self,
        symbol: str,
        horizons: Sequence[int] = (5, 21, 63),
        *,
        target: float | None = None,
        include_options: bool = True,
        calibrate: bool = False,
        with_model: bool = False,
        use_close: bool = False,
    ) -> CompositeEnvelope[StockForecast]:
        """``use_close`` anchors the forecast on the last official daily close instead of the live quote
        (the prediction ledger does this so every logged prediction is graded close-to-close)."""
        s = self._settings
        bench = s.benchmark_symbol
        hist_r, quote_r, bench_r, curve_r, div_r = await asyncio.gather(
            self._market.history(symbol, "1d", FORECAST_HISTORY_DAYS),
            self._market.quote(symbol),
            self._market.history(bench, "1d", FORECAST_HISTORY_DAYS),
            self._rates.curve(),
            self._market.dividend_yield(symbol),
        )
        sources: dict[str, Provenance] = {
            "price_history": hist_r.provenance,
            "quote": quote_r.provenance,
            "benchmark_history": bench_r.provenance,
            "yield_curve": curve_r.provenance,
            "dividend_yield": div_r.provenance,
        }
        notes: list[str] = []
        if use_close:
            closes = np.asarray(hist_r.value.closes, dtype=float)
            spot = float(closes[-1]) if closes.size else quote_r.value.price
        else:
            spot = quote_r.value.price
            closes = np.asarray(closes_with_quote(hist_r.value, quote_r.value), dtype=float)
        if closes.size < 61:
            raise DomainError(f"only {closes.size} daily closes for {symbol}; at least 61 are needed")
        rf = RatesService.rate_from_curve(curve_r.value, 0.25).continuous_rate
        beta = blume_beta(hist_r.value, bench_r.value)
        q = float(div_r.value or 0.0)
        erp = s.equity_risk_premium
        mu = rf + (beta if beta is not None else 1.0) * erp - q
        method = "CAPM: risk-free + beta × equity risk premium − dividend yield"
        if beta is None:
            notes.append("Not enough overlapping history for a beta; using 1.0.")
        model_alpha = None
        if with_model and self.model is not None:
            live = self.model.cached_live(symbol)
            if live is not None:
                model_h = self.model.default_horizon
                model_alpha = float((1 + live.expected_excess_return) ** (252 / model_h) - 1)
                mu += model_alpha
                method += (
                    f" + model tilt ({live.expected_excess_return:+.2%} over {model_h} days, calibrated)"
                )
            else:
                notes.append("No recent stock-model run is cached, so no model tilt was applied.")

        hs = sorted({int(h) for h in horizons})
        fit = await self._fit(symbol, hist_r.value, closes)
        result = await asyncio.to_thread(fc.forecast_prices, closes, spot, hs, mu, fit=fit)
        now = self._clock.now()
        sessions = upcoming_sessions(now, hs[-1])
        horizons_out: list[HorizonOut] = []

        surface = None
        surface_status = None
        if include_options:
            try:
                env = await self._options.surface(symbol, max_expirations=10)
                surface, surface_status = env.data, env.meta.status
                sources.update({f"options:{k}": v for k, v in env.meta.sources.items()})
                if surface_status is DataStatus.SYNTHETIC:
                    notes.append("Option prices are synthetic, so the implied view is illustrative only.")
            except DomainError as exc:
                notes.append(f"No options view: {exc}")
        for h in result.horizons:
            target_date = sessions[h.days - 1]
            implied_out = None
            if surface is not None and surface_status is not None:
                days = (
                    datetime.combine(target_date, datetime.min.time(), NEW_YORK) - now
                ).total_seconds() / 86400
                smile = nearest_smile(surface, max(days, 1.0))
                if smile is not None:
                    implied_out = implied_view(smile, spot, target, surface_status, now)
            horizons_out.append(
                HorizonOut(
                    days=h.days,
                    target_date=target_date,
                    expected_price=h.expected_price,
                    median_price=h.median_price,
                    band=_band(h.quantiles),
                    prob_up=h.prob_up,
                    prob_above_target=result.prob_above(target, h.days) if target else None,
                    expected_return=h.expected_price / spot - 1,
                    volatility=h.volatility,
                    var_95=h.var_95,
                    expected_shortfall_95=h.expected_shortfall_95,
                    implied=implied_out,
                )
            )
        cone = [
            ConePoint(
                day=i + 1,
                date=sessions[i],
                **{f"p{round(qq * 100):02d}": result.cone[qq][i] for qq in fc.QUANTILES},
            )
            for i in range(hs[-1])
        ]
        fit = result.fit
        bars = hist_r.value.bars
        realized: dict[str, float | None] = {
            "close_to_close_21d": None,
            "parkinson_21d": None,
            "garman_klass_21d": None,
        }
        if len(bars) >= 22:
            o = np.array([b.open for b in bars])
            hi = np.array([b.high for b in bars])
            lo = np.array([b.low for b in bars])
            c = np.array([b.close for b in bars])
            realized = {
                "close_to_close_21d": vol.close_to_close(c, 21),
                "parkinson_21d": vol.parkinson(hi, lo, 21),
                "garman_klass_21d": vol.garman_klass(o, hi, lo, c, 21),
            }
        calibration = (
            await self.calibration(symbol, hs[1] if len(hs) > 1 else hs[0], mu) if calibrate else None
        )
        status = DataStatus.worst([hist_r.status, quote_r.status])
        if status is DataStatus.SYNTHETIC:
            notes.insert(
                0, "Price history is synthetic (live data unavailable): this forecast is illustrative only."
            )
        data = StockForecast(
            symbol=symbol,
            as_of=now,
            spot=spot,
            target=target,
            horizons=horizons_out,
            cone=cone,
            volatility=VolModelOut(
                method=fit.method,
                alpha=fit.alpha,
                beta=fit.beta,
                nu=fit.nu if math.isfinite(fit.nu) else None,
                persistence=fit.persistence,
                half_life_days=fit.half_life_days,
                current_vol_annual=math.sqrt(fit.next_variance * 252),
                long_run_vol_annual=math.sqrt(fit.long_run_variance * 252),
                forecast_vol_annual_21d=fit.annualized_volatility(21),
                n_obs=fit.n_obs,
            ),
            drift=DriftOut(
                annual_expected_return=mu,
                risk_free=rf,
                beta=beta,
                equity_risk_premium=erp,
                dividend_yield=q,
                model_alpha=model_alpha,
                method=method,
            ),
            realized_vol=realized,
            calibration=calibration,
            notes=notes,
            data_status=status,
        )
        return CompositeEnvelope(data=data, meta=CompositeMeta.from_sources(sources, now))

    async def _fit(self, symbol: str, history: PriceHistory, closes: np.ndarray) -> vol.GarchFit:
        """GARCH parameters are estimated once per symbol per daily bar (the slow part) and cached; the live
        price only re-runs the variance recursion, which is cheap."""
        last = history.bars[-1].timestamp.isoformat() if history.bars else "none"
        key = f"garch-fit:{symbol}:{last}:{len(history.bars)}"
        hit = self._cache.get(key)
        if hit is None:
            base = np.asarray(history.closes, dtype=float)
            if base.size < 61:
                base = closes
            params = await asyncio.to_thread(vol.fit_best, vol.log_returns(base))
            self._cache.set(key, params, CALIBRATION_TTL)
        else:
            params = hit.value
        return fc.refilter(params, vol.log_returns(closes))

    async def calibration(self, symbol: str, horizon: int, annual_drift: float) -> CalibrationOut:
        """Walk-forward test of this forecaster on the stock's own history (cached for 6 hours)."""
        today = self._clock.now().astimezone(NEW_YORK).date().isoformat()
        key = f"forecast-calibration:{symbol}:{horizon}:{today}"
        hit = self._cache.get(key)
        if hit is not None:
            return hit.value
        hist_r = await self._market.history(symbol, "1d", CALIBRATION_HISTORY_DAYS)
        bars = hist_r.value.bars
        closes = np.array([b.close for b in bars])
        rep = await asyncio.to_thread(
            fc.evaluate_forecasts, closes, horizon, step=5, min_obs=500, annual_drift=annual_drift
        )
        dates = [b.timestamp.astimezone(NEW_YORK).date() for b in bars]
        out = CalibrationOut(
            horizon=horizon,
            n=rep.n,
            effective_n=rep.effective_n,
            coverage_50=rep.coverage(50),
            coverage_90=rep.coverage(90),
            pit_histogram=rep.pit_histogram(),
            brier=rep.brier(),
            brier_climatology=rep.brier_climatology(),
            brier_skill=rep.brier_skill(),
            direction_hit_rate=rep.direction_hit_rate(),
            volatility_ratio=rep.volatility_ratio(),
            start=dates[rep.records[0].origin],
            end=dates[min(rep.records[-1].origin + horizon, len(dates) - 1)],
        )
        self._cache.set(key, out, CALIBRATION_TTL)
        return out
