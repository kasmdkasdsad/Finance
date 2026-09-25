"""Per-stock probabilistic forecasts: volatility model + earnings jumps + options blend + CAPM drift.

* The GARCH-t model is fitted with earnings days neutralised, and the next earnings reaction (a vendor's
  scheduled date, or the company's quarterly rhythm) is simulated as a jump sized from the stock's own
  past reactions.
* When an option surface is available, the diffusive variance is blended with the options' ATM implied
  variance, ex the variance risk premium and ex the earnings jumps the options price
  (``QP_FORECAST_IV_WEIGHT``, ``QP_FORECAST_VARIANCE_PREMIUM``).
* The drift is CAPM, optionally tilted by the stock model's calibrated excess return.
"""

from __future__ import annotations

import asyncio
import bisect
import functools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from quantpulse.config import Settings
from quantpulse.core.cache import TTLCache
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, next_trading_day, upcoming_sessions
from quantpulse.domain import earnings as earn
from quantpulse.quant import forecasting as fc
from quantpulse.quant import implied
from quantpulse.quant import volatility as vol
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus, Provenance
from quantpulse.schemas.forecast import (
    Band,
    CalibrationOut,
    ConePoint,
    DriftOut,
    EarningsForecastOut,
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
    from quantpulse.services.reference import ReferenceService

logger = logging.getLogger(__name__)

FORECAST_HISTORY_DAYS = STANDARD_HISTORY_DAYS
CALIBRATION_HISTORY_DAYS = STANDARD_HISTORY_DAYS
BETA_WINDOW = 504
CALIBRATION_TTL = 6 * 3600.0
NEXT_CYCLE_DAYS = 91  # a second earnings release inside a long horizon is one quarter after the next
IV_POINT_MAX_EXTRA = 30  # use expiries up to this many sessions past the longest horizon


@dataclass
class EarningsInputs:
    """Earnings inputs for one forecast."""

    flags: np.ndarray  # one per return of ``closes``: True on reaction days
    sample: np.ndarray  # past earnings-day log returns
    upcoming: list[date]  # expected reaction days ahead
    source: str | None
    typical_move: float | None
    provenance: Provenance | None


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


def implied_points(surface: VolSurface, sessions: Sequence[date]) -> list[tuple[int, float]]:
    """``(trading days to expiry, ATM implied vol)`` for each expiry inside ``sessions``."""
    out: list[tuple[int, float]] = []
    for smile in surface.smiles:
        if smile.atm_iv is None or smile.atm_iv <= 0:
            continue
        days = bisect.bisect_right(list(sessions), smile.expiration)
        if 1 <= days < len(sessions):
            out.append((days, float(smile.atm_iv)))
    return sorted(out)


def _nearest_iv(points: Sequence[tuple[int, float]], days: int) -> float | None:
    usable = [p for p in points if abs(p[0] - days) <= max(5, days // 2)]
    return min(usable, key=lambda p: abs(p[0] - days))[1] if usable else None


def _blended_vol(fit: vol.GarchFit, scale: np.ndarray | None, days: int) -> float:
    base = fit.variance_term_structure(days)
    mult = np.ones(days) if scale is None else np.asarray(scale[:days], dtype=float)
    return float(math.sqrt(float((base * mult).mean()) * 252))


def _earnings_out(
    e: EarningsInputs | None, offsets: Sequence[int], horizons: Sequence[int], jumps: fc.JumpSpec | None
) -> EarningsForecastOut | None:
    if e is None:
        return None
    first = offsets[0] if offsets else None
    nxt = e.upcoming[0] if e.upcoming else None
    in_horizons = [h for h in horizons if first is not None and first <= h]
    return EarningsForecastOut(
        next_date=nxt,
        source=e.source,
        sessions_ahead=first,
        in_horizons=in_horizons,
        typical_move=e.typical_move,
        events_used=int(e.sample.size),
        modelled=jumps is not None and bool(in_horizons),
    )


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
        self.reference: ReferenceService | None = None  # wired by the container (earnings dates)

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
        now = self._clock.now()
        sessions = upcoming_sessions(now, hs[-1] + IV_POINT_MAX_EXTRA)
        dates = [bar.timestamp.astimezone(NEW_YORK).date() for bar in hist_r.value.bars]
        if len(closes) > len(dates):  # the live quote was appended as today's close
            dates.append(quote_r.value.timestamp.astimezone(NEW_YORK).date())
        prices_real = DataStatus.worst([hist_r.status, quote_r.status]) is not DataStatus.SYNTHETIC
        earnings = await self.earnings_inputs(symbol, dates, closes, prices_real, now)
        if earnings is not None and earnings.provenance is not None:
            sources["earnings_events"] = earnings.provenance
        fit = await self._fit(symbol, hist_r.value, closes, None if earnings is None else earnings.flags)
        jumps = None
        offsets: list[int] = []
        if earnings is not None:
            # Offsets cover the option expiries too: an option priced across a later release carries that
            # jump, which the implied-volatility blend must take out.
            offsets = [sessions.index(d) + 1 for d in earnings.upcoming if d in sessions]
            if earnings.sample.size >= fc.MIN_JUMP_EVENTS:
                jumps = fc.JumpSpec(days=tuple(offsets), sample=earnings.sample)
            elif offsets and offsets[0] <= hs[-1]:
                notes.append(
                    f"Earnings are due in {offsets[0]} sessions, but fewer than {fc.MIN_JUMP_EVENTS} past "
                    "reactions are known, so the jump is not modelled."
                )

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
        scale, implied_21, weight = None, None, 0.0
        if surface is not None and surface_status is not None:
            if prices_real and surface_status is DataStatus.SYNTHETIC:
                notes.append(
                    "Options are synthetic while prices are live, so implied volatility is not blended in."
                )
            else:
                points = implied_points(surface, sessions)
                implied_21 = _nearest_iv(points, 21)
                scale = fc.variance_scale(
                    fit,
                    hs[-1],
                    [p for p in points if p[0] <= hs[-1] + IV_POINT_MAX_EXTRA],
                    weight=s.forecast_iv_weight,
                    premium=s.forecast_variance_premium,
                    jumps=jumps,
                )
                weight = s.forecast_iv_weight if scale is not None else 0.0
        result = await asyncio.to_thread(
            fc.forecast_prices, closes, spot, hs, mu, fit=fit, jumps=jumps, scale=scale
        )
        horizons_out: list[HorizonOut] = []
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
                earnings_days_excluded=fit.jump_days,
                garch_vol_annual_21d=fit.annualized_volatility(21),
                implied_vol_annual_21d=implied_21,
                blended_vol_annual_21d=_blended_vol(fit, scale, 21),
                iv_weight=weight,
                variance_premium=s.forecast_variance_premium if weight > 0 else None,
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
            earnings=_earnings_out(earnings, offsets, hs, jumps),
            calibration=calibration,
            notes=notes,
            data_status=status,
        )
        return CompositeEnvelope(data=data, meta=CompositeMeta.from_sources(sources, now))

    async def earnings_inputs(
        self, symbol: str, dates: Sequence[date], closes: np.ndarray, prices_real: bool, now: datetime
    ) -> EarningsInputs | None:
        """Reaction-day flags for the history, past reaction sizes and the next reaction days.

        Filings are never mixed across worlds: real prices only use real (or stored) filings."""
        if self.reference is None or not self._settings.forecast_earnings_jumps or len(dates) < 2:
            return None
        try:
            view, events_r = await self.reference.earnings(symbol)
        except DomainError:
            return None
        if prices_real and events_r.status is DataStatus.SYNTHETIC:
            return None
        days = {earn.reaction_day(t) for t in events_r.value.earnings}
        flags = np.array([d in days for d in dates[1:]], dtype=bool)
        upcoming: list[date] = []
        nxt = view.next_reaction_date
        if nxt is not None and nxt >= now.astimezone(NEW_YORK).date():
            second = nxt + timedelta(days=NEXT_CYCLE_DAYS)
            upcoming = [nxt, second if is_trading_day(second) else next_trading_day(second)]
        return EarningsInputs(
            flags=flags,
            sample=vol.log_returns(closes)[flags],
            upcoming=upcoming,
            source=view.next_source,
            typical_move=view.typical_move,
            provenance=events_r.provenance,
        )

    async def _fit(
        self, symbol: str, history: PriceHistory, closes: np.ndarray, flags: np.ndarray | None = None
    ) -> vol.GarchFit:
        """GARCH parameters are estimated once per symbol per daily bar (the slow part) and cached; the live
        price only re-runs the variance recursion, which is cheap. ``flags`` marks earnings days (one per
        return of ``closes``), which are neutralised."""
        last = history.bars[-1].timestamp.isoformat() if history.bars else "none"
        n_jumps = int(flags.sum()) if flags is not None else 0
        key = f"garch-fit:{symbol}:{last}:{len(history.bars)}:{n_jumps}"
        hit = self._cache.get(key)
        if hit is None:
            base = np.asarray(history.closes, dtype=float)
            if base.size < 61:
                base = closes
            base_flags = None if flags is None else flags[: base.size - 1]
            params = await asyncio.to_thread(vol.fit_best, vol.log_returns(base), base_flags)
            self._cache.set(key, params, CALIBRATION_TTL)
        else:
            params = hit.value
        return fc.refilter(params, vol.log_returns(closes), flags)

    async def calibration(self, symbol: str, horizon: int, annual_drift: float) -> CalibrationOut:
        """Walk-forward test of this forecaster on the stock's own history (cached for 6 hours)."""
        today = self._clock.now().astimezone(NEW_YORK).date().isoformat()
        key = f"forecast-calibration:{symbol}:{horizon}:{today}:{int(self._settings.forecast_earnings_jumps)}"
        hit = self._cache.get(key)
        if hit is not None:
            return hit.value
        hist_r = await self._market.history(symbol, "1d", CALIBRATION_HISTORY_DAYS)
        bars = hist_r.value.bars
        closes = np.array([b.close for b in bars])
        dates = [b.timestamp.astimezone(NEW_YORK).date() for b in bars]
        earnings = await self.earnings_inputs(
            symbol, dates, closes, hist_r.status is not DataStatus.SYNTHETIC, self._clock.now()
        )
        rep = await asyncio.to_thread(
            functools.partial(
                fc.evaluate_forecasts,
                closes,
                horizon,
                step=5,
                min_obs=500,
                annual_drift=annual_drift,
                jump_flags=None if earnings is None else earnings.flags,
            )
        )
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
