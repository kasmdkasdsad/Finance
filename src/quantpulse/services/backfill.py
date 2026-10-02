"""Backfill the prediction ledger by replaying history point-in-time.

A live ledger needs months before its scorecard means anything. The backfill fills it from day one with
predictions the platform *would* have made, each using only data available on its date, graded against
what actually happened. Rows are stored with ``origin="backfill"`` and are always shown apart from the
live record, because a replay can never be quite as honest as a prediction logged before the fact.

Forecasts (every ``FORECAST_STEP`` sessions, for the picks universe)
    The same GARCH-t + earnings-jump forecaster, refitted walk-forward on each stock's own history: at each
    date the model sees returns before it only, jump sizes come from earlier earnings reactions, and the
    drift uses a beta estimated from the prior two years. Two things are not point-in-time and are said so:
    the risk-free rate and dividend yield are today's (the drift is a small part of a forecast), and no
    options are blended in (historical option prices are not available).

Model rankings (every ``MODEL_STEP`` sessions)
    The stock model's walk-forward out-of-sample predictions: every one was made by a model trained only on
    labels realised before its date. Probabilities come from an *expanding* calibration that uses only
    predictions whose outcomes were known by that date (dates without ``MIN_CALIBRATION_DATES`` of such
    history are skipped), so no row borrows from the future.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.jobs import Job, JobRegistry
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.domain import alpha_model as am
from quantpulse.quant import forecasting as fc
from quantpulse.schemas.common import DataStatus
from quantpulse.services.forecast import BETA_WINDOW, ForecastService
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService
from quantpulse.services.model import ModelService
from quantpulse.services.predictions import FORECAST_HORIZONS, FORECAST_VERSION, MODEL_VERSION
from quantpulse.services.rates import RatesService

logger = logging.getLogger(__name__)

FORECAST_STEP = 5
MODEL_STEP = 5
MIN_OBS = 500  # closes of history before the first backfilled forecast
MIN_CALIBRATION_DATES = 60
FORECAST_PATHS = 2000
MODEL_WAIT = 3600.0
CONCURRENCY = 2
BACKFILL_KEY = "ledger-backfill"


@dataclass
class BackfillResult:
    forecast_rows: int = 0
    model_rows: int = 0
    replaced: int = 0
    first_date: date | None = None
    last_date: date | None = None
    skipped: dict[str, str] = field(default_factory=dict)

    def span(self, d: date) -> None:
        self.first_date = d if self.first_date is None else min(self.first_date, d)
        self.last_date = d if self.last_date is None else max(self.last_date, d)


def rolling_beta(stock: np.ndarray, bench: np.ndarray, window: int = BETA_WINDOW) -> np.ndarray:
    """Blume-adjusted beta known at each close ``t`` (returns up to ``t``); NaN before 60 returns."""
    s = pd.Series(stock).pct_change(fill_method=None)
    b = pd.Series(bench).pct_change(fill_method=None)
    cov = s.rolling(window, min_periods=60).cov(b)
    var = b.rolling(window, min_periods=60).var()
    raw = (cov / var.replace(0.0, np.nan)).to_numpy(dtype=float)
    return 0.67 * raw + 0.33


def forecast_rows(
    symbol: str,
    dates: list[date],
    closes: np.ndarray,
    bench_by_day: dict[date, float],
    drift: np.ndarray,
    flags: np.ndarray | None,
    now: datetime,
    benchmark: str,
    status: DataStatus,
    version: str,
) -> list[dict[str, Any]]:
    """Walk-forward forecasts for one stock, graded against the realised closes."""
    n = closes.size - 1
    rows: list[dict[str, Any]] = []
    hs = list(FORECAST_HORIZONS)
    replay = fc.replay(
        closes,
        hs,
        step=FORECAST_STEP,
        min_obs=MIN_OBS,
        annual_drift=lambda t: float(drift[t]),
        n_paths=FORECAST_PATHS,
        jump_flags=flags,
        until=n - hs[0],
    )
    for t, cum in replay:
        spot = float(closes[t])
        made_on = dates[t]
        for h in hs:
            if t + h > n:
                continue
            lr = cum[:, h - 1]
            q05, q25, q50, q75, q95 = (spot * math.exp(x) for x in np.quantile(lr, fc.QUANTILES))
            realized = float(closes[t + h])
            target = dates[t + h]
            ret = realized / spot - 1
            b0, b1 = bench_by_day.get(made_on), bench_by_day.get(target)
            bench_ret = b1 / b0 - 1 if b0 and b1 else None
            rows.append(
                {
                    "created_at": now,
                    "made_on": made_on,
                    "target_date": target,
                    "symbol": symbol,
                    "source": "forecast",
                    "horizon_days": h,
                    "reference_price": spot,
                    "benchmark": benchmark,
                    "benchmark_reference": b0,
                    "prob_up": float(np.mean(lr > 0)),
                    "prob_outperform": None,
                    "expected_return": float(np.mean(np.exp(lr)) - 1),
                    "q05": q05,
                    "q25": q25,
                    "q50": q50,
                    "q75": q75,
                    "q95": q95,
                    "rank": None,
                    "model_version": version,
                    "data_status": status.value,
                    "origin": "backfill",
                    "status": "resolved",
                    "resolved_at": now,
                    "realized_price": realized,
                    "benchmark_realized": b1,
                    "realized_return": ret,
                    "benchmark_return": bench_ret,
                    "outcome_up": realized > spot,
                    "outcome_outperform": (ret > bench_ret) if bench_ret is not None else None,
                    "in_50": q25 <= realized <= q75,
                    "in_90": q05 <= realized <= q95,
                }
            )
    return rows


def model_rows(
    oos: pd.Series,
    fwd: pd.DataFrame,
    bench_fwd: pd.Series,
    close: pd.DataFrame,
    config: am.ModelConfig,
    now: datetime,
    benchmark: str,
    status: DataStatus,
    version: str,
) -> list[dict[str, Any]]:
    """Out-of-sample rankings with probabilities from an expanding (point-in-time) calibration."""
    h = config.horizon
    pred = oos.unstack()
    realized = fwd.reindex(index=pred.index, columns=pred.columns)
    bench = bench_fwd.reindex(pred.index)
    calendar = close.index
    position = {d: i for i, d in enumerate(calendar)}
    closes = close.ffill()
    rows: list[dict[str, Any]] = []
    for d in pred.index[::MODEL_STEP]:
        p = position[d]
        if p + h >= len(calendar):
            break  # the outcome is not known yet
        known = pred.index[[position[x] + h <= p for x in pred.index]]
        if len(known) < MIN_CALIBRATION_DATES:
            continue
        try:
            cal = am.calibrate(pred.loc[known], realized.loc[known], bench.loc[known], config)
        except DomainError:
            continue
        scores = pred.loc[d].dropna()
        sd = float(scores.std(ddof=0))
        if len(scores) < 3 or sd <= 0:
            continue
        z = (scores - scores.mean()) / sd
        order = z.sort_values(ascending=False)
        target = calendar[p + h].date()
        bench_ret = bench.get(d)
        bench_ret = None if bench_ret is None or not math.isfinite(bench_ret) else float(bench_ret)
        for rank, (symbol, zz) in enumerate(order.items(), start=1):
            ret = realized.at[d, symbol]
            ref = closes.at[d, symbol]
            if not (math.isfinite(ret) and math.isfinite(ref) and ref > 0):
                continue
            rows.append(
                {
                    "created_at": now,
                    "made_on": d.date(),
                    "target_date": target,
                    "symbol": str(symbol),
                    "source": "model",
                    "horizon_days": h,
                    "reference_price": float(ref),
                    "benchmark": benchmark,
                    "benchmark_reference": None,
                    "prob_up": None,
                    "prob_outperform": cal.probability(float(zz)),
                    "expected_return": cal.expected_excess(float(zz)),
                    "q05": None,
                    "q25": None,
                    "q50": None,
                    "q75": None,
                    "q95": None,
                    "rank": rank,
                    "model_version": version,
                    "data_status": status.value,
                    "origin": "backfill",
                    "status": "resolved",
                    "resolved_at": now,
                    "realized_price": float(ref) * (1 + float(ret)),
                    "benchmark_realized": None,
                    "realized_return": float(ret),
                    "benchmark_return": bench_ret,
                    "outcome_up": float(ret) > 0,
                    "outcome_outperform": (float(ret) > bench_ret) if bench_ret is not None else None,
                    "in_50": None,
                    "in_90": None,
                }
            )
    return rows


class BackfillService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        clock: Clock,
        market: MarketService,
        rates: RatesService,
        forecast: ForecastService,
        model: ModelService,
        jobs: JobRegistry,
    ) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._market = market
        self._rates = rates
        self._forecast = forecast
        self._model = model
        self._jobs = jobs

    def _allowed(self, status: DataStatus) -> bool:
        return status is not DataStatus.SYNTHETIC or self._settings.predictions_allow_synthetic

    def job(self) -> Job | None:
        return self._jobs.latest(BACKFILL_KEY)

    def start(self, sources: tuple[str, ...] = ("forecast", "model"), *, replace: bool = False) -> Job:
        """Start (or join) the backfill job."""

        async def work(job: Job) -> BackfillResult:
            return await self.run(job, sources, replace=replace)

        return self._jobs.start("backfill", BACKFILL_KEY, "Track-record backfill", work)

    async def run(self, job: Job, sources: tuple[str, ...], *, replace: bool = False) -> BackfillResult:
        result = BackfillResult()
        if replace:
            async with self._db.session() as s:
                for source in sources:
                    result.replaced += await repo.delete_backfill(s, source)
        shares = {"forecast": 0.5, "model": 0.5} if len(sources) == 2 else {sources[0]: 1.0}
        offset = 0.0
        if "forecast" in sources:
            await self._forecasts(job.reporter(offset, offset + shares["forecast"]), result)
            offset += shares["forecast"]
        if "model" in sources:
            await self._models(job.reporter(offset, offset + shares["model"]), result)
        return result

    async def _forecasts(self, report: Any, result: BackfillResult) -> None:
        s = self._settings
        bench = s.benchmark_symbol
        now = self._clock.now()
        universe = list(s.picks_universe)
        report(0.0, "loading histories for the forecast replay")
        bench_r, curve_r = await asyncio.gather(
            self._market.history(bench, "1d", STANDARD_HISTORY_DAYS), self._rates.curve()
        )
        if not self._allowed(bench_r.status):
            result.skipped["forecast"] = "synthetic prices"
            return
        bench_by_day = {b.timestamp.astimezone(NEW_YORK).date(): b.close for b in bench_r.value.bars}
        rf = RatesService.rate_from_curve(curve_r.value, 0.25).continuous_rate
        sem = asyncio.Semaphore(CONCURRENCY)
        done = 0

        async def one(symbol: str) -> None:
            nonlocal done
            async with sem:
                hist_r, div_r = await asyncio.gather(
                    self._market.history(symbol, "1d", STANDARD_HISTORY_DAYS),
                    self._market.dividend_yield(symbol),
                )
                if not self._allowed(hist_r.status):
                    result.skipped[f"forecast:{symbol}"] = "synthetic prices"
                    return
                bars = hist_r.value.bars
                dates = [b.timestamp.astimezone(NEW_YORK).date() for b in bars]
                closes = np.array([b.close for b in bars], dtype=float)
                if closes.size < MIN_OBS + FORECAST_HORIZONS[0] + 1:
                    result.skipped[f"forecast:{symbol}"] = f"only {closes.size} daily closes"
                    return
                earnings = await self._forecast.earnings_inputs(
                    symbol, dates, closes, hist_r.status is not DataStatus.SYNTHETIC, now
                )
                bench_aligned = np.array([bench_by_day.get(d, np.nan) for d in dates], dtype=float)
                beta = rolling_beta(closes, pd.Series(bench_aligned).ffill().to_numpy())
                q = float(div_r.value or 0.0)
                drift = rf + np.nan_to_num(beta, nan=1.0) * s.equity_risk_premium - q
                version = FORECAST_VERSION + (
                    "+earn" if earnings is not None and earnings.sample.size else ""
                )
                rows = await asyncio.to_thread(
                    forecast_rows,
                    symbol,
                    dates,
                    closes,
                    bench_by_day,
                    drift,
                    None if earnings is None else earnings.flags,
                    now,
                    bench,
                    hist_r.status,
                    version,
                )
                async with self._db.session() as sess:
                    inserted = await repo.insert_predictions_bulk(sess, rows)
                result.forecast_rows += inserted  # add after the await: concurrent symbols share the total
                for r in rows[:1] + rows[-1:]:
                    result.span(r["made_on"])
            done += 1
            report(done / len(universe), f"replaying forecasts ({done}/{len(universe)})")

        await asyncio.gather(*(one(sym) for sym in universe))

    async def _models(self, report: Any, result: BackfillResult) -> None:
        report(0.0, "waiting for the stock model's walk-forward run")
        try:
            run = await self._model.history_run(wait=MODEL_WAIT)
        except DomainError as exc:
            result.skipped["model"] = str(exc)
            return
        status = run.report.data_status
        if not self._allowed(status):
            result.skipped["model"] = "synthetic prices"
            return
        if (
            run.oos is None
            or run.fwd is None
            or run.bench_fwd is None
            or run.close is None
            or run.config is None
        ):
            result.skipped["model"] = "the model run has no out-of-sample history"
            return
        report(0.3, "replaying the model's out-of-sample rankings")
        version = f"{run.report.model_type}-{MODEL_VERSION}"
        rows = await asyncio.to_thread(
            model_rows,
            run.oos,
            run.fwd,
            run.bench_fwd,
            run.close,
            run.config,
            self._clock.now(),
            self._settings.benchmark_symbol,
            status,
            version,
        )
        report(0.9, f"storing {len(rows)} ranked predictions")
        async with self._db.session() as sess:
            inserted = await repo.insert_predictions_bulk(sess, rows)
        result.model_rows += inserted
        for r in rows[:1] + rows[-1:]:
            result.span(r["made_on"])
        report(1.0, "done")
