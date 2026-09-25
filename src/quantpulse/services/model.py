"""The stock model lab: walk-forward model reports, signal research and the market regime."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.cache import SingleFlight, TTLCache
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.gateway import Resolved
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.domain import alpha_model as am
from quantpulse.domain import features as feat
from quantpulse.domain import regime as regime_mod
from quantpulse.domain import research as research_mod
from quantpulse.domain import screener
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus, Provenance
from quantpulse.schemas.market import PriceHistory
from quantpulse.schemas.model import (
    BacktestOut,
    CalibrationBinOut,
    ConditionalOut,
    HorizonICOut,
    ICOut,
    ICPoint,
    ImportanceOut,
    LiveScore,
    ModelReport,
    PerfOut,
    RegimeOut,
    ResearchReport,
    SignalOut,
)
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService
from quantpulse.services.rates import RatesService

logger = logging.getLogger(__name__)

MODEL_HISTORY_DAYS = STANDARD_HISTORY_DAYS
REGIME_HISTORY_DAYS = STANDARD_HISTORY_DAYS
MIN_COVERAGE = 0.9
MIN_REAL_SYMBOLS = 10
FETCH_CONCURRENCY = 4
ROLLING_IC = 63


def build_panel(
    histories: dict[str, PriceHistory],
    benchmark: PriceHistory,
    min_coverage: float = MIN_COVERAGE,
    min_symbols: int = 3,
) -> tuple[feat.Panel, dict[str, str]]:
    """Align OHLCV on the benchmark's New York trading dates; drop symbols with too little history."""

    def frame(h: PriceHistory) -> pd.DataFrame:
        idx = (
            pd.DatetimeIndex([b.timestamp for b in h.bars])
            .tz_convert("America/New_York")
            .normalize()
            .tz_localize(None)
        )
        df = pd.DataFrame(
            {
                "open": [b.open for b in h.bars],
                "high": [b.high for b in h.bars],
                "low": [b.low for b in h.bars],
                "close": [b.close for b in h.bars],
                "volume": [b.volume for b in h.bars],
            },
            index=idx,
        )
        return df[~df.index.duplicated(keep="last")].sort_index()

    bench = frame(benchmark)
    if len(bench) < 60:
        raise DomainError(f"benchmark {benchmark.symbol} has only {len(bench)} daily bars")
    dates = bench.index
    skipped: dict[str, str] = {}
    cols: dict[str, pd.DataFrame] = {}
    for symbol, h in histories.items():
        if not h.bars:
            skipped[symbol] = "no price history"
            continue
        df = frame(h).reindex(dates)
        coverage = float(df["close"].notna().mean())
        if coverage < min_coverage:
            skipped[symbol] = f"history covers only {coverage:.0%} of the benchmark's trading days"
            continue
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill(limit=3)
        cols[symbol] = df
    if len(cols) < min_symbols:
        raise DomainError(f"only {len(cols)} symbols have enough history; at least {min_symbols} are needed")
    symbols = sorted(cols)

    def wide(field: str) -> pd.DataFrame:
        return pd.DataFrame({s: cols[s][field] for s in symbols}, index=dates)

    panel = feat.Panel(
        close=wide("close"),
        high=wide("high"),
        low=wide("low"),
        volume=wide("volume"),
        benchmark=bench["close"],
    )
    return panel, skipped


def _ic_out(s: am.ICStats) -> ICOut:
    return ICOut(
        mean_ic=s.mean_ic,
        ic_std=s.ic_std,
        t_stat=s.t_stat,
        positive_share=s.positive_share,
        hit_rate=s.hit_rate,
        n_dates=s.n_dates,
    )


def _perf(m: dict[str, float | None]) -> PerfOut:
    def fin(x: float | None) -> float | None:
        return x if x is not None and math.isfinite(x) else None

    return PerfOut(
        total_return=m["total_return"] or 0.0,
        annual_return=fin(m["annual_return"]),
        annual_volatility=fin(m["annual_volatility"]),
        sharpe=fin(m["sharpe"]),
        max_drawdown=fin(m["max_drawdown"]),
    )


def verdict(oos: am.ICStats, baseline: am.ICStats) -> tuple[str, bool]:
    t = oos.t_stat
    if t is None or oos.n_dates < 60:
        return "Not enough out-of-sample history to judge skill yet; treat the rankings as unproven.", False
    ic = f"{oos.mean_ic:+.3f}"
    if t >= 2 and oos.mean_ic > 0:
        ahead = "ahead of" if oos.mean_ic > baseline.mean_ic else "but not ahead of"
        return (
            f"Evidence of skill: mean out-of-sample IC {ic} (t = {t:.1f}) over {oos.n_dates} days, {ahead} the "
            f"simple factor rule (IC {baseline.mean_ic:+.3f}). Expect the edge to be small and noisy.",
            True,
        )
    if t <= -2:
        return (
            f"The model has been reliably wrong out of sample (IC {ic}, t = {t:.1f}); do not use its rankings.",
            False,
        )
    return (
        f"No reliable evidence of skill: mean out-of-sample IC {ic} (t = {t:.1f}). Rankings are close to random "
        "over this sample, so probabilities stay near the base rate.",
        False,
    )


@dataclass
class _Run:
    report: ModelReport
    sources: dict[str, Provenance]
    live: dict[str, LiveScore]
    coef: np.ndarray
    latest_raw: pd.DataFrame  # symbols x features on the live date (raw values)
    calibration: am.Calibration


class ModelService:
    def __init__(
        self, settings: Settings, clock: Clock, cache: TTLCache, market: MarketService, rates: RatesService
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._cache = cache
        self._market = market
        self._rates = rates
        self._flight = SingleFlight()
        self.default_horizon = 21

    # ------------------------------------------------------------------ data
    async def _histories(self, symbols: Sequence[str], lookback: int) -> dict[str, Resolved[PriceHistory]]:
        sem = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def one(symbol: str) -> Resolved[PriceHistory]:
            async with sem:
                return await self._market.history(symbol, "1d", lookback)

        results = await asyncio.gather(*(one(s) for s in symbols))
        return dict(zip(symbols, results, strict=True))

    async def _panel(
        self, symbols: Sequence[str], lookback: int
    ) -> tuple[feat.Panel, dict[str, str], dict[str, Provenance], DataStatus]:
        bench = self._settings.benchmark_symbol
        wanted = list(dict.fromkeys(symbols))
        loaded = await self._histories(list(dict.fromkeys([*wanted, bench])), lookback)
        skipped: dict[str, str] = {}
        real = [s for s in wanted if loaded[s].status is not DataStatus.SYNTHETIC]
        if loaded[bench].status is not DataStatus.SYNTHETIC and len(real) >= MIN_REAL_SYMBOLS:
            for s in wanted:
                if s not in real:
                    skipped[s] = "no live price history (synthetic data is not mixed with real data)"
            use = real
        else:
            use = wanted
        panel, dropped = build_panel({s: loaded[s].value for s in use if s != bench}, loaded[bench].value)
        skipped.update(dropped)
        used = [*panel.symbols, bench]
        sources = {f"history:{s}": loaded[s].provenance for s in used}
        return panel, skipped, sources, DataStatus.worst([loaded[s].status for s in used])

    def _key(self, kind: str, *parts: object) -> str:
        today = self._clock.now().astimezone(NEW_YORK).date().isoformat()
        return ":".join([kind, today, *(str(p) for p in parts)])

    # ------------------------------------------------------------------ model report
    async def report(
        self,
        horizon: int = 21,
        lookback_days: int = MODEL_HISTORY_DAYS,
        top_k: int = 5,
        symbols: Sequence[str] | None = None,
        *,
        force: bool = False,
    ) -> CompositeEnvelope[ModelReport]:
        run = await self._run(horizon, lookback_days, top_k, symbols, force=force)
        return CompositeEnvelope(
            data=run.report, meta=CompositeMeta.from_sources(run.sources, run.report.computed_at)
        )

    async def live_scores(
        self, symbols: Sequence[str] | None = None
    ) -> tuple[dict[str, LiveScore], ModelReport]:
        run = await self._run(self.default_horizon, MODEL_HISTORY_DAYS, 5, symbols)
        return run.live, run.report

    async def score_symbol(self, symbol: str) -> tuple[LiveScore | None, ModelReport]:
        """Score any symbol with the default-universe model; symbols outside the universe are z-scored
        against the universe's latest cross-section. ``None`` if the symbol lacks a year of history."""
        run = await self._run(self.default_horizon, MODEL_HISTORY_DAYS, 5, None)
        if symbol in run.live:
            return run.live[symbol], run.report
        bench = self._settings.benchmark_symbol
        hist_r, bench_r = await asyncio.gather(
            self._market.history(symbol, "1d", MODEL_HISTORY_DAYS),
            self._market.history(bench, "1d", MODEL_HISTORY_DAYS),
        )
        if run.report.data_status is not DataStatus.SYNTHETIC and hist_r.status is DataStatus.SYNTHETIC:
            return None, run.report  # never score synthetic prices with a model fitted on real ones
        try:
            panel, _ = build_panel({symbol: hist_r.value}, bench_r.value, min_symbols=1)
        except DomainError:
            return None, run.report
        live_day = pd.Timestamp(run.report.as_of)
        raw = feat.compute_features(panel)
        if live_day not in panel.close.index:
            return None, run.report
        row = pd.Series(
            {name: raw[name].loc[live_day, symbol] for name in run.latest_raw.columns}, name=symbol
        )
        if row.notna().mean() < 0.9:
            return None, run.report
        combined = pd.concat([run.latest_raw, row.to_frame().T])
        sd = combined.std(ddof=0).replace(0.0, np.nan)
        z = ((combined - combined.mean()) / sd).clip(-3, 3).fillna(0.0)
        scores = pd.Series(z.to_numpy(dtype=float) @ run.coef, index=z.index)
        universe = scores.drop(symbol)
        spread = float(universe.std(ddof=0))
        zz = float((scores[symbol] - universe.mean()) / spread) if spread > 0 else 0.0
        return (
            LiveScore(
                symbol=symbol,
                rank=int((universe > scores[symbol]).sum()) + 1,
                score=float(scores[symbol]),
                z=zz,
                rating=screener.rating_from_z(zz),
                prob_outperform=run.calibration.probability(zz),
                expected_excess_return=run.calibration.expected_excess(zz),
            ),
            run.report,
        )

    def cached_live(self, symbol: str) -> LiveScore | None:
        """Latest live score for ``symbol`` from a cached default-universe run, without triggering one."""
        entry = self._cache.get(self._run_key(self.default_horizon, MODEL_HISTORY_DAYS, 5, None))
        if entry is None:
            return None
        run: _Run = entry.value
        return run.live.get(symbol)

    def _run_key(self, horizon: int, lookback: int, top_k: int, symbols: Sequence[str] | None) -> str:
        universe = ",".join(sorted(symbols)) if symbols else "default"
        return self._key("model", horizon, lookback, top_k, universe)

    async def _run(
        self, horizon: int, lookback: int, top_k: int, symbols: Sequence[str] | None, *, force: bool = False
    ) -> _Run:
        key = self._run_key(horizon, lookback, top_k, symbols)
        if not force:
            hit = self._cache.get(key)
            if hit is not None:
                return hit.value

        async def compute() -> _Run:
            run = await self._compute(horizon, lookback, top_k, symbols)
            self._cache.set(key, run, self._settings.ttl_model)
            return run

        return await self._flight.run(key, compute)

    async def _compute(self, horizon: int, lookback: int, top_k: int, symbols: Sequence[str] | None) -> _Run:
        universe = list(symbols or self._settings.picks_universe)
        (panel, skipped, sources, status), curve_r = await asyncio.gather(
            self._panel(universe, lookback), self._rates.curve()
        )
        sources["yield_curve"] = curve_r.provenance
        rf = RatesService.rate_from_curve(curve_r.value, 0.25).bey_rate
        config = am.ModelConfig(horizon=horizon, top_k=min(top_k, len(panel.symbols) - 1))

        def work() -> tuple[am.AlphaRun, pd.DataFrame]:
            data, raw = am.build_data(panel, horizon)
            res = am.run(data, config)
            latest = pd.DataFrame({name: raw[name].loc[res.live_date] for name in data.X.columns})
            return res, latest

        result, latest_raw = await asyncio.to_thread(work)
        text, has_skill = verdict(result.oos, result.baseline)
        warnings: list[str] = []
        if status is DataStatus.SYNTHETIC:
            warnings.append(
                "Prices are synthetic (live data unavailable): the report demonstrates the method only."
            )
        if len(panel.symbols) < 15:
            warnings.append(f"Only {len(panel.symbols)} stocks: cross-sectional statistics are very noisy.")

        roll = result.ic_series.rolling(ROLLING_IC, min_periods=20).mean()
        base_roll = (
            result.baseline_ic_series.reindex(result.ic_series.index)
            .rolling(ROLLING_IC, min_periods=20)
            .mean()
        )
        timeline = [
            ICPoint(
                date=d.date(),
                ic=float(result.ic_series[d]),
                rolling=None if pd.isna(roll[d]) else float(roll[d]),
                baseline_rolling=None if pd.isna(base_roll[d]) else float(base_roll[d]),
            )
            for d in result.ic_series.index[::5]
        ]
        bt = result.backtest
        metrics = bt.metrics(252 / horizon, rf)
        live = {
            p.symbol: LiveScore(
                symbol=p.symbol,
                rank=p.rank,
                score=p.score,
                z=p.z,
                rating=screener.rating_from_z(p.z),
                prob_outperform=p.prob_outperform,
                expected_excess_return=p.expected_excess,
            )
            for p in result.live
        }
        importance = sorted(
            (
                ImportanceOut(
                    feature=name,
                    description=feat.FEATURES[name],
                    coefficient=coef,
                    sign_consistency=cons,
                )
                for name, (coef, cons) in result.importance.items()
            ),
            key=lambda i: -abs(i.coefficient),
        )
        report = ModelReport(
            as_of=result.live_date.date(),
            computed_at=self._clock.now(),
            horizon=horizon,
            benchmark=self._settings.benchmark_symbol,
            symbols=panel.symbols,
            skipped=skipped,
            oos_start=result.oos_start.date(),
            oos_end=result.oos_end.date(),
            oos=_ic_out(result.oos),
            baseline=_ic_out(result.baseline),
            verdict=text,
            has_skill=has_skill,
            ic_timeline=timeline,
            buckets=result.buckets,
            backtest=BacktestOut(
                dates=[d.date() for d in bt.dates],
                strategy=bt.strategy,
                universe=bt.universe,
                benchmark=bt.benchmark,
                strategy_metrics=_perf(metrics["strategy"]),
                universe_metrics=_perf(metrics["universe"]),
                benchmark_metrics=_perf(metrics["benchmark"]),
                beat_universe_share=metrics["strategy"].get("beat_universe_share"),
                turnover=bt.turnover,
                periods=len(bt.period_returns),
                cost_bps=config.cost_bps,
            ),
            base_rate=result.calibration.base_rate,
            calibration=[
                CalibrationBinOut(
                    z_mid=b.z_mid,
                    n=b.n,
                    observed=b.observed,
                    probability=b.probability,
                    mean_excess=b.mean_excess,
                )
                for b in result.calibration.bins
            ],
            importance=importance,
            live=sorted(live.values(), key=lambda x: x.rank),
            retrains=len(result.fits),
            lambdas=[lam for _, lam, _ in result.fits],
            data_status=status,
            warnings=warnings,
        )
        return _Run(
            report=report,
            sources=sources,
            live=live,
            coef=result.live_coef,
            latest_raw=latest_raw,
            calibration=result.calibration,
        )

    # ------------------------------------------------------------------ research
    async def research(
        self,
        horizon: int = 21,
        lookback_days: int = MODEL_HISTORY_DAYS,
        symbols: Sequence[str] | None = None,
    ) -> CompositeEnvelope[ResearchReport]:
        universe = list(symbols or self._settings.picks_universe)
        key = self._key("research", horizon, lookback_days, ",".join(sorted(universe)))
        hit = self._cache.get(key)
        if hit is not None:
            return hit.value
        panel, _skipped, sources, status = await self._panel(universe, lookback_days)

        def work() -> research_mod.ResearchResult:
            raw = feat.compute_features(panel)
            return research_mod.factor_research(raw, panel.close, (1, 5, 21, 63), horizon)

        res = await asyncio.to_thread(work)
        corr = res.correlation
        report = ResearchReport(
            computed_at=self._clock.now(),
            start=res.start.date(),
            end=res.end.date(),
            horizons=res.horizons,
            main_horizon=res.main_horizon,
            n_symbols=res.n_symbols,
            signals=[
                SignalOut(
                    feature=f.name,
                    description=f.description,
                    by_horizon=[
                        HorizonICOut(
                            horizon=h.horizon,
                            mean_ic=h.mean_ic,
                            t_stat=h.t_stat,
                            positive_share=h.positive_share,
                            n_dates=h.n_dates,
                        )
                        for h in f.by_horizon
                    ],
                    quintile_returns=f.quintile_returns,
                    spread=f.spread,
                )
                for f in res.features
            ],
            correlation={
                a: {
                    b: (None if pd.isna(corr.loc[a, b]) else round(float(corr.loc[a, b]), 4))
                    for b in corr.columns
                }
                for a in corr.index
            },
            data_status=status,
        )
        env: CompositeEnvelope[ResearchReport] = CompositeEnvelope(
            data=report, meta=CompositeMeta.from_sources(sources, report.computed_at)
        )
        self._cache.set(key, env, self._settings.ttl_model)
        return env

    # ------------------------------------------------------------------ regime
    async def regime(self) -> CompositeEnvelope[RegimeOut]:
        bench = self._settings.benchmark_symbol
        universe = list(self._settings.picks_universe)
        (panel, _skipped, sources, status), curve_r = await asyncio.gather(
            self._panel(universe, REGIME_HISTORY_DAYS), self._rates.curve()
        )
        sources["yield_curve"] = curve_r.provenance
        curve = curve_r.value
        ten = RatesService.rate_from_curve(curve, 10.0).bey_rate
        three_m = RatesService.rate_from_curve(curve, 0.25).bey_rate
        r = regime_mod.market_regime(panel.benchmark, panel.close, ten, three_m)
        now = self._clock.now()
        out = RegimeOut(
            label=r.label,
            benchmark=bench,
            as_of=r.as_of.date(),
            price=r.price,
            above_sma200=r.above_sma200,
            sma50_above_sma200=r.sma50_above_sma200,
            distance_sma200=r.distance_sma200,
            return_3m=r.return_3m,
            drawdown_52w=r.drawdown_52w,
            volatility_21d=r.volatility_21d,
            volatility_percentile=r.volatility_percentile,
            breadth_above_sma200=r.breadth_above_sma200,
            breadth_above_sma50=r.breadth_above_sma50,
            curve_slope_10y_3m=r.curve_slope_10y_3m,
            curve_inverted=r.curve_inverted,
            history=[
                ConditionalOut(
                    state=c.state, n=c.n, mean=c.mean, median=c.median, positive_share=c.positive_share
                )
                for c in r.history
            ],
            notes=r.notes,
            data_status=DataStatus.worst([status, curve_r.status]),
        )
        return CompositeEnvelope(data=out, meta=CompositeMeta.from_sources(sources, now))
