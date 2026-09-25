"""The stock model lab: walk-forward model reports, signal research and the market regime.

Universe (``QP_MODEL_UNIVERSE``)
    ``sp500`` models every stock that belonged to the S&P 500 at any point in the window, each only on
    the dates it was a member (no survivorship bias: stocks that were later removed, acquired or went
    bankrupt stay in the history). ``auto`` uses the S&P 500 when a bulk price vendor (Alpaca) is
    configured and the picks list otherwise; ``picks`` or a ticker list model a fixed set.

Inputs
    Prices come from the warehouse-first daily panel; industries (Fama-French 12 from SEC SIC codes) and
    earnings-release times from SEC filings; value and quality factors from SEC XBRL frames, used only
    after they were published. Real and synthetic data are never mixed.

Runs
    A run for a large universe takes minutes the first time (downloading five years of prices for ~600
    stocks) and about a minute afterwards, so runs execute as background jobs with progress. Callers wait
    up to ``QP_MODEL_SYNC_WAIT_SECONDS``; meanwhile the previous day's run keeps serving rankings, and the
    poller keeps today's run warm.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.cache import TTLCache
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.jobs import Job, JobPending, JobRegistry
from quantpulse.core.market_calendar import last_settled_session
from quantpulse.domain import alpha_model as am
from quantpulse.domain import earnings as earn
from quantpulse.domain import features as feat
from quantpulse.domain import regime as regime_mod
from quantpulse.domain import research as research_mod
from quantpulse.domain import screener
from quantpulse.domain.fundamental_factors import FUNDAMENTAL_FEATURES, CompanyFacts, fundamental_features
from quantpulse.domain.sectors import FF12_NAMES
from quantpulse.domain.universe import Membership
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus, Provenance
from quantpulse.schemas.market import PriceHistory
from quantpulse.schemas.model import (
    BacktestOut,
    CalibrationBinOut,
    ConditionalOut,
    CoverageOut,
    HorizonICOut,
    ICOut,
    ICPoint,
    ImportanceOut,
    LiveScore,
    ModelCompareOut,
    ModelReport,
    PerfOut,
    RegimeOut,
    ResearchReport,
    SignalOut,
    UniverseInfo,
    UniverseOut,
)
from quantpulse.schemas.reference import CompanyProfile
from quantpulse.services.facts import FactsService
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService, history_frame
from quantpulse.services.rates import RatesService
from quantpulse.services.reference import ReferenceService

logger = logging.getLogger(__name__)

MODEL_HISTORY_DAYS = STANDARD_HISTORY_DAYS
REGIME_HISTORY_DAYS = STANDARD_HISTORY_DAYS
MIN_COVERAGE = 0.9
MIN_REAL_SYMBOLS = 10
ROLLING_IC = 63
MIN_BARS = 126  # point-in-time universes: a former member needs about six months of prices
MIN_SECTOR_SHARE = 0.5  # industry-neutral features need industries for at least half the stocks
MISSING_SAMPLE = 40
WARM_RETRY_AFTER = timedelta(minutes=30)  # the poller does not restart a failed run more often than this
Progress = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    return None


class ModelTraining(DomainError):
    """The run is still computing in the background (``job`` has its progress)."""

    def __init__(self, job: Job) -> None:
        super().__init__(f"the stock model is still training ({job.progress:.0%}: {job.stage})")
        self.job = job


def panel_from_frames(
    frames: Mapping[str, pd.DataFrame],
    bench: pd.DataFrame,
    bench_symbol: str,
    min_coverage: float = MIN_COVERAGE,
    min_symbols: int = 3,
    min_bars: int = 0,
) -> tuple[feat.Panel, dict[str, str]]:
    """Align OHLCV on the benchmark's New York trading dates; drop symbols with too little history."""
    if len(bench) < 60:
        raise DomainError(f"benchmark {bench_symbol} has only {len(bench)} daily bars")
    dates = bench.index
    skipped: dict[str, str] = {}
    cols: dict[str, pd.DataFrame] = {}
    for symbol, f in frames.items():
        df = f.reindex(dates)
        n = int(df["close"].notna().sum())
        coverage = n / len(dates)
        if n == 0:
            skipped[symbol] = "no prices inside the window"
            continue
        if coverage < min_coverage:
            skipped[symbol] = f"history covers only {coverage:.0%} of the benchmark's trading days"
            continue
        if n < min_bars:
            skipped[symbol] = f"only {n} daily bars in the window"
            continue
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill(limit=3)
        cols[symbol] = df
    if len(cols) < min_symbols:
        raise DomainError(f"only {len(cols)} symbols have enough history; at least {min_symbols} are needed")
    symbols = sorted(cols)

    def wide(name: str) -> pd.DataFrame:
        return pd.DataFrame({s: cols[s][name] for s in symbols}, index=dates)

    panel = feat.Panel(
        close=wide("close"),
        high=wide("high"),
        low=wide("low"),
        volume=wide("volume"),
        benchmark=bench["close"],
    )
    return panel, skipped


def build_panel(
    histories: dict[str, PriceHistory],
    benchmark: PriceHistory,
    min_coverage: float = MIN_COVERAGE,
    min_symbols: int = 3,
) -> tuple[feat.Panel, dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for symbol, h in histories.items():
        if h.bars:
            frames[symbol] = history_frame(h)
        else:
            skipped[symbol] = "no price history"
    panel, dropped = panel_from_frames(
        frames, history_frame(benchmark), benchmark.symbol, min_coverage, min_symbols
    )
    return panel, {**skipped, **dropped}


def feature_group(name: str) -> str:
    if name in feat.FEATURES:
        return "price"
    if name in feat.EARNINGS_FEATURES:
        return "earnings"
    if name in feat.SECTOR_FEATURES:
        return "sector"
    return "fundamental" if name in FUNDAMENTAL_FEATURES else "other"


def _fin(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


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
class UniverseSpec:
    kind: str  # "sp500", "picks" or "custom"
    label: str
    symbols: list[str]  # fixed lists; the S&P 500 is resolved per window from ``membership``
    membership: Membership | None = None
    membership_provenance: Provenance | None = None

    @property
    def key(self) -> str:
        return self.kind if self.kind != "custom" else ",".join(sorted(self.symbols))

    def members(self, start: datetime, end: datetime) -> list[str]:
        if self.membership is None:
            return list(self.symbols)
        return sorted(self.membership.ever_members(start.date(), end.date()))


@dataclass
class _Inputs:
    spec: UniverseSpec
    panel: feat.Panel
    status: DataStatus
    sources: dict[str, Provenance]
    skipped: dict[str, str]
    missing: dict[str, str]
    eligible: pd.DataFrame | None
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)
    sectors: pd.Series | None = None
    profiles: dict[str, CompanyProfile] = field(default_factory=dict)
    earnings_status: DataStatus | None = None
    earnings_companies: int = 0
    facts_status: DataStatus | None = None
    facts_companies: int = 0
    frames: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)

    def eligible_now(self) -> list[str]:
        if self.eligible is None:
            return list(self.panel.symbols)
        last = self.eligible.iloc[-1]
        return [s for s in self.panel.symbols if bool(last.get(s, True))]


@dataclass
class _Run:
    report: ModelReport
    sources: dict[str, Provenance]
    live: dict[str, LiveScore]
    final: am.FinalModel
    names: list[str]
    latest_raw: pd.DataFrame  # symbols x features on the live date (raw values)
    calibration: am.Calibration
    sectors: pd.Series | None
    neutral: bool
    config: am.ModelConfig | None = None
    oos: pd.Series | None = None  # the live model's out-of-sample predictions, (date, symbol)
    fwd: pd.DataFrame | None = None  # realised forward returns (cash-out after a delisting)
    bench_fwd: pd.Series | None = None
    close: pd.DataFrame | None = None


@dataclass(frozen=True)
class ModelSnapshot:
    live: dict[str, LiveScore]
    features: pd.DataFrame  # symbols × raw features on the model's live date
    as_of: date
    data_status: DataStatus
    label: str


class ModelService:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        cache: TTLCache,
        market: MarketService,
        rates: RatesService,
        reference: ReferenceService,
        facts: FactsService,
        jobs: JobRegistry,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._cache = cache
        self._market = market
        self._rates = rates
        self._reference = reference
        self._facts = facts
        self._jobs = jobs
        self._latest: dict[str, _Run] = {}  # run family (no date) -> most recent completed run
        self._latest_research: dict[str, CompositeEnvelope[ResearchReport]] = {}
        self.default_horizon = 21

    # ------------------------------------------------------------------ universe
    def universe_mode(self) -> str:
        mode = self._settings.model_universe
        if mode == "auto":
            return "sp500" if self._market.has_bulk_history else "picks"
        return mode

    async def universe(self, symbols: Sequence[str] | None = None) -> UniverseSpec:
        if symbols:
            return UniverseSpec("custom", "custom list", list(dict.fromkeys(symbols)))
        mode = self.universe_mode()
        if mode == "picks":
            return UniverseSpec(
                "picks", "picks list (QP_PICKS_UNIVERSE)", list(self._settings.picks_universe)
            )
        if mode == "sp500":
            m = await self._reference.membership()
            return UniverseSpec("sp500", "S&P 500, point-in-time", [], m.value, m.provenance)
        return UniverseSpec("custom", "QP_MODEL_UNIVERSE", mode.split(","))

    async def universe_info(self) -> UniverseInfo:
        spec = await self.universe(None)
        m = spec.membership
        prov = spec.membership_provenance
        return UniverseInfo(
            setting=self._settings.model_universe,
            kind=spec.kind,
            label=spec.label,
            bulk_prices=self._market.has_bulk_history,
            current_members=len(m.current) if m is not None else len(spec.symbols),
            membership_status=prov.status if prov else None,
            membership_as_of=m.as_of if m is not None else None,
            changes_logged=len(m.changes) if m is not None else None,
            history_from=m.log_start if m is not None else None,
        )

    # ------------------------------------------------------------------ inputs
    def _day(self) -> str:
        """Runs are keyed by the last session with final closing prices: the model ranks at the close,
        so a run stays valid from one close until the next."""
        return last_settled_session(self._clock.now()).isoformat()

    async def _inputs(
        self, spec: UniverseSpec, lookback: int, progress: Progress = _noop, *, enrich: bool = True
    ) -> _Inputs:
        key = f"model-inputs:{self._day()}:{spec.key}:{lookback}:{int(enrich)}"
        hit = self._cache.get(key)
        if hit is not None:
            return hit.value
        inputs = await self._load_inputs(spec, lookback, progress, enrich)
        ttl = self._settings.ttl_model if enrich else self._settings.ttl_bars_daily
        self._cache.set(key, inputs, ttl)
        return inputs

    async def _load_inputs(
        self, spec: UniverseSpec, lookback: int, progress: Progress, enrich: bool
    ) -> _Inputs:
        bench = self._settings.benchmark_symbol
        now = self._clock.now()
        start = now - timedelta(days=lookback)
        wanted = [s for s in spec.members(start, now) if s != bench]
        price_share = 0.45 if enrich else 1.0
        prices = await self._market.daily_panel(
            [*wanted, bench], lookback, progress=lambda f, st: progress(price_share * f, st)
        )
        warnings: list[str] = []
        real = [s for s in wanted if s in prices.frames]
        needed = min(MIN_REAL_SYMBOLS, max(3, len(wanted) // 2))
        if self._settings.enable_live_data and (bench not in prices.frames or len(real) < needed):
            warnings.append(
                f"Only {len(real)} symbols have live prices (need {needed} plus the benchmark), so the model "
                "runs on synthetic prices."
            )
            prices = await self._market.daily_panel([*wanted, bench], lookback, synthetic_only=True)
        if bench not in prices.frames:
            raise DomainError(f"no price history for the benchmark {bench}")
        point_in_time = spec.membership is not None
        session = pd.Timestamp(last_settled_session(now))
        bench_frame = prices.frames[bench]
        bench_frame = bench_frame[bench_frame.index <= session]  # never rank on a partial intraday bar
        # Heavy pandas work runs in a worker thread: the API keeps answering while a large run loads.
        panel, skipped = await asyncio.to_thread(
            panel_from_frames,
            {s: f for s, f in prices.frames.items() if s != bench},
            bench_frame,
            bench,
            min_coverage=0.0 if point_in_time else MIN_COVERAGE,
            min_bars=MIN_BARS if point_in_time else 0,
        )
        for s, why in prices.missing.items():
            skipped.setdefault(s, why)
        used = [*panel.symbols, bench]
        status = DataStatus.worst([prices.status(s) for s in used])
        sources = prices.summary()
        if spec.membership_provenance is not None:
            sources["sp500_membership"] = spec.membership_provenance
        membership = spec.membership
        eligible = (
            await asyncio.to_thread(membership.mask, panel.close.index, panel.symbols) if membership else None
        )
        inputs = _Inputs(
            spec=spec,
            panel=panel,
            status=status,
            sources=sources,
            skipped=skipped,
            missing=dict(prices.missing),
            eligible=eligible,
            warnings=warnings,
        )
        if enrich:
            await self._enrich(inputs, progress)
        return inputs

    async def _enrich(self, inputs: _Inputs, progress: Progress) -> None:
        """Industries, earnings reactions and point-in-time fundamentals for the panel's symbols."""
        panel = inputs.panel
        real = inputs.status is not DataStatus.SYNTHETIC
        progress(0.45, "loading industries and earnings dates (SEC)")
        # Synthetic prices get synthetic companies: real filings would not match the simulated paths.
        events = await self._reference.events_many(panel.symbols, simulated=not real)
        usable = {s: r for s, r in events.items() if not (real and r.status is DataStatus.SYNTHETIC)}
        if usable:
            inputs.earnings_status = DataStatus.worst([r.status for r in usable.values()])
            inputs.sources["sec_events"] = next(iter(usable.values())).provenance.model_copy(
                update={"message": f"{len(usable)} companies (industries and earnings releases)"}
            )
        inputs.profiles = {s: r.value.profile for s, r in usable.items()}
        reaction_days = {
            s: [pd.Timestamp(earn.reaction_day(t)) for t in r.value.earnings] for s, r in usable.items()
        }
        inputs.earnings_companies = sum(1 for d in reaction_days.values() if d)
        inputs.extra["earn_reaction"] = await asyncio.to_thread(
            feat.earnings_reaction, panel.close, panel.benchmark, reaction_days
        )
        sectors = pd.Series({s: p.sector for s, p in inputs.profiles.items()}, dtype=object)
        if len(sectors) >= MIN_SECTOR_SHARE * len(panel.symbols):
            inputs.sectors = sectors
        else:
            inputs.warnings.append(
                f"Industries are known for only {len(sectors)} of {len(panel.symbols)} stocks, so features are "
                "compared with the whole market rather than with industry peers."
            )

        progress(0.6, "loading fundamentals (SEC XBRL frames)")
        ciks = {s: int(p.cik) for s, p in inputs.profiles.items() if p.cik.isdigit()}
        first, last = panel.close.index[0].date(), panel.close.index[-1].date()
        facts: dict[str, CompanyFacts] = {}
        if ciks:
            facts, fcov = await self._facts.company_facts(
                ciks,
                first,
                last,
                progress=lambda f: progress(0.6 + 0.12 * f, "loading fundamentals (SEC XBRL frames)"),
                simulated=not real,
            )
            if real and fcov.status is DataStatus.SYNTHETIC:
                facts = {}
            else:
                inputs.facts_status = fcov.status
                inputs.facts_companies = fcov.companies_with_facts
                inputs.frames = (fcov.frames_available, fcov.frames_requested)
                now = self._clock.now()
                inputs.sources["sec_fundamentals"] = Provenance(
                    status=fcov.status,
                    provider="sec_edgar" if fcov.status is not DataStatus.SYNTHETIC else "synthetic",
                    as_of=now,
                    fetched_at=now,
                    message=(
                        f"{fcov.companies_with_facts} companies, {fcov.frames_available}/{fcov.frames_requested} "
                        "XBRL frames"
                        + (f"; missing {', '.join(fcov.frames_missing[:5])}" if fcov.frames_missing else "")
                    ),
                )
        if facts:
            inputs.extra.update(await asyncio.to_thread(fundamental_features, facts, panel.close))
        else:
            inputs.warnings.append(
                "No fundamentals are available, so value and quality factors are left out."
            )

    # ------------------------------------------------------------------ runs
    def _family(self, horizon: int, lookback: int, top_k: int, spec_key: str) -> str:
        s = self._settings
        return ":".join(
            str(p)
            for p in ("model", horizon, lookback, top_k, spec_key, s.model_type, int(s.model_sector_neutral))
        )

    def _run_key(self, horizon: int, lookback: int, top_k: int, spec_key: str) -> str:
        return f"{self._family(horizon, lookback, top_k, spec_key)}:{self._day()}"

    def _spec_key(self, symbols: Sequence[str] | None) -> str:
        if symbols:
            return ",".join(sorted(dict.fromkeys(symbols)))
        mode = self.universe_mode()
        return mode if mode in ("sp500", "picks") else ",".join(sorted(mode.split(",")))

    def model_job(self) -> Job | None:
        """The latest job for the default model (for progress displays)."""
        key = self._run_key(self.default_horizon, MODEL_HISTORY_DAYS, 5, self._spec_key(None))
        return self._jobs.latest(key)

    async def _run(
        self,
        horizon: int,
        lookback: int,
        top_k: int,
        symbols: Sequence[str] | None,
        *,
        force: bool = False,
        wait: float | None = None,
        allow_stale: bool = True,
    ) -> _Run:
        spec_key = self._spec_key(symbols)
        key = self._run_key(horizon, lookback, top_k, spec_key)
        family = self._family(horizon, lookback, top_k, spec_key)
        if not force:
            hit = self._cache.get(key)
            if hit is not None:
                return hit.value

        async def work(job: Job) -> _Run:
            spec = await self.universe(symbols)
            run = await self._compute(job, spec, horizon, lookback, top_k)
            self._cache.set(key, run, self._settings.ttl_model)
            self._latest[family] = run
            return run

        job = self._jobs.start("model", key, f"Stock model ({spec_key}, {horizon}-day horizon)", work)
        timeout = self._settings.model_sync_wait_seconds if wait is None else wait
        try:
            return await self._jobs.wait(job, timeout)
        except JobPending:
            stale = self._latest.get(family)
            if allow_stale and stale is not None:
                return stale
            raise ModelTraining(job) from None

    async def _compute(self, job: Job, spec: UniverseSpec, horizon: int, lookback: int, top_k: int) -> _Run:
        inputs = await self._inputs(spec, lookback, job.reporter(0.0, 0.72))
        curve_r = await self._rates.curve()
        sources = dict(inputs.sources)
        sources["yield_curve"] = curve_r.provenance
        rf = RatesService.rate_from_curve(curve_r.value, 0.25).bey_rate
        neutral = self._settings.model_sector_neutral and inputs.sectors is not None
        panel = inputs.panel
        n_live = len(inputs.eligible_now())
        config = am.ModelConfig(
            horizon=horizon, top_k=max(1, min(top_k, n_live - 1)), model_type=self._settings.model_type
        )
        job.update(0.72, "computing features")
        train = job.reporter(0.78, 0.98)

        def work() -> tuple[am.ModelData, am.AlphaRun, dict[str, pd.DataFrame]]:
            data, raw = am.build_data(
                panel,
                horizon,
                extra=inputs.extra,
                sectors=inputs.sectors,
                eligible=inputs.eligible,
                neutral=neutral,
            )
            return data, am.run(data, config, train), raw

        data, result, raw = await asyncio.to_thread(work)
        job.update(0.98, "writing the report")
        names = data.feature_names
        latest_raw = pd.DataFrame({name: raw[name].loc[result.live_date] for name in names})
        report, live = self._report(inputs, config, data, result, names, neutral, rf)
        return _Run(
            report=report,
            sources=sources,
            live=live,
            final=result.final,
            names=names,
            latest_raw=latest_raw,
            calibration=result.calibration,
            sectors=inputs.sectors,
            neutral=neutral,
            config=config,
            oos=result.models[config.model_type].predictions,
            fwd=data.fwd,
            bench_fwd=data.bench_fwd,
            close=panel.close,
        )

    def _report(
        self,
        inputs: _Inputs,
        config: am.ModelConfig,
        data: am.ModelData,
        result: am.AlphaRun,
        names: list[str],
        neutral: bool,
        rf: float,
    ) -> tuple[ModelReport, dict[str, LiveScore]]:
        horizon = config.horizon
        text, has_skill = verdict(result.oos, result.baseline)
        warnings = list(inputs.warnings)
        if inputs.status is DataStatus.SYNTHETIC:
            warnings.append(
                "Prices are synthetic (live data unavailable): the report demonstrates the method only."
            )
        if len(result.live) < 15:
            warnings.append(f"Only {len(result.live)} stocks: cross-sectional statistics are very noisy.")
        if inputs.spec.membership is None:
            warnings.append(
                "The universe is a fixed list of today's tickers chosen with hindsight, so backtest returns are "
                "flattered by survivorship bias. The S&P 500 universe (QP_MODEL_UNIVERSE=sp500) avoids this."
            )

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
        profiles = inputs.profiles
        live = {
            p.symbol: LiveScore(
                symbol=p.symbol,
                rank=p.rank,
                score=p.score,
                z=p.z,
                rating=screener.rating_from_z(p.z),
                prob_outperform=p.prob_outperform,
                expected_excess_return=p.expected_excess,
                sector=profiles[p.symbol].sector if p.symbol in profiles else None,
                sector_label=profiles[p.symbol].sector_label if p.symbol in profiles else None,
            )
            for p in result.live
        }
        descriptions = feat.all_features()
        importance = sorted(
            (
                ImportanceOut(
                    feature=name,
                    description=descriptions.get(name, name),
                    group=feature_group(name),
                    coefficient=result.importance.get(name, (0.0, 0.0))[0],
                    sign_consistency=result.importance.get(name, (0.0, 0.0))[1],
                    tree_importance=result.tree_importance.get(name),
                )
                for name in names
            ),
            key=lambda i: -(abs(i.coefficient) + max(i.tree_importance or 0.0, 0.0)),
        )
        comparison = []
        for name in ("ensemble", "ridge", "gbm", "baseline"):
            ev = result.models.get(name)
            if ev is None:
                continue
            m = ev.backtest.metrics(252 / horizon, rf)["strategy"]
            comparison.append(
                ModelCompareOut(
                    name=name,
                    label=am.MODEL_LABELS[name],
                    chosen=name == config.model_type,
                    mean_ic=ev.oos.mean_ic,
                    t_stat=ev.oos.t_stat,
                    hit_rate=ev.oos.hit_rate,
                    within_sector_ic=ev.within_sector.mean_ic if ev.within_sector else None,
                    spread=ev.spread,
                    annual_return=_fin(m["annual_return"]),
                    sharpe=_fin(m["sharpe"]),
                    refits=ev.refits,
                )
            )
        spec = inputs.spec
        now_members = set(inputs.eligible_now())
        former = len([s for s in inputs.panel.symbols if s not in now_members]) if spec.membership else None
        missing = {**inputs.missing, **{s: w for s, w in inputs.skipped.items() if s not in inputs.missing}}
        universe = UniverseOut(
            kind=spec.kind,
            label=spec.label,
            point_in_time=spec.membership is not None,
            requested=len(inputs.panel.symbols) + len(missing),
            with_prices=len(inputs.panel.symbols),
            current_members=len(now_members) if spec.membership else None,
            former_members=former,
            missing=dict(sorted(missing.items())[:MISSING_SAMPLE]),
            missing_count=len(missing),
            membership_status=spec.membership_provenance.status if spec.membership_provenance else None,
            note=(
                "Each stock is ranked, trained on and scored only on the dates it was in the S&P 500; former "
                "members keep their history and a delisted stock is cashed out at its last price."
                if spec.membership is not None
                else "A fixed list of current tickers: companies that failed or left are missing from the history."
            ),
        )
        sector_counts: dict[str, int] = {}
        for s in live:
            label = FF12_NAMES.get(profiles[s].sector, profiles[s].sector) if s in profiles else "Unknown"
            sector_counts[label] = sector_counts.get(label, 0) + 1
        coverage = CoverageOut(
            sectors=dict(sorted(sector_counts.items(), key=lambda kv: -kv[1])),
            sector_neutral=neutral,
            earnings_companies=inputs.earnings_companies,
            earnings_status=inputs.earnings_status,
            fundamentals_companies=inputs.facts_companies,
            fundamentals_status=inputs.facts_status,
            frames_available=inputs.frames[0],
            frames_requested=inputs.frames[1],
            features=names,
        )
        ridge_fits = result.fits.get("ridge", [])
        gbm_fits = result.fits.get("gbm", [])
        report = ModelReport(
            as_of=result.live_date.date(),
            computed_at=self._clock.now(),
            horizon=horizon,
            benchmark=self._settings.benchmark_symbol,
            symbols=sorted(live),
            skipped=inputs.skipped,
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
            retrains=len(ridge_fits) + len(gbm_fits),
            lambdas=[f.params[0] for _, f in ridge_fits],
            model_type=config.model_type,
            model_label=am.MODEL_LABELS[config.model_type],
            comparison=comparison,
            within_sector=_ic_out(result.within_sector) if result.within_sector else None,
            tree_sizes=[f.label for _, f in gbm_fits],
            universe=universe,
            coverage=coverage,
            data_status=inputs.status,
            warnings=warnings,
        )
        return report, live

    # ------------------------------------------------------------------ public API
    async def report(
        self,
        horizon: int = 21,
        lookback_days: int = MODEL_HISTORY_DAYS,
        top_k: int = 5,
        symbols: Sequence[str] | None = None,
        *,
        force: bool = False,
        wait: float | None = None,
    ) -> CompositeEnvelope[ModelReport]:
        run = await self._run(
            horizon, lookback_days, top_k, symbols, force=force, wait=wait, allow_stale=not force
        )
        return CompositeEnvelope(
            data=run.report, meta=CompositeMeta.from_sources(run.sources, run.report.computed_at)
        )

    async def live_scores(
        self, symbols: Sequence[str] | None = None, *, wait: float | None = None
    ) -> tuple[dict[str, LiveScore], ModelReport]:
        run = await self._run(self.default_horizon, MODEL_HISTORY_DAYS, 5, symbols, wait=wait)
        return run.live, run.report

    async def trading_snapshot(self, *, wait: float | None = None) -> ModelSnapshot:
        """The default run's live scores and the raw features behind them (fundamentals, earnings reaction)
        for the trading strategy. The last completed run is used while a new one trains."""
        run = await self._run(self.default_horizon, MODEL_HISTORY_DAYS, 5, None, wait=wait)
        return ModelSnapshot(
            live=run.live,
            features=run.latest_raw,
            as_of=run.report.as_of,
            data_status=run.report.data_status,
            label=run.report.model_label,
        )

    async def warm(self) -> str:
        """Start today's default run in the background unless it is cached or already running."""
        key = self._run_key(self.default_horizon, MODEL_HISTORY_DAYS, 5, self._spec_key(None))
        if self._cache.get(key) is not None:
            return "cached"
        running = self._jobs.running(key)
        if running is not None:
            return f"running ({running.progress:.0%})"
        last = self._jobs.latest(key)
        if (
            last is not None
            and last.error is not None
            and last.finished_at is not None
            and self._clock.now() - last.finished_at < WARM_RETRY_AFTER
        ):
            return f"last run failed ({last.error}); retrying later"
        with contextlib.suppress(DomainError):
            await self._run(self.default_horizon, MODEL_HISTORY_DAYS, 5, None, wait=0)
        return "started"

    async def score_symbol(
        self, symbol: str, *, wait: float | None = None
    ) -> tuple[LiveScore | None, ModelReport]:
        """Score any symbol with the default-universe model; symbols outside the universe are scored
        within the universe's latest cross-section. ``None`` if the symbol lacks a year of usable history."""
        run = await self._run(self.default_horizon, MODEL_HISTORY_DAYS, 5, None, wait=wait)
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
        if live_day not in panel.close.index:
            return None, run.report
        row, sector = await self._outside_row(symbol, panel, live_day, run)
        price_names = [n for n in run.names if n in feat.FEATURES]
        if pd.Series({n: row.get(n) for n in price_names}, dtype=float).notna().mean() < MIN_COVERAGE:
            return None, run.report
        combined = pd.concat([run.latest_raw, pd.DataFrame([row], index=[symbol])])
        sectors = run.sectors
        if sectors is not None and sector is not None:
            sectors = pd.concat([sectors, pd.Series({symbol: sector})])
        wides = {
            n: pd.DataFrame([combined[n].to_numpy(dtype=float)], index=[live_day], columns=combined.index)
            for n in run.names
        }
        prepared = am.prepare_features(wides, sectors, run.neutral)
        X = feat.feature_matrix(
            prepared, run.names, MIN_COVERAGE, coverage_names=price_names, dtype=np.float32
        )
        syms = X.index.get_level_values("symbol")
        if symbol not in syms:
            return None, run.report
        scores = pd.Series(run.final.score(X.to_numpy(dtype=np.float32)), index=syms)
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
                sector=sector,
                sector_label=FF12_NAMES.get(sector) if sector else None,
            ),
            run.report,
        )

    async def _outside_row(
        self, symbol: str, panel: feat.Panel, live_day: pd.Timestamp, run: _Run
    ) -> tuple[dict[str, float], str | None]:
        """Raw features for a symbol outside the universe, on the model's live date."""
        raw = feat.compute_features(panel)
        row: dict[str, float] = {n: float(raw[n].loc[live_day, symbol]) for n in run.names if n in raw}
        real = run.report.data_status is not DataStatus.SYNTHETIC
        ev = await self._reference.events(symbol)
        sector: str | None = None
        if not (real and ev.status is DataStatus.SYNTHETIC):
            sector = ev.value.profile.sector
            days = {symbol: [pd.Timestamp(earn.reaction_day(t)) for t in ev.value.earnings]}
            if "earn_reaction" in run.names:
                row["earn_reaction"] = float(
                    feat.earnings_reaction(panel.close, panel.benchmark, days).loc[live_day, symbol]
                )
            cik = ev.value.profile.cik
            if cik.isdigit() and any(n in FUNDAMENTAL_FEATURES for n in run.names):
                first, last = panel.close.index[0].date(), panel.close.index[-1].date()
                facts, fcov = await self._facts.company_facts({symbol: int(cik)}, first, last)
                if facts and not (real and fcov.status is DataStatus.SYNTHETIC):
                    fund = fundamental_features(facts, panel.close)
                    for n in FUNDAMENTAL_FEATURES:
                        if n in run.names:
                            row[n] = float(fund[n].loc[live_day, symbol])
        if run.sectors is not None and sector is not None:
            peers = [s for s, g in run.sectors.items() if g == sector and s in run.latest_raw.index]
            for n in feat.SECTOR_FEATURES:
                if n in run.names and peers:
                    vals = run.latest_raw.loc[peers, n].dropna()
                    row[n] = float(vals.iloc[0]) if len(vals) else float("nan")
        return row, sector

    async def history_run(self, *, wait: float | None = None) -> _Run:
        """The default model run, including its out-of-sample predictions (for the ledger backfill)."""
        return await self._run(
            self.default_horizon, MODEL_HISTORY_DAYS, 5, None, wait=wait, allow_stale=False
        )

    def cached_live(self, symbol: str) -> LiveScore | None:
        """Latest live score for ``symbol`` from a completed default-universe run, without triggering one."""
        spec_key = self._spec_key(None)
        entry = self._cache.get(self._run_key(self.default_horizon, MODEL_HISTORY_DAYS, 5, spec_key))
        run: _Run | None = entry.value if entry is not None else None
        if run is None:
            run = self._latest.get(self._family(self.default_horizon, MODEL_HISTORY_DAYS, 5, spec_key))
        return run.live.get(symbol) if run is not None else None

    # ------------------------------------------------------------------ research
    async def research(
        self,
        horizon: int = 21,
        lookback_days: int = MODEL_HISTORY_DAYS,
        symbols: Sequence[str] | None = None,
        *,
        wait: float | None = None,
    ) -> CompositeEnvelope[ResearchReport]:
        spec_key = self._spec_key(symbols)
        family = f"research:{horizon}:{lookback_days}:{spec_key}:{int(self._settings.model_sector_neutral)}"
        key = f"{family}:{self._day()}"
        hit = self._cache.get(key)
        if hit is not None:
            return hit.value

        async def work(job: Job) -> CompositeEnvelope[ResearchReport]:
            spec = await self.universe(symbols)
            inputs = await self._inputs(spec, lookback_days, job.reporter(0.0, 0.8))
            env = await self._research(inputs, horizon)
            self._cache.set(key, env, self._settings.ttl_model)
            self._latest_research[family] = env
            return env

        job = self._jobs.start("research", key, f"Signal research ({spec_key})", work)
        timeout = self._settings.model_sync_wait_seconds if wait is None else wait
        try:
            result: CompositeEnvelope[ResearchReport] = await self._jobs.wait(job, timeout)
            return result
        except JobPending:
            stale_research = self._latest_research.get(family)
            if stale_research is not None:
                return stale_research
            raise ModelTraining(job) from None

    async def _research(self, inputs: _Inputs, horizon: int) -> CompositeEnvelope[ResearchReport]:
        panel = inputs.panel
        neutral = self._settings.model_sector_neutral and inputs.sectors is not None

        def work() -> research_mod.ResearchResult:
            raw, _ = am.raw_features(
                panel, extra=inputs.extra, sectors=inputs.sectors, eligible=inputs.eligible
            )
            names = am.default_names(raw)
            prepared = am.prepare_features({n: raw[n] for n in names}, inputs.sectors, neutral)
            return research_mod.factor_research(prepared, panel.close.ffill(), (1, 5, 21, 63), horizon, names)

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
            data_status=inputs.status,
        )
        return CompositeEnvelope(
            data=report, meta=CompositeMeta.from_sources(inputs.sources, report.computed_at)
        )

    # ------------------------------------------------------------------ regime
    async def regime(self) -> CompositeEnvelope[RegimeOut]:
        """Trend, volatility, breadth and the yield curve. Breadth uses the model universe's current members
        when their prices are already loaded, and the picks list otherwise (never a slow download)."""
        bench = self._settings.benchmark_symbol
        spec_key = self._spec_key(None)
        inputs: _Inputs | None = None
        for enrich in (1, 0):
            hit = self._cache.get(f"model-inputs:{self._day()}:{spec_key}:{REGIME_HISTORY_DAYS}:{enrich}")
            if hit is not None:
                inputs = hit.value
                break
        if inputs is None:
            spec = (
                UniverseSpec("picks", "picks list", list(self._settings.picks_universe))
                if self.universe_mode() == "sp500"
                else await self.universe(None)
            )
            inputs = await self._inputs(spec, REGIME_HISTORY_DAYS, enrich=False)
        curve_r = await self._rates.curve()
        panel, status = inputs.panel, inputs.status
        sources = dict(inputs.sources)
        sources["yield_curve"] = curve_r.provenance
        curve = curve_r.value
        ten = RatesService.rate_from_curve(curve, 10.0).bey_rate
        three_m = RatesService.rate_from_curve(curve, 0.25).bey_rate
        members = inputs.eligible_now()
        r = regime_mod.market_regime(panel.benchmark, panel.close[members], ten, three_m)
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
            notes=[*r.notes, f"Breadth is measured on {len(members)} stocks ({inputs.spec.label})."],
            data_status=DataStatus.worst([status, curve_r.status]),
        )
        return CompositeEnvelope(data=out, meta=CompositeMeta.from_sources(sources, now))
