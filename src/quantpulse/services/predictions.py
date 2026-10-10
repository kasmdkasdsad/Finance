"""The prediction ledger: log forecasts after each close, grade them on their target date, keep score.

What is logged (once per trading day, after ``QP_PREDICTIONS_LOG_TIME``, from live data only)
    * ``forecast``: for every stock in the universe, the 5- and 21-day price distribution (P(up), expected
      return and the 5/25/50/75/95% price quantiles), anchored on that day's official close;
    * ``model``: the stock model's rank, calibrated probability of beating the benchmark over 21 days,
      and expected excess return.

Grading
    On the target date's close each prediction is marked with the realised price, whether the stock rose,
    whether it beat the benchmark, and whether it landed inside the 50% / 90% bands. Predictions whose
    target-date close never becomes available (e.g. a delisting) are voided after 10 days.

Scorecard
    Brier score (mean squared error of the probabilities; 0.25 = coin flip), Brier skill against always
    predicting the observed base rate, hit rate, band coverage, a reliability table, and for the model the
    average excess return of its top-5 names against the rest.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.market_calendar import (
    NEW_YORK,
    is_trading_day,
    previous_trading_day,
    regular_close,
    sessions_after,
)
from quantpulse.db import repositories as repo
from quantpulse.db.models import PredictionRow
from quantpulse.db.session import Database
from quantpulse.schemas.common import DataStatus
from quantpulse.schemas.forecast import StockForecast
from quantpulse.schemas.market import PriceHistory
from quantpulse.schemas.predictions import (
    CalibrationBucket,
    LedgerCounts,
    LogResult,
    PredictionOut,
    ResolveResult,
    Scorecard,
    SourceScore,
)
from quantpulse.services.forecast import ForecastService
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService
from quantpulse.services.model import ModelService

logger = logging.getLogger(__name__)

FORECAST_VERSION = "garch-t-fhs/2"  # + "+earn" (earnings jumps) and "+iv" (options blend) when used
MODEL_VERSION = "walk-forward/2"  # stored as "<model type>-walk-forward/2"
LEDGER_MODEL_WAIT = 1800.0  # the ledger runs in the background and waits for today's model run
FORECAST_HORIZONS = (5, 21)
VOID_AFTER_DAYS = 10
RESOLVE_EVERY = timedelta(minutes=15)
CONCURRENCY = 4
MAX_FAILURES = 3


def last_completed_session(now: datetime) -> date:
    """The most recent trading day whose regular session has closed."""
    local = now.astimezone(NEW_YORK)
    day = local.date()
    if is_trading_day(day) and local.time() >= regular_close(day):
        return day
    return previous_trading_day(day)


def forecast_version(f: StockForecast) -> str:
    """Which forecaster produced ``f``: the base model plus the extras it actually used."""
    extras = ("+earn" if f.earnings is not None and f.earnings.events_used else "") + (
        "+iv" if f.volatility.iv_weight > 0 else ""
    )
    return FORECAST_VERSION + extras


def _close_on(history: PriceHistory, day: date) -> float | None:
    for bar in reversed(history.bars):
        d = bar.timestamp.astimezone(NEW_YORK).date()
        if d == day:
            return bar.close
        if d < day:
            return None
    return None


def to_out(row: PredictionRow) -> PredictionOut:
    return PredictionOut(
        id=row.id,
        created_at=row.created_at,
        made_on=row.made_on,
        target_date=row.target_date,
        symbol=row.symbol,
        source=row.source,
        horizon_days=row.horizon_days,
        reference_price=row.reference_price,
        benchmark=row.benchmark,
        prob_up=row.prob_up,
        prob_outperform=row.prob_outperform,
        expected_return=row.expected_return,
        q05=row.q05,
        q25=row.q25,
        q50=row.q50,
        q75=row.q75,
        q95=row.q95,
        rank=row.rank,
        model_version=row.model_version,
        data_status=DataStatus(row.data_status),
        origin="backfill" if row.origin == "backfill" else "live",
        status=row.status,
        resolved_at=row.resolved_at,
        realized_price=row.realized_price,
        realized_return=row.realized_return,
        benchmark_return=row.benchmark_return,
        outcome_up=row.outcome_up,
        outcome_outperform=row.outcome_outperform,
        in_50=row.in_50,
        in_90=row.in_90,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _reliability(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> list[CalibrationBucket]:
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inside = [(p, o) for p, o in pairs if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        out.append(
            CalibrationBucket(
                lower=lo,
                upper=hi,
                n=len(inside),
                mean_predicted=_mean([p for p, _ in inside]),
                observed=_mean([float(o) for _, o in inside]),
            )
        )
    return out


def score_probabilities(pairs: Sequence[tuple[float, bool]]) -> dict[str, float | None]:
    if not pairs:
        return {
            "brier": None,
            "brier_base_rate": None,
            "brier_skill": None,
            "hit_rate": None,
            "base_rate": None,
        }
    brier = sum((p - o) ** 2 for p, o in pairs) / len(pairs)
    base = sum(o for _, o in pairs) / len(pairs)
    ref = base * (1 - base)
    called = [(p, o) for p, o in pairs if p != 0.5]
    return {
        "brier": brier,
        "brier_base_rate": ref,
        "brier_skill": None if ref == 0 else 1 - brier / ref,
        "hit_rate": _mean([float((p > 0.5) == o) for p, o in called]),
        "base_rate": base,
    }


class PredictionService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        clock: Clock,
        market: MarketService,
        forecast: ForecastService,
        model: ModelService,
    ) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._market = market
        self._forecast = forecast
        self._model = model
        self._logged_on: date | None = None
        self._last_resolve: datetime | None = None
        self._failures: dict[date, int] = {}

    def _allowed(self, status: DataStatus) -> bool:
        return status is not DataStatus.SYNTHETIC or self._settings.predictions_allow_synthetic

    # ------------------------------------------------------------------ logging
    async def log_daily(self) -> LogResult:
        """Log today's predictions, anchored on the latest completed session's close."""
        now = self._clock.now()
        local = now.astimezone(NEW_YORK)
        if is_trading_day(local.date()) and time(9, 30) <= local.time() < regular_close(local.date()):
            raise DomainError(
                "predictions are logged from official closing prices; run this after the 4 pm ET close"
            )
        made_on = last_completed_session(now)
        async with self._db.session() as session:
            if await repo.predictions_logged_on(session, made_on):
                return LogResult(
                    made_on=made_on,
                    logged=0,
                    skipped={"ledger": f"predictions for {made_on} were already logged"},
                    data_status=DataStatus.LIVE,
                )
        s = self._settings
        bench = s.benchmark_symbol
        universe = list(s.picks_universe)
        skipped: dict[str, str] = {}
        rows: list[dict[str, Any]] = []
        statuses: list[DataStatus] = []

        bench_hist = await self._market.history(bench, "1d", STANDARD_HISTORY_DAYS)
        bench_close = _close_on(bench_hist.value, made_on) if self._allowed(bench_hist.status) else None

        sem = asyncio.Semaphore(CONCURRENCY)

        async def forecast_one(symbol: str) -> None:
            async with sem:
                try:
                    env = await self._forecast.forecast(
                        symbol, FORECAST_HORIZONS, include_options=True, use_close=True
                    )
                    hist = await self._market.history(symbol, "1d", STANDARD_HISTORY_DAYS)
                except DomainError as exc:
                    skipped[f"forecast:{symbol}"] = str(exc)
                    return
            f = env.data
            if not self._allowed(f.data_status):
                skipped[f"forecast:{symbol}"] = "synthetic prices"
                return
            close = _close_on(hist.value, made_on)
            if close is None or abs(close - f.spot) > 1e-6 * max(1.0, close):
                skipped[f"forecast:{symbol}"] = f"no official close for {made_on} yet"
                return
            statuses.append(f.data_status)
            for h in f.horizons:
                rows.append(
                    {
                        "created_at": now,
                        "made_on": made_on,
                        "target_date": sessions_after(made_on, h.days),
                        "symbol": symbol,
                        "source": "forecast",
                        "horizon_days": h.days,
                        "reference_price": f.spot,
                        "benchmark": bench,
                        "benchmark_reference": bench_close,
                        "prob_up": h.prob_up,
                        "prob_outperform": None,
                        "expected_return": h.expected_return,
                        "q05": h.band.p05,
                        "q25": h.band.p25,
                        "q50": h.band.p50,
                        "q75": h.band.p75,
                        "q95": h.band.p95,
                        "rank": None,
                        "model_version": forecast_version(f),
                        "data_status": f.data_status.value,
                        "status": "open",
                    }
                )

        await asyncio.gather(*(forecast_one(sym) for sym in universe))

        all_synthetic = not statuses and all(v == "synthetic prices" for v in skipped.values())
        if all_synthetic:
            skipped["model"] = "synthetic prices"  # same price data; do not spend a model run to discard it
        else:
            rows.extend(await self._model_rows(made_on, now, bench_close, rows, skipped, statuses))
        async with self._db.session() as session:
            inserted = await repo.insert_predictions(session, rows)
        if inserted or rows:
            self._logged_on = made_on
        return LogResult(
            made_on=made_on,
            logged=inserted,
            skipped=skipped,
            data_status=DataStatus.worst(statuses) if statuses else DataStatus.SYNTHETIC,
        )

    async def _model_rows(
        self,
        made_on: date,
        now: datetime,
        bench_close: float | None,
        forecast_rows: list[dict[str, Any]],
        skipped: dict[str, str],
        statuses: list[DataStatus],
    ) -> list[dict[str, Any]]:
        try:
            live, report = await self._model.live_scores(wait=LEDGER_MODEL_WAIT)
        except DomainError as exc:
            skipped["model"] = str(exc)
            return []
        if not self._allowed(report.data_status):
            skipped["model"] = "synthetic prices"
            return []
        if report.as_of != made_on:
            skipped["model"] = f"model data ends {report.as_of}, not {made_on}"
            return []
        statuses.append(report.data_status)
        closes = {r["symbol"]: r["reference_price"] for r in forecast_rows}
        out: list[dict[str, Any]] = []
        for score in live.values():
            ref = closes.get(score.symbol)
            if ref is None:
                hist = await self._market.history(score.symbol, "1d", STANDARD_HISTORY_DAYS)
                ref = _close_on(hist.value, made_on)
            if ref is None:
                skipped[f"model:{score.symbol}"] = f"no official close for {made_on}"
                continue
            out.append(
                {
                    "created_at": now,
                    "made_on": made_on,
                    "target_date": sessions_after(made_on, report.horizon),
                    "symbol": score.symbol,
                    "source": "model",
                    "horizon_days": report.horizon,
                    "reference_price": ref,
                    "benchmark": self._settings.benchmark_symbol,
                    "benchmark_reference": bench_close,
                    "prob_up": None,
                    "prob_outperform": score.prob_outperform,
                    "expected_return": score.expected_excess_return,
                    "q05": None,
                    "q25": None,
                    "q50": None,
                    "q75": None,
                    "q95": None,
                    "rank": score.rank,
                    "model_version": f"{report.model_type}-{MODEL_VERSION}",
                    "data_status": report.data_status.value,
                    "status": "open",
                }
            )
        return out

    # ------------------------------------------------------------------ grading
    async def resolve_due(self) -> ResolveResult:
        now = self._clock.now()
        self._last_resolve = now
        cutoff = last_completed_session(now)
        async with self._db.session() as session:
            due = await repo.due_predictions(session, cutoff)
        if not due:
            return ResolveResult(resolved=0, voided=0, pending=0)
        today = now.astimezone(NEW_YORK).date()
        oldest = min(p.made_on for p in due)
        lookback = max(STANDARD_HISTORY_DAYS, (today - oldest).days + 15)
        symbols = sorted({p.symbol for p in due} | {p.benchmark for p in due})
        sem = asyncio.Semaphore(CONCURRENCY)

        async def load(symbol: str) -> tuple[str, PriceHistory | None]:
            async with sem:
                res = await self._market.history(symbol, "1d", lookback)
            return symbol, res.value if self._allowed(res.status) else None

        histories = dict(await asyncio.gather(*(load(s) for s in symbols)))
        resolved = voided = pending = 0
        async with self._db.session() as session:
            for stale in due:
                row = await session.get(PredictionRow, stale.id)
                if row is None or row.status != "open":
                    continue
                hist = histories.get(row.symbol)
                realized = _close_on(hist, row.target_date) if hist else None
                if realized is None:
                    if (today - row.target_date).days > VOID_AFTER_DAYS:
                        row.status, row.resolved_at = "void", now
                        voided += 1
                    else:
                        pending += 1
                    continue
                bench_hist = histories.get(row.benchmark)
                bench_ref = row.benchmark_reference
                if bench_ref is None and bench_hist is not None:
                    bench_ref = _close_on(bench_hist, row.made_on)
                bench_real = _close_on(bench_hist, row.target_date) if bench_hist is not None else None
                ret = realized / row.reference_price - 1
                row.realized_price = realized
                row.realized_return = ret
                row.outcome_up = realized > row.reference_price
                if bench_ref and bench_real:
                    row.benchmark_realized = bench_real
                    row.benchmark_return = bench_real / bench_ref - 1
                    row.outcome_outperform = ret > row.benchmark_return
                if row.q05 is not None and row.q95 is not None:
                    row.in_90 = row.q05 <= realized <= row.q95
                if row.q25 is not None and row.q75 is not None:
                    row.in_50 = row.q25 <= realized <= row.q75
                row.status, row.resolved_at = "resolved", now
                resolved += 1
        return ResolveResult(resolved=resolved, voided=voided, pending=pending)

    # ------------------------------------------------------------------ reading
    async def list(
        self,
        *,
        symbol: str | None = None,
        status: str | None = None,
        source: str | None = None,
        origin: str | None = None,
        limit: int = 200,
    ) -> list[PredictionOut]:
        async with self._db.session() as session:
            rows = await repo.list_predictions(
                session, symbol=symbol, status=status, source=source, origin=origin, limit=limit
            )
        return [to_out(r) for r in rows]

    async def counts(self) -> Sequence[LedgerCounts]:
        async with self._db.session() as session:
            raw = await repo.prediction_counts(session)
        keys = sorted({(o, src) for o, src, _ in raw})
        return [
            LedgerCounts(
                origin="backfill" if o == "backfill" else "live",
                source="model" if src == "model" else "forecast",
                open=raw.get((o, src, "open"), 0),
                resolved=raw.get((o, src, "resolved"), 0),
                void=raw.get((o, src, "void"), 0),
            )
            for o, src in keys
        ]

    async def scorecard(
        self, symbol: str | None = None, origin: Literal["live", "backfill", "all"] = "all"
    ) -> Scorecard:
        """Scores for live predictions, the backfilled replay, or both (``origin``)."""
        which = None if origin == "all" else origin
        async with self._db.session() as session:
            rows = await repo.score_rows(session, symbol=symbol, origin=which)
            recent = await repo.recent_predictions(session, symbol=symbol, origin=which, limit=60)
        groups: dict[tuple[str, int], list[Any]] = defaultdict(list)
        for r in rows:
            groups[(r.source, r.horizon_days)].append(r)
        sources: list[SourceScore] = []
        for (source, horizon), items in sorted(groups.items()):
            done = [r for r in items if r.status == "resolved"]
            open_n = sum(1 for r in items if r.status == "open")
            if source == "forecast":
                pairs = [
                    (r.prob_up, bool(r.outcome_up))
                    for r in done
                    if r.prob_up is not None and r.outcome_up is not None
                ]
                stats = score_probabilities(pairs)
                in50 = [float(r.in_50) for r in done if r.in_50 is not None]
                in90 = [float(r.in_90) for r in done if r.in_90 is not None]
                sources.append(
                    SourceScore(
                        source="forecast",
                        horizon_days=horizon,
                        resolved=len(done),
                        open=open_n,
                        coverage_50=_mean(in50),
                        coverage_90=_mean(in90),
                        calibration=_reliability(pairs),
                        **stats,
                    )
                )
            else:
                pairs = [
                    (r.prob_outperform, bool(r.outcome_outperform))
                    for r in done
                    if r.prob_outperform is not None and r.outcome_outperform is not None
                ]
                stats = score_probabilities(pairs)
                excess = [
                    (r.rank, r.realized_return - r.benchmark_return)
                    for r in done
                    if r.rank is not None and r.realized_return is not None and r.benchmark_return is not None
                ]
                sources.append(
                    SourceScore(
                        source="model",
                        horizon_days=horizon,
                        resolved=len(done),
                        open=open_n,
                        top_ranked_excess=_mean([e for rank, e in excess if rank <= 5]),
                        others_excess=_mean([e for rank, e in excess if rank > 5]),
                        calibration=_reliability(pairs),
                        **stats,
                    )
                )
        return Scorecard(
            computed_at=self._clock.now(),
            symbol=symbol,
            origin=origin,
            sources=sources,
            recent=[to_out(r) for r in recent],
        )

    # ------------------------------------------------------------------ scheduler
    async def run_scheduled(self) -> str:
        s = self._settings
        if not s.predictions_enabled:
            return "disabled"
        now = self._clock.now()
        report: list[str] = []
        if self._last_resolve is None or now - self._last_resolve >= RESOLVE_EVERY:
            res = await self.resolve_due()
            if res.resolved or res.voided:
                report.append(f"resolved {res.resolved}, voided {res.voided}")
        local = now.astimezone(NEW_YORK)
        today = local.date()
        due = is_trading_day(today) and local.time() >= time.fromisoformat(s.predictions_log_time)
        if due and self._logged_on != today:
            async with self._db.session() as session:
                already = await repo.predictions_logged_on(session, today)
            if already:
                self._logged_on = today
            elif self._failures.get(today, 0) >= MAX_FAILURES:
                report.append(f"gave up logging for {today} after {MAX_FAILURES} failures")
            else:
                try:
                    out = await self.log_daily()
                except Exception:
                    self._failures[today] = self._failures.get(today, 0) + 1
                    raise
                self._logged_on = today  # one run per day; skipped symbols are reported, not retried
                report.append(f"logged {out.logged} for {out.made_on}")
        return "; ".join(report) or "idle"
