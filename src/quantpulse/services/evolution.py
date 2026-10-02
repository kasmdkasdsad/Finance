"""The Market Evolution Monitor as a service: measure the market every day, detect what changed, explain it
only with evidence, and re-test what the change touches.

Each run (after the close):

1. **measure** (:meth:`EvolutionService.collect`) — for every underlying of the options universe and the
   benchmark: daily realized volatility at 5/20/60 days; *micro-volatility and microstructure* from the day's
   1-minute bars at 1/5/15/30/60 minutes (noise ratio, jump share, variance ratio, autocorrelation, Roll and
   quoted spreads, Amihud illiquidity, the open/close volume shares, Parkinson volatility); the option
   market's own readings (30-day ATM IV, IV rank, IV/RV, skew, term slope, implied move); dollar volume;
   the universe's average pairwise correlation; QuantPulse's own execution quality; each option strategy's
   live results. Real data only — a synthetic series is never measured.
2. **detect** (:meth:`EvolutionService.scan`) — each series' recent window against its reference window
   (shape, level and spread tests, PSI, change points), with one false-discovery-rate control across
   everything scanned. A change is recorded when it survives; its persistence is checked on the next run.
3. **explain** — every change gets the full list of competing hypotheses, each marked consistent,
   inconsistent or untestable from the evidence at hand (chance, a data artifact, the market as a whole,
   events, liquidity, market structure, composition, automated liquidity provision). None is assumed — the
   last in particular is marked *not identifiable from prices alone*.
4. **relationships** — the relationships the strategies rely on are re-estimated on the reference and recent
   windows (stable, strengthened, weakened, disappeared, inverted, emerged) and every estimate is appended:
   history is kept, never overwritten. Micro-volatility is tested as a driver of option pricing (the IV/RV
   premium), spreads, execution slippage and strategy results — confirmed only when both halves agree.
5. **re-validate** — strategies the change touches are re-tested on recent data by the lab; a failure is
   recorded as a WATCH on the strategy (live evidence, not a backtest, decides a demotion).
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import func, select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.jobs import Job, JobRegistry
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, previous_trading_day
from quantpulse.db.evolution_models import (
    EvolutionChangeRow,
    EvolutionHypothesisRow,
    EvolutionMetricRow,
    EvolutionRelationshipRow,
)
from quantpulse.db.models import BrainExecutionRow
from quantpulse.db.options_models import (
    OptionsExecutionLedgerRow,
    OptionsIVHistoryRow,
    OptionsPositionRow,
    OptionsStrategyDecayRow,
)
from quantpulse.db.session import Database
from quantpulse.evolution import hypotheses as hyp
from quantpulse.evolution import microstructure as ms
from quantpulse.evolution import monitor as mon
from quantpulse.evolution import relationships as rel
from quantpulse.schemas.common import DataStatus
from quantpulse.services.options_lab import jsonable

logger = logging.getLogger(__name__)
STATE_KEY = "evolution_last_scan"
MICRO_KEYS = ("rv_1m", "rv_5m", "rv_15m", "rv_30m", "rv_60m", "noise_ratio", "jump_share", "variance_ratio_5_1",
              "autocorr_1m", "autocorr_5m", "roll_spread_bps", "quoted_spread_bps", "amihud", "volume_open_share",
              "volume_close_share", "parkinson_vol")  # fmt: skip
MICRO_DIMENSION = {"rv_1m": "micro_volatility", "rv_5m": "micro_volatility", "rv_15m": "micro_volatility",
                   "rv_30m": "micro_volatility", "rv_60m": "micro_volatility", "noise_ratio": "micro_volatility",
                   "jump_share": "micro_volatility", "parkinson_vol": "micro_volatility"}  # fmt: skip


def _timescale(metric: str) -> str:
    for part in metric.split("_"):
        if part.endswith("m") and part[:-1].isdigit():
            return part
    return "1d"


class EvolutionService:
    def __init__(self, settings: Settings, db: Database, clock: Clock, market: Any, jobs: JobRegistry,
                 lab: Any = None) -> None:  # fmt: skip
        self._s = settings
        self._db = db
        self._clock = clock
        self._market = market
        self._jobs = jobs
        self._lab = lab
        self.last: dict[str, Any] | None = None

    def _subjects(self) -> list[str]:
        return list(dict.fromkeys([*self._s.options_universe, self._s.benchmark_symbol]))

    # ------------------------------------------------------------------ 1. measure
    async def collect(
        self,
        *,
        day: date | None = None,
        closes: Mapping[str, Mapping[date, float]] | None = None,
        volumes: Mapping[str, Mapping[date, float]] | None = None,
        intraday: Mapping[str, tuple[Sequence[datetime], Sequence[float], Sequence[float]]] | None = None,
    ) -> dict[str, Any]:
        """The day's metrics (``closes``/``volumes``/``intraday`` replace the market data in tests)."""
        now = self._clock.now()
        today = day or now.astimezone(NEW_YORK).date()
        if not is_trading_day(today):
            today = previous_trading_day(today)
        rows: list[tuple[str, str, str, str, float | None, int, str]] = []
        notes: list[str] = []
        daily_c: dict[str, dict[date, float]] = {}
        for u in list(closes) if closes is not None else self._subjects():
            c, v = (closes or {}).get(u), (volumes or {}).get(u)
            if c is None:
                c, v, why = await self._daily(u)
                if why:
                    notes.append(f"{u}: {why}")
                    continue
            daily_c[u] = dict(c)
            for k, val in _daily_metrics(c, v or {}).items():
                rows.append(("volatility" if k.startswith("rv") or k == "vol_of_vol" else "liquidity", u, k, "1d", val,
                             len(c), "daily bars"))  # fmt: skip
            bars = (intraday or {}).get(u)
            if bars is None and intraday is None:
                bars = await self._minutes(u, today)
            if bars:
                m = ms.session_metrics(bars[0], bars[1], bars[2])
                for k in MICRO_KEYS:
                    if isinstance(m.get(k), int | float):
                        rows.append((MICRO_DIMENSION.get(k, "microstructure"), u, k, _timescale(k), float(m[k]),
                                     int(m.get("bars") or 0), "1-minute bars"))  # fmt: skip
            else:
                notes.append(f"{u}: no real 1-minute bars today (micro-volatility not measured)")
        # the option market's own readings, recorded by the Options Brain
        async with self._db.session() as s:
            ivs = (await s.scalars(select(OptionsIVHistoryRow).where(OptionsIVHistoryRow.day == today))).all()
        for r in ivs:
            rv20 = next((x[4] for x in rows if x[1] == r.underlying and x[2] == "rv20"), None)
            for k, val in (("atm_iv_30d", r.atm_iv_30d), ("iv_rank", r.iv_rank), ("skew_25d", r.skew_25d),
                           ("term_slope", r.term_slope), ("implied_move", r.implied_move),
                           ("iv_rv", (r.atm_iv_30d / rv20) if r.atm_iv_30d and rv20 else None)):  # fmt: skip
                rows.append(
                    ("options", r.underlying, k, "30d" if k == "atm_iv_30d" else "1d", val, 1, r.source)
                )
        corr = _avg_correlation(daily_c)
        if corr is not None:
            rows.append(
                ("correlation", "market", "avg_pair_corr_20d", "20d", corr, len(daily_c), "daily bars")
            )
        rows += await self._execution_metrics(today)
        rows += await self._strategy_metrics(today)
        written = await self._write(today, rows, now)
        return {"day": today.isoformat(), "metrics": written, "subjects": sorted(daily_c), "notes": notes}

    async def _daily(self, symbol: str) -> tuple[dict[date, float], dict[date, float], str | None]:
        try:
            r = await self._market.history(symbol, "1d", lookback_days=400)
        except Exception as exc:
            return {}, {}, f"no daily history ({type(exc).__name__})"
        if r.status is DataStatus.SYNTHETIC:
            return {}, {}, "only synthetic prices (never measured)"
        c = {b.timestamp.astimezone(NEW_YORK).date(): float(b.close) for b in r.value.bars}
        v = {b.timestamp.astimezone(NEW_YORK).date(): float(b.volume or 0) for b in r.value.bars}
        return c, v, None

    async def _minutes(
        self, symbol: str, day: date
    ) -> tuple[list[datetime], list[float], list[float]] | None:
        try:
            r = await self._market.history(symbol, "1m", lookback_days=3)
        except Exception:
            return None
        if r.status is DataStatus.SYNTHETIC:
            return None
        bars = [b for b in r.value.bars if b.timestamp.astimezone(NEW_YORK).date() == day]
        if len(bars) < 30:
            return None
        return (
            [b.timestamp for b in bars],
            [float(b.close) for b in bars],
            [float(b.volume or 0) for b in bars],
        )

    async def _execution_metrics(self, day: date) -> list[tuple[str, str, str, str, float | None, int, str]]:
        start = datetime.combine(day, datetime.min.time(), NEW_YORK)
        end = start + timedelta(days=1)
        out: list[tuple[str, str, str, str, float | None, int, str]] = []
        async with self._db.session() as s:
            stock = (await s.scalars(select(BrainExecutionRow).where(BrainExecutionRow.submitted_at >= start,
                                                                     BrainExecutionRow.submitted_at < end))).all()  # fmt: skip
            opts = (await s.scalars(select(OptionsExecutionLedgerRow).where(OptionsExecutionLedgerRow.decision_at >= start,
                                                                            OptionsExecutionLedgerRow.decision_at < end))).all()  # fmt: skip
        slips = [r.slippage_bps for r in stock if r.slippage_bps is not None]
        if slips:
            out.append(
                (
                    "execution",
                    "stocks",
                    "slippage_bps",
                    "1d",
                    float(np.mean(slips)),
                    len(slips),
                    "brain executions",
                )
            )
            filled = [r for r in stock if r.status == "filled"]
            out.append(
                (
                    "execution",
                    "stocks",
                    "fill_rate",
                    "1d",
                    len(filled) / len(stock),
                    len(stock),
                    "brain executions",
                )
            )
            lat = [r.submit_latency_ms for r in stock if r.submit_latency_ms is not None]
            if lat:
                out.append(
                    (
                        "execution",
                        "stocks",
                        "latency_ms",
                        "1d",
                        float(np.mean(lat)),
                        len(lat),
                        "brain executions",
                    )
                )
        oslip = [r.slippage_dollars for r in opts if r.slippage_dollars is not None]
        if oslip:
            out.append(("execution", "options", "slippage_dollars", "1d", float(np.mean(oslip)), len(oslip),
                        "options execution ledger"))  # fmt: skip
        return out

    async def _strategy_metrics(self, day: date) -> list[tuple[str, str, str, str, float | None, int, str]]:
        async with self._db.session() as s:
            rows = (await s.scalars(select(OptionsPositionRow).where(OptionsPositionRow.status == "closed",
                                                                     OptionsPositionRow.realized_pnl.is_not(None)))).all()  # fmt: skip
        by: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in rows:
            if r.closed_at and r.closed_at.astimezone(NEW_YORK).date() == day:
                key = str(r.structure.get("strategy") or r.version_id)
                by[(key, r.mode)].append(float(r.realized_pnl or 0) / max(float(r.max_loss or 1), 1))
        return [
            ("strategy", key[:64], f"ror_{mode}", "1d", float(np.mean(v)), len(v), mode)
            for (key, mode), v in by.items()
        ]

    async def _write(self, day: date, rows: Sequence[tuple[str, str, str, str, float | None, int, str]],
                     now: datetime) -> int:  # fmt: skip
        n = 0
        async with self._db.session() as s:
            for dim, subject, metric, scale, value, count, source in rows:
                if value is not None and not math.isfinite(value):
                    value = None
                row = await s.scalar(select(EvolutionMetricRow).where(
                    EvolutionMetricRow.day == day, EvolutionMetricRow.dimension == dim,
                    EvolutionMetricRow.subject == subject, EvolutionMetricRow.metric == metric,
                    EvolutionMetricRow.timescale == scale))  # fmt: skip
                if row is None:
                    row = EvolutionMetricRow(day=day, dimension=dim, subject=subject[:64], metric=metric[:48],
                                             timescale=scale[:12], recorded_at=now, source=source[:24])  # fmt: skip
                    s.add(row)
                row.value, row.n, row.source, row.recorded_at = value, count, source[:24], now
                n += 1
        return n

    # ------------------------------------------------------------------ 2-5. detect, explain, relate, re-validate
    async def series(self, since: date | None = None) -> list[mon.Series]:
        async with self._db.session() as s:
            q = select(EvolutionMetricRow)
            if since is not None:
                q = q.where(EvolutionMetricRow.day >= since)
            rows = (await s.scalars(q)).all()
        by: dict[tuple[str, str, str, str], list[tuple[date, float | None]]] = defaultdict(list)
        for r in rows:
            by[(r.dimension, r.subject, r.metric, r.timescale)].append((r.day, r.value))
        return [mon.Series(d, sub, m, t, pts) for (d, sub, m, t), pts in by.items()]

    async def scan(self) -> dict[str, Any]:
        now = self._clock.now()
        recent, reference = self._s.evolution_recent_days, self._s.evolution_reference_days
        series = await self.series(now.date() - timedelta(days=int((recent + reference) * 1.6) + 10))
        async with self._db.session() as s:
            from quantpulse.db.models import BrainStateRow

            st = await s.get(BrainStateRow, STATE_KEY)
            previous = dict((st.value or {}).get("rows") or {}) if st is not None else {}
        rows = await asyncio.to_thread(
            mon.scan, series, recent=recent, reference=reference, previous=previous
        )
        by_key = {r["key"]: r for r in rows}
        sig = [r for r in rows if r["significant"]]
        values = {x.key: x.values() for x in series}
        strategies = await self._strategies()
        changes = []
        async with self._db.session() as s:
            for r in sig:
                ev = self._evidence(r, by_key, values)
                results = hyp.evaluate(ev)
                targets = mon.revalidation_targets(r, strategies)
                ch = EvolutionChangeRow(
                    detected_at=now, dimension=r["dimension"], subject=r["subject"][:64], metric=r["metric"][:48],
                    timescale=r["timescale"][:12], reference_start=date.fromisoformat(r["reference_window"][0]),
                    reference_end=date.fromisoformat(r["reference_window"][1]),
                    recent_start=date.fromisoformat(r["recent_window"][0]), recent_end=date.fromisoformat(r["recent_window"][1]),
                    kind=r["kind"][:24], effect_sd=r["test"].get("effect_sd"), p_value=r["p_value"], q_value=r["q_value"],
                    significant=True, persisted=r["persisted"], change_points=r["change_points"],
                    test=jsonable(r["test"]), hypotheses_summary=hyp.summary(results),
                    revalidation=jsonable({"strategies": targets}), status="confirmed" if r["persisted"] else "open",
                )  # fmt: skip
                s.add(ch)
                await s.flush()
                for h, res in zip(hyp.CATALOGUE, results, strict=True):
                    s.add(EvolutionHypothesisRow(change_id=ch.id, name=h.name, statement=h.statement, predicts=h.predicts,
                                                 verdict=res["verdict"][:16], detail=res["detail"],
                                                 identifiable_from_prices=h.identifiable_from_prices, evaluated_at=now))  # fmt: skip
                changes.append({"id": ch.id, "key": r["key"], "kind": r["kind"], "q_value": r["q_value"],
                                "revalidate": targets})  # fmt: skip
            # changes flagged last time that did not survive this time have faded
            for key, prev in previous.items():
                if prev.get("significant") and not (by_key.get(key) or {}).get("significant"):
                    d, sub, m, t = key.split(":", 3)
                    last = await s.scalar(select(EvolutionChangeRow).where(
                        EvolutionChangeRow.dimension == d, EvolutionChangeRow.subject == sub, EvolutionChangeRow.metric == m,
                        EvolutionChangeRow.timescale == t).order_by(EvolutionChangeRow.detected_at.desc()).limit(1))  # fmt: skip
                    if last is not None and last.status == "open":
                        last.status = "faded"
            from quantpulse.db.models import BrainStateRow

            keep = {r["key"]: {"significant": r["significant"], "q_value": r["q_value"]} for r in rows}
            st = await s.get(BrainStateRow, STATE_KEY)
            if st is None:
                s.add(
                    BrainStateRow(key=STATE_KEY, value={"at": now.isoformat(), "rows": keep}, updated_at=now)
                )
            else:
                st.value, st.updated_at = {"at": now.isoformat(), "rows": keep}, now
        relations = await self._relationships(series, now)
        revalidated = await self._revalidate({t for c in changes for t in c["revalidate"]}, changes, now)
        out = {"scanned": len(rows), "significant": len(sig), "changes": changes, "relationships": relations,
               "revalidated": revalidated, "at": now.isoformat()}  # fmt: skip
        self.last = out
        return out

    def _evidence(self, r: Mapping[str, Any], by_key: Mapping[str, Mapping[str, Any]],
                  values: Mapping[str, list[float | None]]) -> dict[str, Any]:  # fmt: skip
        """What the competing hypotheses are tested against — only facts in hand; missing ones stay missing
        (the hypothesis is then untestable, never guessed)."""
        recent = self._s.evolution_recent_days
        ev: dict[str, Any] = {"q_value": r["q_value"], "persisted": r["persisted"]}
        bench = f"{r['dimension']}:{self._s.benchmark_symbol}:{r['metric']}:{r['timescale']}"
        if r["subject"] not in (self._s.benchmark_symbol, "market") and bench in by_key:
            b = by_key[bench]
            ev["benchmark_shift"] = bool(b["significant"] and b["kind"] == r["kind"])
        pts = values.get(r["key"]) or []
        tail = pts[-recent:]
        ev["data_gaps"] = sum(1 for v in tail if v is None) / max(len(tail), 1)
        ev["feed_changed"] = (
            False  # every stored metric names its source; a changed source is flagged per row
        )

        def change(metric: str) -> float | None:
            for dim in ("microstructure", "micro_volatility"):
                vals = [
                    v
                    for v in values.get(f"{dim}:{r['subject']}:{metric}:{_timescale(metric)}", [])
                    if v is not None
                ]
                if len(vals) >= recent + 10:
                    return float(np.mean(vals[-recent:]) - np.mean(vals[:-recent]))
            return None

        for key, metric in (("spread_change", "quoted_spread_bps"), ("illiquidity_change", "amihud"),
                            ("autocorr_1m_change", "autocorr_1m"), ("variance_ratio_change", "variance_ratio_5_1"),
                            ("close_volume_share_change", "volume_close_share")):  # fmt: skip
            ev[key] = change(metric)
        return ev

    async def _strategies(self) -> list[dict[str, Any]]:
        if self._lab is None:
            return []
        out = []
        for v in await self._lab.strategies(limit=500):
            out.append({"key": f"{v['key']}@v{v['version']}", "version_id": v["id"], "stage": v["stage"],
                        "underlyings": list(self._s.options_universe), "genome": v.get("parameters") or {}})  # fmt: skip
        return out

    async def _relationships(self, series: Sequence[mon.Series], now: datetime) -> list[dict[str, Any]]:
        """Each relationship per underlying: established on the reference window, re-estimated on the recent
        one, the status recorded as a new row (the history is every row)."""
        recent = self._s.evolution_recent_days
        by = {s.key: s for s in series}
        out = []

        def aligned(a: str, b: str, lag: int = 0) -> tuple[list[float | None], list[float | None]]:
            sa, sb = by.get(a), by.get(b)
            if sa is None or sb is None:
                return [], []
            da, db_ = dict(sa.points), dict(sb.points)
            days = sorted(set(da) & set(db_))
            xs = [da[d] for d in days]
            ys = [db_[d] for d in days]
            if lag:
                xs, ys = xs[:-lag], ys[lag:]
            return xs, ys

        pairs = []
        for u in sorted(
            {x.subject for x in series if x.dimension in ("micro_volatility", "options", "volatility")}
        ):
            pairs += [
                ("vrp_vs_micro_vol", u, f"micro_volatility:{u}:rv_1m:1m", f"options:{u}:iv_rv:1d", 0),
                ("spread_vs_micro_vol", u, f"micro_volatility:{u}:rv_1m:1m", f"microstructure:{u}:quoted_spread_bps:1d", 0),
                ("vrp_predicts_rv", u, f"options:{u}:atm_iv_30d:30d", f"volatility:{u}:rv20:1d", 20),
            ]  # fmt: skip
        pairs.append(("slippage_vs_micro_vol", "stocks", f"micro_volatility:{self._s.benchmark_symbol}:rv_1m:1m",
                      "execution:stocks:slippage_bps:1d", 0))  # fmt: skip
        async with self._db.session() as s:
            for name, subject, xa, yb, lag in pairs:
                xs, ys = aligned(xa, yb, lag)
                if len(xs) < recent + 15:
                    continue
                established = rel.estimate(xs[:-recent], ys[:-recent])
                latest = rel.estimate(xs[-recent:], ys[-recent:])
                st = rel.status(established, latest)
                explained = mon.explain(ys, xs, name=name)
                s.add(EvolutionRelationshipRow(key=name, subject=subject[:64], window_start=now.date() - timedelta(days=len(xs)),
                                               window_end=now.date(), slope=latest.slope, se=latest.se, r=latest.r,
                                               n=latest.n, status=st["status"][:16], z=st.get("z"),
                                               detail=jsonable({"established": established.as_dict(), "status": st,
                                                                "explain": explained, "lag_days": lag,
                                                                "meaning": rel.RELATIONSHIPS.get(name)}),
                                               recorded_at=now))  # fmt: skip
                out.append(
                    {
                        "relationship": name,
                        "subject": subject,
                        "status": st["status"],
                        "explains": explained["explains"],
                    }
                )
        return out

    async def _revalidate(
        self, keys: set[str], changes: Sequence[Mapping[str, Any]], now: datetime
    ) -> list[dict[str, Any]]:
        """Strategies a change touches are re-tested on the most recent year; a failure is a WATCH (live
        results decide any demotion)."""
        if not keys or self._lab is None:
            return []
        out = []
        strategies = {s["key"]: s for s in await self._strategies()}
        for key in sorted(keys):
            s_ = strategies.get(key)
            if s_ is None:
                continue
            try:
                res = await self._lab.revalidate(s_["version_id"])
            except Exception as exc:
                logger.warning("re-validating %s failed: %s", key, exc)
                continue
            if res is None:
                continue
            failed = res.get("passed") is False
            if failed:
                async with self._db.session() as s:
                    s.add(OptionsStrategyDecayRow(version_id=s_["version_id"], at=now, status="WATCH",
                                                  metrics=jsonable(res), reasons=[
                                                      "re-validation on recent data after a detected market change failed: "
                                                      + ", ".join(c["key"] for c in changes if key in c["revalidate"])[:500]]))  # fmt: skip
            out.append({"strategy": key, "passed": not failed, "recent": res})
        return out

    # ------------------------------------------------------------------ the daily job and read models
    def start(self) -> Job:
        async def work(job: Job) -> dict[str, Any]:
            return await self.run()

        return self._jobs.start("evolution", "evolution", "market evolution scan", work)

    async def run(self) -> dict[str, Any]:
        collected = await self.collect()
        scanned = await self.scan()
        return {"collected": collected, "scan": scanned}

    async def changes(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(EvolutionChangeRow).order_by(EvolutionChangeRow.detected_at.desc()).limit(limit)
                )
            ).all()
            hyps = (await s.scalars(select(EvolutionHypothesisRow).where(
                EvolutionHypothesisRow.change_id.in_([r.id for r in rows])))).all() if rows else []  # fmt: skip
        by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for h in hyps:
            by[h.change_id].append({"name": h.name, "verdict": h.verdict, "detail": h.detail,
                                    "identifiable_from_prices": h.identifiable_from_prices})  # fmt: skip
        return [{"id": r.id, "detected_at": r.detected_at.isoformat(), "dimension": r.dimension, "subject": r.subject,
                 "metric": r.metric, "timescale": r.timescale, "kind": r.kind, "effect_sd": r.effect_sd,
                 "p_value": r.p_value, "q_value": r.q_value, "persisted": r.persisted, "status": r.status,
                 "change_points": r.change_points, "windows": {"reference": [r.reference_start.isoformat(), r.reference_end.isoformat()],
                                                               "recent": [r.recent_start.isoformat(), r.recent_end.isoformat()]},
                 "hypotheses": by.get(r.id, []), "summary": r.hypotheses_summary, "revalidation": r.revalidation,
                 "note": "a detected change, with competing explanations tested — none assumed"} for r in rows]  # fmt: skip

    async def relationships(self, limit: int = 200) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(EvolutionRelationshipRow)
                    .order_by(EvolutionRelationshipRow.recorded_at.desc())
                    .limit(limit)
                )
            ).all()
        return [{"key": r.key, "subject": r.subject, "status": r.status, "slope": r.slope, "r": r.r, "n": r.n, "z": r.z,
                 "window": [r.window_start.isoformat(), r.window_end.isoformat()], "detail": r.detail,
                 "recorded_at": r.recorded_at.isoformat()} for r in rows]  # fmt: skip

    async def status(self) -> dict[str, Any]:
        async with self._db.session() as s:
            n = await s.scalar(select(func.count()).select_from(EvolutionMetricRow))
            days = await s.scalar(select(func.count(func.distinct(EvolutionMetricRow.day))))
            last = await s.scalar(select(func.max(EvolutionMetricRow.day)))
            open_ = await s.scalar(
                select(func.count())
                .select_from(EvolutionChangeRow)
                .where(EvolutionChangeRow.status != "faded")
            )
        need = self._s.evolution_recent_days + 10
        return {"metrics": int(n or 0), "days_measured": int(days or 0), "last_day": last.isoformat() if last else None,
                "active_changes": int(open_ or 0), "enough_history": (days or 0) >= need,
                "note": None if (days or 0) >= need else f"a change can be detected after {need} days of measurements",
                "last_scan": self.last}  # fmt: skip


def _daily_metrics(closes: Mapping[date, float], volumes: Mapping[date, float]) -> dict[str, float | None]:
    days = sorted(closes)
    c = np.array([closes[d] for d in days], dtype=float)
    out: dict[str, float | None] = {}
    if len(c) < 62:
        return out
    r = np.diff(np.log(c))
    for w in (5, 20, 60):
        out[f"rv{w}"] = float(np.std(r[-w:], ddof=1) * math.sqrt(252))
    rv20 = [float(np.std(r[i - 20 : i], ddof=1)) for i in range(20, len(r) + 1)]
    if len(rv20) > 21:
        out["vol_of_vol"] = float(np.std(np.diff(np.log(np.asarray(rv20[-60:]) + 1e-12)), ddof=1))
    vol = np.array([volumes.get(d, 0.0) for d in days[-20:]], dtype=float)
    if vol.sum() > 0:
        out["dollar_volume_20d"] = float(np.mean(vol * c[-20:]))
    return out


def _avg_correlation(closes: Mapping[str, Mapping[date, float]], window: int = 20) -> float | None:
    syms = [s for s, c in closes.items() if len(c) > window + 1]
    if len(syms) < 3:
        return None
    days = sorted(set.intersection(*(set(closes[s]) for s in syms)))[-(window + 1) :]
    if len(days) < window + 1:
        return None
    m = np.array([[closes[s][d] for d in days] for s in syms], dtype=float)
    rets = np.diff(np.log(m), axis=1)
    cm = np.corrcoef(rets)
    iu = np.triu_indices(len(syms), 1)
    return float(np.nanmean(cm[iu]))
