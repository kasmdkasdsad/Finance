"""The options research lab as a service: the research library, the strategy population and its evidence,
kept in the database and advanced in the background.

* :meth:`OptionsLabService.seed` — the documented sources (never trusted: each claim starts ``UNTESTED``),
  their extracted rules (with what QuantPulse had to assume), and generation 0 of the population. Idempotent.
* :meth:`OptionsLabService.research` — one budgeted research run: strategies are evaluated
  (:func:`quantpulse.options.lab.pipeline.evaluate`) on real underlying prices with model-priced chains
  (labelled "model" on every result), promoted at most one gate at a time through the stages, and the
  population's false-discovery control, decay checks, experiments and next generation follow. The CPU work
  runs in a worker thread; nothing here can place an order.
* read models for the API and the dashboard.

Every version is immutable: a change is a new version with its parent and reason; stage changes append to
``stage_history`` and are never rewritten. Shadow and paper results (from the Options Brain's positions)
are the only evidence that can move a strategy past ``PAPER_SHADOW``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.jobs import Job, JobRegistry
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.options_models import (
    OptionsExperimentRow,
    OptionsGenerationRunRow,
    OptionsHypothesisRow,
    OptionsKnowledgeEdgeRow,
    OptionsPositionRow,
    OptionsStrategyBacktestRow,
    OptionsStrategyClaimRow,
    OptionsStrategyDecayRow,
    OptionsStrategyGenomeRow,
    OptionsStrategyRegimeRow,
    OptionsStrategyRuleRow,
    OptionsStrategyScoreRow,
    OptionsStrategySourceRow,
    OptionsStrategyStressTestRow,
    OptionsStrategyVersionRow,
    OptionsStrategyWalkforwardRow,
)
from quantpulse.db.session import Database
from quantpulse.options.lab import decay, experiments, overfit, population, promotion
from quantpulse.options.lab.discovery import graph_edges
from quantpulse.options.lab.extraction import extract
from quantpulse.options.lab.genome import Genome, from_dict
from quantpulse.options.lab.metrics import trade_stats
from quantpulse.options.lab.pipeline import LabData, evaluate, prepare
from quantpulse.options.lab.research import SEED
from quantpulse.schemas.common import DataStatus

logger = logging.getLogger(__name__)
S = promotion.Stage
RESEARCH_STAGES = (S.RESEARCH, S.EXTRACTED, S.BACKTESTING, S.VALIDATION, S.WALK_FORWARD)
REEVALUATE_AFTER = timedelta(days=7)
GENERATION_BUDGET = 12  # new candidates per generation (2 of them random immigrants)
HISTORY_DAYS = 1300  # about three and a half years of daily prices
RULE_TYPES = {"entry_signal": "entry", "iv_rank_min": "filter", "iv_rank_max": "filter", "iv_rv_min": "filter",
              "iv_rv_max": "filter", "event_filter": "filter", "regime_filter": "filter", "dte_min": "expiration",
              "dte_max": "expiration", "delta_target": "strike", "width_pct": "strike", "wing_pct": "strike",
              "take_profit": "exit", "stop_loss": "exit", "exit_dte": "exit", "max_hold_days": "exit",
              "risk_per_trade": "sizing"}  # fmt: skip


def jsonable(x: Any) -> Any:
    """JSON-safe: numpy scalars to numbers, dates to ISO strings, non-finite floats to ``None``."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple | set):
        return [jsonable(v) for v in x]
    if isinstance(x, date | datetime):
        return x.isoformat()
    if hasattr(x, "item") and not isinstance(x, str):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class OptionsLabService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        clock: Clock,
        market: Any,
        jobs: JobRegistry,
        reference: Any = None,
    ) -> None:
        self._s = settings
        self._db = db
        self._clock = clock
        self._market = market
        self._jobs = jobs
        self._reference = reference
        self._seeded = False
        self._running = asyncio.Lock()  # one research run at a time (the daily run and the research queue)
        self.last_run: dict[str, Any] | None = None

    # ------------------------------------------------------------------ seeding
    async def seed(self) -> dict[str, int]:
        """The research library and generation 0 (only what is missing is added)."""
        now = self._clock.now()
        added = {"sources": 0, "versions": 0}
        async with self._db.session() as s:
            have = set((await s.scalars(select(OptionsStrategySourceRow.source_key))).all())
            src_ids: dict[str, int] = {}
            for src in SEED:
                if src.key in have:
                    continue
                e = extract(src.rules_text)
                row = OptionsStrategySourceRow(
                    source_key=src.key, title=src.title, author=src.author, publication_date=src.published,
                    source_type=src.source_type, quality=src.quality, reference=src.reference, market=src.market,
                    time_period=src.period, limitations=src.limitations,
                    evidence_grade={**src.grade, "score": src.evidence_score()},
                    extraction_confidence=e.confidence, reproducibility="model-priced chains",
                    status="UNVERIFIED_RESEARCH" if e.status != "EXTRACTED" else "EXTRACTED", created_at=now,
                )  # fmt: skip
                s.add(row)
                await s.flush()
                src_ids[src.key] = row.id
                s.add(OptionsStrategyClaimRow(source_id=row.id, claim=src.claim, assumptions=list(src.assumptions),
                                              status="UNTESTED", test={}, created_at=now))  # fmt: skip
                added["sources"] += 1
            for row in (await s.scalars(select(OptionsStrategySourceRow))).all():
                src_ids.setdefault(row.source_key, row.id)
            known = set((await s.scalars(select(OptionsStrategyGenomeRow.genome_hash))).all())
            for child in population.generation0():
                if child.genome.hash in known:
                    continue
                key = child.source_key or f"simple-{child.genome.family}-{child.genome.entry_signal}"
                version = await self._add_version(
                    s, child.genome, key=key, origin=child.origin, reason=child.reason, generation=0,
                    source_id=src_ids.get(child.source_key or ""), now=now,
                )  # fmt: skip
                if child.assumptions and child.source_key:
                    claim = await s.scalar(select(OptionsStrategyClaimRow).where(
                        OptionsStrategyClaimRow.source_id == src_ids[child.source_key]))  # fmt: skip
                    if claim is not None:
                        for name, value in {**child.assumptions.get("stated", {}),
                                            **child.assumptions.get("assumed", {})}.items():  # fmt: skip
                            s.add(OptionsStrategyRuleRow(
                                claim_id=claim.id, genome_id=version.genome_id, rule_type=RULE_TYPES.get(name, "filter"),
                                text=f"{name} = {value}", expression={name: jsonable(value)},
                                explicit=name in child.assumptions.get("stated", {}),
                                assumed=name in child.assumptions.get("assumed", {})))  # fmt: skip
                added["versions"] += 1
        self._seeded = True
        return added

    async def _add_version(
        self,
        s: Any,
        g: Genome,
        *,
        key: str,
        origin: str,
        reason: str,
        generation: int,
        now: datetime,
        source_id: int | None = None,
        parent_id: int | None = None,
        second_parent_id: int | None = None,
        experiment_id: int | None = None,
    ) -> OptionsStrategyVersionRow:
        genome = await s.scalar(
            select(OptionsStrategyGenomeRow).where(OptionsStrategyGenomeRow.genome_hash == g.hash)
        )
        if genome is None:
            genome = OptionsStrategyGenomeRow(genome_hash=g.hash, family=g.family, direction=g.direction,
                                              params=jsonable(g.canonical()), parameter_count=g.parameter_count,
                                              created_at=now)  # fmt: skip
            s.add(genome)
            await s.flush()
        last = await s.scalar(select(func.max(OptionsStrategyVersionRow.version))
                              .where(OptionsStrategyVersionRow.strategy_key == key))  # fmt: skip
        stage = S.EXTRACTED if g.valid else S.RESEARCH
        history = [promotion.stage_record(S.RESEARCH, now.isoformat(), reason)]
        if g.valid:
            history.append(
                promotion.stage_record(S.EXTRACTED, now.isoformat(), "the genome is explicit and valid")
            )
        row = OptionsStrategyVersionRow(
            strategy_key=key[:64], version=(last or 0) + 1, name=g.describe()[:160], genome_id=genome.id,
            parent_id=parent_id, second_parent_id=second_parent_id, generation=generation, origin=origin,
            source_id=source_id, experiment_id=experiment_id, reason=reason, stage=stage.value,
            stage_history=history, role="none", is_baseline=origin == "seed", created_at=now, stage_changed_at=now,
        )  # fmt: skip
        s.add(row)
        await s.flush()
        return row

    # ------------------------------------------------------------------ data
    async def _prices(self, symbols: Sequence[str]) -> tuple[dict[str, dict[date, float]], dict[str, str]]:
        closes: dict[str, dict[date, float]] = {}
        skipped: dict[str, str] = {}
        for sym in symbols:
            try:
                r = await self._market.history(sym, "1d", lookback_days=HISTORY_DAYS)
            except Exception as exc:
                skipped[sym] = f"no history ({type(exc).__name__})"
                continue
            if r.status is DataStatus.SYNTHETIC:
                skipped[sym] = "only synthetic prices: never researched on"
                continue
            closes[sym] = {b.timestamp.astimezone(NEW_YORK).date(): float(b.close) for b in r.value.bars}
        return closes, skipped

    # ------------------------------------------------------------------ the research run
    def start_research(self) -> Job:
        async def work(job: Job) -> dict[str, Any]:
            return await self.research()

        return self._jobs.start("options-research", "options-research", "options research lab run", work)

    async def research(
        self,
        *,
        budget_seconds: float | None = None,
        closes: Mapping[str, Mapping[date, float]] | None = None,
        max_evaluations: int | None = None,
    ) -> dict[str, Any]:
        """One budgeted research run (see the module docstring). ``closes`` replaces the market data (tests).
        One run at a time: a second caller while one runs gets ``{"skipped": ...}``."""
        if self._running.locked():
            return {"skipped": "an options research run is already in progress"}
        async with self._running:
            return await self._research(
                budget_seconds=budget_seconds, closes=closes, max_evaluations=max_evaluations
            )

    async def _research(
        self,
        *,
        budget_seconds: float | None,
        closes: Mapping[str, Mapping[date, float]] | None,
        max_evaluations: int | None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        budget = budget_seconds if budget_seconds is not None else self._s.options_research_budget_seconds
        deadline = started + budget
        now = self._clock.now()
        await self.seed()
        report: dict[str, Any] = {"started_at": now.isoformat(), "budget_seconds": budget, "evaluated": [],
                                  "promoted": [], "demoted": [], "experiments": 0, "generation": None,
                                  "data": {}, "label": "model-priced chains over real underlying prices"}  # fmt: skip
        if closes is None:
            prices, skipped = await self._prices(self._s.options_universe)
            report["data"]["skipped"] = skipped
        else:
            prices = {u: dict(v) for u, v in closes.items()}
        data = await asyncio.to_thread(prepare, prices)
        report["data"]["underlyings"] = list(data.underlyings)
        report["data"]["grade"] = data.grade
        if not data.underlyings:
            report["note"] = "no underlying has enough real price history: nothing was evaluated"
            self.last_run = report
            return report

        todo = await self._queue(now)
        async with self._db.session() as s:
            n_trials = int(await s.scalar(select(func.count(OptionsStrategyVersionRow.id))) or 1)
        for vid in todo:
            if time.monotonic() > deadline or (
                max_evaluations is not None and len(report["evaluated"]) >= max_evaluations
            ):
                break
            async with self._db.session() as s:
                v = await s.get(OptionsStrategyVersionRow, vid)
                genome_row = await s.get(OptionsStrategyGenomeRow, v.genome_id) if v is not None else None
            if v is None or genome_row is None:
                continue
            g = from_dict(genome_row.params)
            res = await asyncio.to_thread(evaluate, g, data, n_trials=n_trials, equity=self._s.brain_book_capital,
                                          deadline=deadline)  # fmt: skip
            await self._store(vid, g, res, data, now)
            report["evaluated"].append({"version_id": vid, "strategy": v.strategy_key, "family": g.family,
                                        "stopped_at": res["stopped_at"], "incomplete": res["incomplete"],
                                        "trades": res["evidence"].get("backtest_trades")})  # fmt: skip
        await self._fdr(now)
        report["promoted"], report["demoted"] = await self._promote_all(now)
        report["experiments"] = await self._experiments(now)
        if time.monotonic() < deadline:
            report["generation"] = await self._generation(now, budget=GENERATION_BUDGET)
        report["seconds"] = round(time.monotonic() - started, 1)
        self.last_run = report
        logger.info("options research: %d evaluated, %d promoted, %d experiments", len(report["evaluated"]),
                    len(report["promoted"]), report["experiments"])  # fmt: skip
        return report

    async def _queue(self, now: datetime) -> list[int]:
        """What to evaluate: experiment children first (by information value), then versions never evaluated,
        then the stalest evaluations — only versions still in research."""
        async with self._db.session() as s:
            versions = (await s.scalars(select(OptionsStrategyVersionRow).where(
                OptionsStrategyVersionRow.stage.in_([x.value for x in RESEARCH_STAGES])))).all()  # fmt: skip
            last = dict((await s.execute(select(OptionsStrategyBacktestRow.version_id,
                                                func.max(OptionsStrategyBacktestRow.run_at))
                                         .group_by(OptionsStrategyBacktestRow.version_id))).all())  # fmt: skip
            queued = {r.child_version_id: r.priority for r in (await s.scalars(select(OptionsExperimentRow).where(
                OptionsExperimentRow.status.in_(["QUEUED", "RUNNING"])))).all() if r.child_version_id}  # fmt: skip
        due = []
        for v in versions:
            at = last.get(v.id)
            if at is not None and now - at < REEVALUATE_AFTER:
                continue
            due.append(
                (0 if v.id in queued else 1, -(queued.get(v.id) or 0.0), at is not None, at or now, v.id)
            )
        return [x[-1] for x in sorted(due)]

    async def _store(self, vid: int, g: Genome, res: dict[str, Any], data: LabData, now: datetime) -> None:
        ev = res["evidence"]
        async with self._db.session() as s:
            first = data.days[min(260, len(data.days) - 1)]
            for model, summary in (res.get("backtests") or {}).items():
                s.add(OptionsStrategyBacktestRow(
                    version_id=vid, run_at=now, purpose="full", data_source=data.grade, execution_model=model,
                    period_start=first, period_end=data.days[-1], universe=list(data.underlyings),
                    trades=int(summary["metrics"].get("trades") or 0), metrics=jsonable(summary["metrics"]),
                    label=summary["label"], details=jsonable({"skipped": summary["skipped"], "stopped_at": res["stopped_at"],
                                                             "incomplete": res["incomplete"]})))  # fmt: skip
            if "walkforward" in res:
                wf = res["walkforward"]
                s.add(OptionsStrategyWalkforwardRow(version_id=vid, run_at=now, data_source=data.grade,
                                                    windows=jsonable(wf["windows"]),
                                                    summary=jsonable({k: v for k, v in wf.items() if k != "windows"}
                                                                     | {"overfit": res.get("overfit")}),
                                                    passed=bool(ev.get("walkforward_passed"))))  # fmt: skip
            for kind, key, passed in (("monte_carlo", "montecarlo", (ev.get("montecarlo_ruin") or 1) <= 0.01),
                                      ("tail", "tail", ev.get("tail_passed")), ("baselines", "baselines", ev.get("beats_baselines")),
                                      ("critic", "critic", ev.get("critic_survived"))):  # fmt: skip
                if key in res:
                    body = res[key]
                    s.add(OptionsStrategyStressTestRow(version_id=vid, run_at=now, kind=kind,
                                                       scenarios=jsonable(list((body.get("scenarios") or body.get("attacks") or {}).keys())
                                                                          if isinstance(body, dict) else []),
                                                       summary=jsonable(body), passed=bool(passed)))  # fmt: skip
            sc = res.get("score")
            if sc is not None:
                s.add(OptionsStrategyScoreRow(version_id=vid, scored_at=now, dimensions=jsonable(sc["dimensions"]),
                                              overfit_risk=ev.get("overfit_risk"), eligible=bool(sc["eligible"]),
                                              reasons=jsonable(sc.get("blocking", []))))  # fmt: skip
            trades = res.get("trades") or []
            by_regime: dict[str, list[dict[str, Any]]] = {}
            for t in trades:
                by_regime.setdefault(str(t.get("regime") or "?"), []).append(t)
            for regime, ts in by_regime.items():
                stats = trade_stats([t["pnl"] for t in ts], [t.get("max_loss") or 0 for t in ts])
                row = await s.scalar(select(OptionsStrategyRegimeRow).where(
                    OptionsStrategyRegimeRow.version_id == vid, OptionsStrategyRegimeRow.regime == regime,
                    OptionsStrategyRegimeRow.source == "backtest"))  # fmt: skip
                if row is None:
                    row = OptionsStrategyRegimeRow(
                        version_id=vid, regime=regime, source="backtest", updated_at=now
                    )
                    s.add(row)
                row.trades, row.expectancy = len(ts), stats.get("expectancy_on_risk")
                row.win_rate, row.evidence, row.updated_at = stats.get("win_rate"), jsonable(stats), now
            v = await s.get(OptionsStrategyVersionRow, vid)
            assert v is not None
            v_ev = dict((v.stage_history[-1].get("evidence") or {}) if v.stage_history else {})
            v_ev.update({"latest": jsonable(ev), "stopped_at": res["stopped_at"], "evaluated_at": now.isoformat(),
                         "grade": data.grade})  # fmt: skip
            # the evidence travels with the version (the stage history keeps each stage's own snapshot)
            v.stage_history = (
                [*v.stage_history[:-1], {**v.stage_history[-1], "evidence": v_ev}] if v.stage_history else []
            )
            src = await s.get(OptionsStrategySourceRow, v.source_id) if v.source_id else None
            edges = graph_edges(strategy=f"{v.strategy_key}@v{v.version}", regimes=res.get("regimes"),
                                source=src.source_key if src else None,
                                features=[f for f in ("iv_rank", "iv_rv", "trend", "event") if _uses(g, f)])  # fmt: skip
            await self._edges(s, edges, now)
            # the experiment this version answers
            exp = await s.scalar(select(OptionsExperimentRow).where(OptionsExperimentRow.child_version_id == vid,
                                                                    OptionsExperimentRow.status.in_(["QUEUED", "RUNNING"])))  # fmt: skip
            if exp is not None:
                parent = (res.get("walkforward") or {}).get("oos") or {}
                pm = await self._latest_metrics(s, exp.parent_version_id)
                child = {
                    "trades": ev.get("backtest_trades"),
                    "expectancy_on_risk": (ev.get("ror_by_model") or {}).get("REALISTIC"),
                }
                status, why = experiments.decide(pm, child)
                exp.status, exp.decision, exp.finished_at = status, why, now
                exp.metrics = jsonable({"child": child, "parent": pm, "child_oos": parent})
                exp.result = jsonable({"stopped_at": res["stopped_at"]})
            # the claim the source made, tested (a sign test on the strategy's own trades, labelled model)
            if v.source_id and trades:
                claim = await s.scalar(
                    select(OptionsStrategyClaimRow).where(OptionsStrategyClaimRow.source_id == v.source_id)
                )
                if claim is not None:
                    from quantpulse.options.lab.research import sign_test

                    st = sign_test([t["pnl"] / max(float(t.get("max_loss") or 1), 1) for t in trades])
                    claim.test = jsonable(
                        {**st, "data": data.grade, "version_id": vid, "at": now.isoformat()}
                    )
                    p = st.get("p_value")
                    claim.status = ("SUPPORTED" if p is not None and p < 0.05 and (st.get("mean") or 0) > 0
                                    else "NOT_REPRODUCED" if p is not None and (st.get("mean") or 0) <= 0 and len(trades) >= 30
                                    else "INCONCLUSIVE")  # fmt: skip

    async def _latest_metrics(self, s: Any, version_id: int | None) -> dict[str, Any]:
        if version_id is None:
            return {}
        row = await s.scalar(select(OptionsStrategyBacktestRow).where(
            OptionsStrategyBacktestRow.version_id == version_id,
            OptionsStrategyBacktestRow.execution_model == "REALISTIC").order_by(OptionsStrategyBacktestRow.run_at.desc()).limit(1))  # fmt: skip
        return (
            {"trades": row.trades, "expectancy_on_risk": row.metrics.get("expectancy_on_risk")} if row else {}
        )

    async def _edges(self, s: Any, edges: Sequence[Mapping[str, Any]], now: datetime) -> None:
        for e in edges:
            row = await s.scalar(select(OptionsKnowledgeEdgeRow).where(
                OptionsKnowledgeEdgeRow.src_type == e["src_type"], OptionsKnowledgeEdgeRow.src_id == str(e["src_id"])[:64],
                OptionsKnowledgeEdgeRow.relation == e["relation"], OptionsKnowledgeEdgeRow.dst_type == e["dst_type"],
                OptionsKnowledgeEdgeRow.dst_id == str(e["dst_id"])[:64]))  # fmt: skip
            if row is None:
                row = OptionsKnowledgeEdgeRow(src_type=e["src_type"], src_id=str(e["src_id"])[:64], relation=e["relation"],
                                              dst_type=e["dst_type"], dst_id=str(e["dst_id"])[:64], updated_at=now)  # fmt: skip
                s.add(row)
            row.weight, row.evidence, row.updated_at = (
                float(e.get("weight", 1.0)),
                jsonable(e.get("evidence", {})),
                now,
            )

    # ------------------------------------------------------------------ the population's discipline
    async def _fdr(self, now: datetime) -> None:
        """Benjamini–Hochberg across every version with a REALISTIC backtest: a strategy is a discovery only if
        it survives the whole population's multiple testing."""
        async with self._db.session() as s:
            rows = (await s.scalars(select(OptionsStrategyBacktestRow).where(
                OptionsStrategyBacktestRow.execution_model == "REALISTIC").order_by(OptionsStrategyBacktestRow.run_at))).all()  # fmt: skip
            latest = {r.version_id: r for r in rows}
            ids = list(latest)
            ps = [overfit.p_value_from_t(latest[i].metrics.get("t_stat"), latest[i].trades) for i in ids]
            flags = overfit.benjamini_hochberg(ps, 0.10) if ps else []
            for vid, p, ok in zip(ids, ps, flags, strict=True):
                v = await s.get(OptionsStrategyVersionRow, vid)
                if v is None or not v.stage_history:
                    continue
                last = dict(v.stage_history[-1])
                evd = dict(last.get("evidence") or {})
                evd["fdr"] = {
                    "p_value": round(p, 6),
                    "discovery": bool(ok),
                    "population": len(ids),
                    "q": 0.10,
                }
                v.stage_history = [*v.stage_history[:-1], {**last, "evidence": evd}]

    async def live_evidence(self, s: Any, version_id: int) -> dict[str, Any]:
        """Shadow and paper results of a version (from the Options Brain's closed positions)."""
        rows = (await s.scalars(select(OptionsPositionRow).where(OptionsPositionRow.version_id == version_id,
                                                                  OptionsPositionRow.status == "closed"))).all()  # fmt: skip
        out: dict[str, Any] = {}
        for mode in ("shadow", "paper"):
            ts = [r for r in rows if r.mode == mode and r.realized_pnl is not None]
            ror = [float(r.realized_pnl or 0) / max(float(r.max_loss or 1), 1) for r in ts]
            stats = trade_stats(
                [float(r.realized_pnl or 0) for r in ts], [float(r.max_loss or 0) for r in ts]
            )
            out[mode] = {"trades": len(ts), "sessions": len({r.closed_at.date() for r in ts if r.closed_at}),
                         "ror": round(sum(ror) / len(ror), 5) if ror else None, "t": stats.get("t_stat"),
                         "returns": ror}  # fmt: skip
        return out

    def _evidence(
        self, v: OptionsStrategyVersionRow, family: str, live: Mapping[str, Any]
    ) -> promotion.Evidence:
        latest = (v.stage_history[-1].get("evidence") or {}) if v.stage_history else {}
        ev = dict(latest.get("latest") or {})
        fdr = latest.get("fdr") or {}
        e = promotion.Evidence(family=family)
        for k in ("genome_problems", "backtests", "ror_by_model", "backtest_trades", "validation_ror", "overfit_risk",
                  "walkforward_passed", "montecarlo_ruin", "tail_passed", "beats_baselines", "critic_survived"):  # fmt: skip
            if k in ev and ev[k] is not None:
                setattr(e, k, ev[k])
        e.fdr_discovery = fdr.get("discovery")
        sh, pp = live.get("shadow") or {}, live.get("paper") or {}
        e.shadow_trades, e.shadow_sessions, e.shadow_ror = (
            sh.get("trades", 0),
            sh.get("sessions", 0),
            sh.get("ror"),
        )
        e.paper_trades, e.paper_ror, e.paper_t = pp.get("trades", 0), pp.get("ror"), pp.get("t")
        e.decay_status = latest.get("decay")
        return e

    async def _promote_all(self, now: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Each version moves forward as far as its evidence allows — one gate at a time, each recorded — and
        live decay can move it back. Nothing here reaches PAPER_ACTIVE without real shadow trades."""
        policy = promotion.Policy(min_shadow_trades=self._s.options_min_shadow_trades,
                                  min_shadow_sessions=min(self._s.options_min_shadow_trades, 20))  # fmt: skip
        promoted, demoted = [], []
        async with self._db.session() as s:
            versions = (await s.scalars(select(OptionsStrategyVersionRow).where(
                OptionsStrategyVersionRow.stage != S.RETIRED.value))).all()  # fmt: skip
            for v in versions:
                genome = await s.get(OptionsStrategyGenomeRow, v.genome_id)
                family = genome.family if genome else ""
                live = await self.live_evidence(s, v.id)
                ev = self._evidence(v, family, live)
                stage = S(v.stage)
                if stage in promotion.SHADOW_STAGES:
                    returns = (
                        (live.get("paper") or {}).get("returns")
                        or (live.get("shadow") or {}).get("returns")
                        or []
                    )
                    val = ev.validation_ror or 0.0
                    status = decay.assess(returns, decay.Expectation(mean=val, sd=max(abs(val) * 3, 0.2)))
                    s.add(OptionsStrategyDecayRow(version_id=v.id, at=now, status=status["status"],
                                                  metrics=jsonable({k: x for k, x in status.items() if k != "reasons"}),
                                                  reasons=jsonable(status.get("reasons", []))))  # fmt: skip
                    ev.decay_status = status["status"]
                    down = promotion.demote_for(stage, ev)
                    if down is not None:
                        v.stage, v.stage_changed_at = down[0].value, now
                        v.stage_history = [*v.stage_history, promotion.stage_record(down[0], now.isoformat(), down[1],
                                                                                    {"decay": status})]  # fmt: skip
                        demoted.append(
                            {
                                "version_id": v.id,
                                "strategy": v.strategy_key,
                                "to": down[0].value,
                                "why": down[1],
                            }
                        )
                        continue
                while True:
                    new, _missing = promotion.advance(stage, ev, policy)
                    if new == stage:
                        break
                    v.stage_history = [*v.stage_history, promotion.stage_record(new, now.isoformat(), f"gate for {new.value} passed",
                                                                                jsonable(ev.__dict__))]  # fmt: skip
                    promoted.append(
                        {"version_id": v.id, "strategy": v.strategy_key, "from": stage.value, "to": new.value}
                    )
                    stage = new
                    v.stage, v.stage_changed_at = new.value, now
                if v.stage_history:
                    last = dict(v.stage_history[-1])
                    last["next_gate"] = promotion.gate(promotion.next_stage(stage) or stage, ev, policy)
                    last.setdefault("evidence", {})
                    v.stage_history = [*v.stage_history[:-1], jsonable(last)]
        return promoted, demoted

    async def _experiments(self, now: datetime) -> int:
        """Competing fixes for versions stopped by a recognisable failure; each a child version (never an edit),
        queued by expected information."""
        made = 0
        async with self._db.session() as s:
            versions = (await s.scalars(select(OptionsStrategyVersionRow).where(
                OptionsStrategyVersionRow.stage.in_([S.VALIDATION.value, S.BACKTESTING.value, S.WALK_FORWARD.value])))).all()  # fmt: skip
            tried = set((await s.scalars(select(OptionsExperimentRow.parent_version_id))).all())
            rationales: Sequence[dict[str, Any]] = (
                await s.scalars(select(OptionsHypothesisRow.rationale))
            ).all()
            kinds = [str((r or {}).get("kind")) for r in rationales]
            for v in versions:
                if v.id in tried:
                    continue
                genome = await s.get(OptionsStrategyGenomeRow, v.genome_id)
                if genome is None:
                    continue
                regimes = {r.regime: r.expectancy for r in (await s.scalars(select(OptionsStrategyRegimeRow).where(
                    OptionsStrategyRegimeRow.version_id == v.id))).all() if r.expectancy is not None}  # fmt: skip
                findings = _findings(v, regimes)
                if not findings:
                    continue
                parent = from_dict(genome.params)
                m = await self._latest_metrics(s, v.id)
                for finding in findings[:2]:
                    for prop in experiments.from_failure(parent, finding, {"regimes": regimes})[:3]:
                        h = OptionsHypothesisRow(created_at=now, source="failure", statement=prop.hypothesis,
                                                 parent_version_id=v.id, rationale=jsonable({"finding": finding, "kind": prop.kind, **prop.rationale}),
                                                 status="testing")  # fmt: skip
                        s.add(h)
                        await s.flush()
                        exp = OptionsExperimentRow(hypothesis_id=h.id, parent_version_id=v.id, status="QUEUED",
                                                   parameter_changes=jsonable(prop.changes), created_at=now,
                                                   dataset={"grade": "model", "universe": list(self._s.options_universe)},
                                                   priority=experiments.information_value(
                                                       n=float(m.get("trades") or 1), sd=0.3,
                                                       mean=float(m.get("expectancy_on_risk") or 0.0), added=30,
                                                       tested_kinds=kinds, kind=prop.kind))  # fmt: skip
                        s.add(exp)
                        await s.flush()
                        child = await self._add_version(s, prop.child, key=v.strategy_key, origin="experiment",
                                                        reason=prop.hypothesis, generation=v.generation + 1, now=now,
                                                        parent_id=v.id, experiment_id=exp.id)  # fmt: skip
                        exp.child_version_id = child.id
                        made += 1
        return made

    async def _generation(self, now: datetime, *, budget: int) -> dict[str, Any] | None:
        async with self._db.session() as s:
            last = await s.scalar(select(func.max(OptionsGenerationRunRow.generation))) or 0
            versions = (await s.scalars(select(OptionsStrategyVersionRow))).all()
            members = []
            for v in versions:
                genome = await s.get(OptionsStrategyGenomeRow, v.genome_id)
                if genome is None:
                    continue
                m = await self._latest_metrics(s, v.id)
                regimes = {r.regime: r.expectancy for r in (await s.scalars(select(OptionsStrategyRegimeRow).where(
                    OptionsStrategyRegimeRow.version_id == v.id))).all() if r.expectancy is not None}  # fmt: skip
                members.append(population.Member(f"{v.strategy_key}@v{v.version}", v.id, from_dict(genome.params), v.stage,
                                                 score=m.get("expectancy_on_risk"), regimes=regimes))  # fmt: skip
            alive = [m for m in members if m.stage in population.PASSED]
            if alive:
                gen = last + 1
                children = population.next_generation(
                    members, gen, budget=budget, seed=self._s.options_min_shadow_trades
                )
            else:  # nothing has passed VALIDATION yet: explore instead of waiting (the generations start later)
                gen = last
                children = population.explore(members, budget=budget, seed=self._s.options_min_shadow_trades)
            if not children:
                return None
            run = OptionsGenerationRunRow(generation=gen, started_at=now, budget={"children": budget},
                                          summary={"kind": "generation" if alive else "exploration"})  # fmt: skip
            s.add(run)
            await s.flush()
            by_key = {m.key: m for m in members}
            for c in children:
                parents = [by_key[p] for p in c.parents if p in by_key]
                key = (
                    parents[0].key.split("@")[0]
                    if len(parents) == 1
                    else f"x-{c.genome.hash[:10]}"
                    if parents
                    else f"new-{c.genome.hash[:10]}"
                )
                await self._add_version(s, c.genome, key=key, origin=c.origin, reason=c.reason, generation=gen, now=now,
                                        parent_id=parents[0].version_id if parents else None,
                                        second_parent_id=parents[1].version_id if len(parents) > 1 else None)  # fmt: skip
            run.created, run.finished_at = len(children), now
            run.summary = {**run.summary, "origins": sorted({c.origin for c in children})}
            return {"generation": gen, "children": len(children), "kind": run.summary["kind"],
                    "origins": run.summary["origins"]}  # fmt: skip

    async def revalidate(
        self,
        version_id: int,
        *,
        closes: Mapping[str, Mapping[date, float]] | None = None,
        sessions: int = 252,
    ) -> dict[str, Any] | None:
        """Re-test a version on the most recent ``sessions`` only (after a detected market change): REALISTIC
        and PESSIMISTIC fills must both stay positive per dollar at risk. Stored as a ``revalidation`` backtest;
        fewer than ten trades is inconclusive (``passed`` None), never a pass."""
        from quantpulse.options.fills import ExecutionModel
        from quantpulse.options.lab.backtest import BacktestConfig, run

        async with self._db.session() as s:
            v = await s.get(OptionsStrategyVersionRow, version_id)
            g_row = await s.get(OptionsStrategyGenomeRow, v.genome_id) if v is not None else None
        if v is None or g_row is None:
            return None
        g = from_dict(g_row.params)
        prices = (
            {u: dict(c) for u, c in closes.items()}
            if closes is not None
            else (await self._prices(self._s.options_universe))[0]
        )
        data = await asyncio.to_thread(prepare, prices)
        if not data.underlyings:
            return None
        days = data.days
        start = days[max(0, len(days) - sessions)]
        now = self._clock.now()
        out: dict[str, Any] = {"window": [start.isoformat(), days[-1].isoformat()], "grade": data.grade}
        for model in (ExecutionModel.REALISTIC, ExecutionModel.PESSIMISTIC):
            cfg = BacktestConfig(
                start, days[-1], data.underlyings, equity=self._s.brain_book_capital, model=model
            )
            res = await asyncio.to_thread(run, g, data.source, data.features, cfg)
            out[model.value] = {
                "trades": res.metrics.get("trades"),
                "expectancy_on_risk": res.metrics.get("expectancy_on_risk"),
            }
            async with self._db.session() as s:
                s.add(OptionsStrategyBacktestRow(version_id=version_id, run_at=now, purpose="revalidation",
                                                 data_source=data.grade, execution_model=model.value, period_start=start,
                                                 period_end=days[-1], universe=list(data.underlyings),
                                                 trades=int(res.metrics.get("trades") or 0), metrics=jsonable(res.metrics),
                                                 label=res.label, details={"reason": "re-validation after a market change"}))  # fmt: skip
        n = int(out["REALISTIC"]["trades"] or 0)
        if n < 10:
            out["passed"], out["note"] = None, f"{n} trades in the recent window: inconclusive"
        else:
            out["passed"] = all((out[m]["expectancy_on_risk"] or 0) > 0 for m in ("REALISTIC", "PESSIMISTIC"))
        return out

    # ------------------------------------------------------------------ read models
    async def strategies(self, *, stage: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            q = (
                select(OptionsStrategyVersionRow)
                .order_by(OptionsStrategyVersionRow.stage_changed_at.desc())
                .limit(limit)
            )
            if stage:
                q = q.where(OptionsStrategyVersionRow.stage == stage)
            rows = (await s.scalars(q)).all()
            out = []
            for v in rows:
                g = await s.get(OptionsStrategyGenomeRow, v.genome_id)
                m = await self._latest_metrics(s, v.id)
                live = await self.live_evidence(s, v.id)
                out.append(_version_out(v, g, m, live))
            return out

    async def strategy(self, version_id: int) -> dict[str, Any] | None:
        async with self._db.session() as s:
            v = await s.get(OptionsStrategyVersionRow, version_id)
            if v is None:
                return None
            g = await s.get(OptionsStrategyGenomeRow, v.genome_id)
            m = await self._latest_metrics(s, v.id)
            live = await self.live_evidence(s, v.id)
            out = _version_out(v, g, m, live)
            bts = (await s.scalars(select(OptionsStrategyBacktestRow).where(OptionsStrategyBacktestRow.version_id == v.id)
                                   .order_by(OptionsStrategyBacktestRow.run_at.desc()).limit(10))).all()  # fmt: skip
            out["backtests"] = [{"execution_model": b.execution_model, "data_source": b.data_source, "label": b.label,
                                 "period": [b.period_start.isoformat(), b.period_end.isoformat()], "trades": b.trades,
                                 "metrics": b.metrics, "run_at": b.run_at.isoformat()} for b in bts]  # fmt: skip
            wf = await s.scalar(select(OptionsStrategyWalkforwardRow).where(OptionsStrategyWalkforwardRow.version_id == v.id)
                                .order_by(OptionsStrategyWalkforwardRow.run_at.desc()).limit(1))  # fmt: skip
            out["walkforward"] = (
                {"passed": wf.passed, "summary": wf.summary, "windows": wf.windows} if wf else None
            )
            stress = (await s.scalars(select(OptionsStrategyStressTestRow).where(OptionsStrategyStressTestRow.version_id == v.id)
                                      .order_by(OptionsStrategyStressTestRow.run_at.desc()).limit(8))).all()  # fmt: skip
            out["stress"] = [{"kind": x.kind, "passed": x.passed, "summary": x.summary} for x in stress]
            regimes = (
                await s.scalars(
                    select(OptionsStrategyRegimeRow).where(OptionsStrategyRegimeRow.version_id == v.id)
                )
            ).all()
            out["regimes"] = [{"regime": r.regime, "source": r.source, "trades": r.trades, "expectancy_on_risk": r.expectancy,
                               "win_rate": r.win_rate} for r in regimes]  # fmt: skip
            score = await s.scalar(select(OptionsStrategyScoreRow).where(OptionsStrategyScoreRow.version_id == v.id)
                                   .order_by(OptionsStrategyScoreRow.scored_at.desc()).limit(1))  # fmt: skip
            out["score"] = (
                {"dimensions": score.dimensions, "eligible": score.eligible, "blocking": score.reasons}
                if score
                else None
            )
            decays = (await s.scalars(select(OptionsStrategyDecayRow).where(OptionsStrategyDecayRow.version_id == v.id)
                                      .order_by(OptionsStrategyDecayRow.at.desc()).limit(5))).all()  # fmt: skip
            out["decay"] = [
                {"at": d.at.isoformat(), "status": d.status, "reasons": d.reasons} for d in decays
            ]
            if v.source_id:
                src = await s.get(OptionsStrategySourceRow, v.source_id)
                claim = await s.scalar(
                    select(OptionsStrategyClaimRow).where(OptionsStrategyClaimRow.source_id == v.source_id)
                )
                out["source"] = _source_out(src, claim) if src else None
            out["lineage"] = {"parent_id": v.parent_id, "second_parent_id": v.second_parent_id,
                              "generation": v.generation, "origin": v.origin, "experiment_id": v.experiment_id}  # fmt: skip
            return out

    async def research_library(self) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            srcs = (
                await s.scalars(select(OptionsStrategySourceRow).order_by(OptionsStrategySourceRow.id))
            ).all()
            out = []
            for src in srcs:
                claim = await s.scalar(
                    select(OptionsStrategyClaimRow).where(OptionsStrategyClaimRow.source_id == src.id)
                )
                out.append(_source_out(src, claim))
            return out

    async def experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(OptionsExperimentRow).order_by(OptionsExperimentRow.created_at.desc()).limit(limit)
                )
            ).all()
            out = []
            for e in rows:
                h = await s.get(OptionsHypothesisRow, e.hypothesis_id) if e.hypothesis_id else None
                out.append({"id": e.id, "hypothesis": h.statement if h else None, "status": e.status,
                            "priority": e.priority, "parent_version_id": e.parent_version_id,
                            "child_version_id": e.child_version_id, "changes": e.parameter_changes,
                            "decision": e.decision, "metrics": e.metrics, "created_at": e.created_at.isoformat(),
                            "finished_at": e.finished_at.isoformat() if e.finished_at else None})  # fmt: skip
            return out

    async def knowledge(self, limit: int = 300) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(OptionsKnowledgeEdgeRow)
                    .order_by(OptionsKnowledgeEdgeRow.updated_at.desc())
                    .limit(limit)
                )
            ).all()
            return [{"src": f"{r.src_type}:{r.src_id}", "relation": r.relation, "dst": f"{r.dst_type}:{r.dst_id}",
                     "weight": r.weight} for r in rows]  # fmt: skip

    async def counts(self) -> dict[str, Any]:
        async with self._db.session() as s:
            by_stage = dict((await s.execute(select(OptionsStrategyVersionRow.stage, func.count())
                                             .group_by(OptionsStrategyVersionRow.stage))).all())  # fmt: skip
            queued = await s.scalar(
                select(func.count())
                .select_from(OptionsExperimentRow)
                .where(OptionsExperimentRow.status == "QUEUED")
            )
            gens = await s.scalar(select(func.max(OptionsGenerationRunRow.generation)))
        return {"by_stage": by_stage, "experiments_queued": int(queued or 0), "generation": gens or 0,
                "last_run": self.last_run}  # fmt: skip

    async def eligible_versions(self) -> list[dict[str, Any]]:
        """Versions the Options Brain may use: shadow at PAPER_SHADOW (and exploration), paper at
        PAPER_ACTIVE/PROVEN."""
        async with self._db.session() as s:
            rows = (await s.scalars(select(OptionsStrategyVersionRow).where(
                OptionsStrategyVersionRow.stage.in_([x.value for x in promotion.SHADOW_STAGES])))).all()  # fmt: skip
            out = []
            for v in rows:
                g = await s.get(OptionsStrategyGenomeRow, v.genome_id)
                if g is None:
                    continue
                out.append({"version_id": v.id, "key": f"{v.strategy_key}@v{v.version}", "stage": v.stage,
                            "genome": g.params, "family": g.family, "direction": g.direction,
                            "expected_ror": ((v.stage_history[-1].get("evidence") or {}).get("latest") or {}).get("validation_ror")
                            if v.stage_history else None})  # fmt: skip
            return out


def _uses(g: Genome, feature: str) -> bool:
    if feature == "iv_rank":
        return (
            g.iv_rank_min is not None or g.iv_rank_max is not None or g.entry_signal in ("iv_high", "iv_low")
        )
    if feature == "iv_rv":
        return g.iv_rv_min is not None or g.iv_rv_max is not None
    if feature == "trend":
        return g.entry_signal.startswith(("trend", "momentum", "breakout"))
    return g.event_filter != "ignore"


def _findings(v: OptionsStrategyVersionRow, regimes: Mapping[str, float]) -> list[str]:
    """Recognisable failure modes from a version's evidence (the experiment generator's inputs)."""
    ev = ((v.stage_history[-1].get("evidence") or {}).get("latest") or {}) if v.stage_history else {}
    ror = ev.get("ror_by_model") or {}
    out = []
    if (ror.get("OPTIMISTIC") or 0) > 0 and (ror.get("REALISTIC") or 0) <= 0:
        out.append("execution")
    if (regimes.get("LOW_IV") or 0) < 0:
        out.append("low_iv")
    if (regimes.get("HIGH_IV") or 0) < 0:
        out.append("high_iv")
    for name in ("TRENDING_DOWN", "TRENDING_UP", "PANIC", "MEAN_REVERTING"):
        if (regimes.get(name) or 0) < 0:
            out.append(f"regime:{name}")
    return out


def _version_out(v: OptionsStrategyVersionRow, g: OptionsStrategyGenomeRow | None, m: Mapping[str, Any],
                 live: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    last = v.stage_history[-1] if v.stage_history else {}
    return {
        "id": v.id, "key": v.strategy_key, "version": v.version, "name": v.name, "stage": v.stage, "role": v.role,
        "origin": v.origin, "generation": v.generation, "reason": v.reason,
        "family": g.family if g else None, "direction": g.direction if g else None,
        "parameters": g.params if g else None, "parameter_count": g.parameter_count if g else None,
        "backtest": dict(m), "shadow": {k: x for k, x in (live.get("shadow") or {}).items() if k != "returns"},
        "paper": {k: x for k, x in (live.get("paper") or {}).items() if k != "returns"},
        "next_gate": last.get("next_gate"), "evidence": (last.get("evidence") or {}),
        "stage_history": [{k: x for k, x in h.items() if k != "evidence"} for h in v.stage_history],
        "created_at": v.created_at.isoformat(), "stage_changed_at": v.stage_changed_at.isoformat(),
        "label": "evidence below PAPER_SHADOW is model-priced (no historical option quotes)",
    }  # fmt: skip


def _source_out(src: OptionsStrategySourceRow, claim: OptionsStrategyClaimRow | None) -> dict[str, Any]:
    return {"key": src.source_key, "title": src.title, "author": src.author,
            "published": src.publication_date.isoformat() if src.publication_date else None,
            "type": src.source_type, "quality": src.quality, "reference": src.reference, "market": src.market,
            "period": src.time_period, "limitations": src.limitations, "evidence_grade": src.evidence_grade,
            "status": src.status, "claim": claim.claim if claim else None,
            "claim_status": claim.status if claim else None, "claim_test": claim.test if claim else None,
            "note": "a source's claim is a hypothesis QuantPulse tests, never a fact it assumes"}  # fmt: skip
