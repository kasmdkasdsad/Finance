"""The strategy lab service: propose → version → validate (backtest, walk-forward, overfitting checks,
stress tests) → paper (shadow) tracking → compare → promote only if validated.

* **Data** — daily bars for the trading universe's most liquid names over ``QP_BRAIN_LAB_HISTORY_DAYS``
  (warehouse first, then vendors), completed sessions only. Synthetic prices are refused: the lab says it
  cannot test rather than test on invented data. The universe is today's candidates, so results carry
  survivorship bias (names that left the index are missing); the report says so.
* **Proposals** — the brain proposes the template catalogue (:mod:`.spec`); people can add versions with
  different parameters. A version never changes.
* **Validation** — :func:`.validation.validate`; the verdict is *validated* only if every gate passes.
* **Paper** — a validated version can be paper-tracked: on its own schedule the lab records the portfolio it
  would hold (no orders — a shadow portfolio) and measures, from real closes afterwards, how it did against
  the benchmark.
* **Promotion** — only a person can promote (``POST /brain/lab/.../promote``), and only a version that was
  validated and has been paper-tracked long enough without falling short. A promoted strategy becomes a
  voice in the brain's consensus (the ``strategy_lab`` agent), graded like any other agent. Nothing in the
  lab can place an order.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError, NotFoundError
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import BrainStrategyRow, BrainStrategyRunRow
from quantpulse.db.session import Database
from quantpulse.domain.features import Panel, compute_features
from quantpulse.schemas.common import DataStatus
from quantpulse.services.market import MarketService
from quantpulse.services.trading_data import TradingDataLoader

from .backtest import scores
from .spec import TEMPLATES, StrategySpec, from_template
from .validation import Thresholds, validate

STATUSES = ("proposed", "validated", "rejected", "paper", "promoted", "retired")
SIGNAL_KEY = "promoted_signals"


def _row_out(r: BrainStrategyRow) -> dict[str, Any]:
    return {
        "id": r.id,
        "strategy_id": r.strategy_id,
        "version": r.version,
        "key": f"{r.strategy_id}@v{r.version}",
        "name": r.name,
        "spec": r.spec,
        "status": r.status,
        "source": r.source,
        "parent_version": r.parent_version,
        "validation": r.validation,
        "paper": r.paper,
        "decided_by": r.decided_by,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "promoted_at": r.promoted_at,
    }


class StrategyLab:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        db: Database,
        market: MarketService | None,
        data: TradingDataLoader,
        state: Any,
    ) -> None:
        self._s = settings
        self._clock = clock
        self._db = db
        self._market = market
        self._data = data
        self._state = state  # the brain store (get_state / set_state)

    @property
    def thresholds(self) -> Thresholds:
        return Thresholds(min_days=self._s.brain_lab_min_oos_sessions)

    # ------------------------------------------------------------------ data
    async def panel(self) -> tuple[Panel, dict[str, Any]]:
        if self._market is None:
            raise DomainError("the strategy lab needs the market service")
        bench = self._s.benchmark_symbol
        pool = [s for s in await self._data.candidates() if s not in self._s.trading_etfs]
        got = await self._market.daily_panel([*dict.fromkeys([*pool, bench])], self._s.brain_lab_history_days)
        if bench not in got.frames or got.status(bench) is DataStatus.SYNTHETIC:
            raise DomainError("no real benchmark history: the lab does not test on synthetic prices")
        today = pd.Timestamp(self._clock.now().astimezone(NEW_YORK).date())
        index = got.frames[bench].index
        index = index[index < today]
        real = {
            s: f for s, f in got.frames.items() if s != bench and got.status(s) is not DataStatus.SYNTHETIC
        }
        close = pd.DataFrame({s: f["close"].reindex(index) for s, f in real.items()}, index=index)
        volume = pd.DataFrame({s: f["volume"].reindex(index) for s, f in real.items()}, index=index)
        coverage = close.notna().mean()
        close = close.loc[:, coverage >= 0.6]
        adv = (close * volume[close.columns]).mean().sort_values(ascending=False)
        keep = list(adv.index[: self._s.brain_lab_universe_size])
        close = close[keep]
        high = pd.DataFrame({s: real[s]["high"].reindex(index) for s in keep}, index=index)
        low = pd.DataFrame({s: real[s]["low"].reindex(index) for s in keep}, index=index)
        vol = volume[keep]
        panel = Panel(close, high, low, vol, got.frames[bench]["close"].reindex(index))
        meta = {
            "symbols": len(keep),
            "sessions": len(index),
            "from": str(index[0].date()) if len(index) else None,
            "to": str(index[-1].date()) if len(index) else None,
            "excluded_synthetic": sorted(s for s in got.frames if got.status(s) is DataStatus.SYNTHETIC),
            "survivorship_bias": "the universe is today's candidates: names that left it are missing",
        }
        return panel, meta

    # ------------------------------------------------------------------ catalogue
    async def strategies(self, status: str | None = None) -> list[dict[str, Any]]:
        stmt = select(BrainStrategyRow).order_by(BrainStrategyRow.strategy_id, BrainStrategyRow.version)
        if status:
            stmt = stmt.where(BrainStrategyRow.status == status)
        async with self._db.session() as s:
            return [_row_out(r) for r in (await s.scalars(stmt)).all()]

    async def _row(self, s: Any, strategy_id: str, version: int) -> BrainStrategyRow:
        row = (
            await s.scalars(
                select(BrainStrategyRow).where(
                    BrainStrategyRow.strategy_id == strategy_id, BrainStrategyRow.version == version
                )
            )
        ).first()
        if row is None:
            raise NotFoundError(f"strategy {strategy_id}@v{version} not found")
        return row

    async def get(self, strategy_id: str, version: int) -> dict[str, Any]:
        async with self._db.session() as s:
            row = await self._row(s, strategy_id, version)
            runs = (
                await s.scalars(
                    select(BrainStrategyRunRow)
                    .where(BrainStrategyRunRow.strategy_row_id == row.id)
                    .order_by(BrainStrategyRunRow.id.desc())
                    .limit(50)
                )
            ).all()
            out = _row_out(row)
            out["runs"] = [
                {"id": r.id, "kind": r.kind, "result": r.result, "created_at": r.created_at} for r in runs
            ]
        return out

    async def create(self, spec: StrategySpec, source: str, parent: int | None = None) -> dict[str, Any]:
        now = self._clock.now()
        async with self._db.session() as s:
            exists = (
                await s.scalars(
                    select(BrainStrategyRow).where(
                        BrainStrategyRow.strategy_id == spec.id, BrainStrategyRow.version == spec.version
                    )
                )
            ).first()
            if exists is not None:
                raise DomainError(f"{spec.key} already exists: versions never change, create a new one")
            row = BrainStrategyRow(
                strategy_id=spec.id, version=spec.version, name=spec.name, spec=spec.to_dict(), status="proposed",
                source=source, parent_version=parent, validation={}, paper={}, created_at=now, updated_at=now,
            )  # fmt: skip
            s.add(row)
            await s.flush()
            return _row_out(row)

    async def new_version(self, template: str, source: str = "user", **overrides: Any) -> dict[str, Any]:
        """A new version of ``template`` (the next free number) with the given parameter overrides."""
        async with self._db.session() as s:
            latest = await s.scalar(
                select(func.max(BrainStrategyRow.version)).where(BrainStrategyRow.strategy_id == template)
            )
        spec = from_template(template, version=int(latest or 0) + 1, **overrides)
        return await self.create(spec, source, parent=int(latest) if latest else None)

    async def propose(self) -> list[dict[str, Any]]:
        """The brain proposes every catalogue template it has not tried yet (version 1)."""
        have = {r["strategy_id"] for r in await self.strategies()}
        return [await self.create(from_template(t), "template") for t in TEMPLATES if t not in have]

    # ------------------------------------------------------------------ validation
    async def validate(self, strategy_id: str, version: int) -> dict[str, Any]:
        async with self._db.session() as s:
            row = await self._row(s, strategy_id, version)
            if row.status in ("retired",):
                raise DomainError(f"{strategy_id}@v{version} is retired")
            spec = StrategySpec.from_dict(row.spec)
        panel, meta = await self.panel()
        features = compute_features(panel)
        report = validate(spec, features, panel.close, panel.benchmark, self.thresholds)
        report["data"] = meta
        summary = {
            "verdict": report["verdict"],
            "gates": report["gates"],
            "backtest": report["backtest"],
            "walk_forward": {k: v for k, v in report["walk_forward"].items() if k != "folds"},
            "random_percentile": report["random_percentile"],
            "stress": {k: v for k, v in report["stress"].items() if k != "windows"},
            "data": meta,
            "at": self._clock.now().isoformat(),
        }
        now = self._clock.now()
        async with self._db.session() as s:
            row = await self._row(s, strategy_id, version)
            row.validation = summary
            if row.status in ("proposed", "validated", "rejected"):
                row.status = report["verdict"]
            row.updated_at = now
            s.add(
                BrainStrategyRunRow(
                    strategy_row_id=row.id, kind="validation", result=_jsonable(report), created_at=now
                )
            )
        return summary

    async def validate_pending(self, limit: int = 3) -> list[dict[str, Any]]:
        out = []
        for r in (await self.strategies("proposed"))[:limit]:
            out.append({"key": r["key"], **(await self.validate(r["strategy_id"], r["version"]))})
        return out

    # ------------------------------------------------------------------ paper, promotion
    async def set_status(self, strategy_id: str, version: int, status: str, by: str) -> dict[str, Any]:
        if status not in STATUSES:
            raise DomainError(f"unknown status {status!r}")
        now = self._clock.now()
        async with self._db.session() as s:
            row = await self._row(s, strategy_id, version)
            if status == "paper" and row.status != "validated":
                raise DomainError("only a validated version can be paper-tracked (validate it first)")
            if status == "promoted":
                ok, why = self.promotable(row)
                if not ok:
                    raise DomainError(f"not promotable: {why}")
                row.promoted_at = now
            row.status, row.decided_by, row.updated_at = status, by, now
            if status == "paper":
                row.paper = {"started": now.isoformat(), "entries": 0, "sessions": 0}
            out = _row_out(row)
        await self.publish_signals()
        return out

    def promotable(self, row: BrainStrategyRow) -> tuple[bool, str]:
        if row.status != "paper":
            return False, f"status is {row.status}: a version must be validated and then paper-tracked"
        if (row.validation or {}).get("verdict") != "validated":
            return False, "its last validation did not pass every gate"
        paper = row.paper or {}
        sessions = int(paper.get("sessions") or 0)
        if sessions < self._s.brain_lab_paper_days:
            return False, f"only {sessions} paper sessions (needs {self._s.brain_lab_paper_days})"
        excess = paper.get("excess_return")
        if excess is None or excess < -0.02:
            return False, f"paper excess return {excess} fell short of the benchmark"
        return True, "validated, and paper performance held up"

    async def paper_update(self) -> dict[str, Any]:
        """Record the shadow portfolio of every paper/promoted version when its rebalance is due and measure
        how the recorded portfolios did since, from real closes (no orders are ever placed)."""
        rows = [r for r in await self.strategies() if r["status"] in ("paper", "promoted")]
        if not rows:
            await self.publish_signals()
            return {"tracked": 0}
        panel, _ = await self.panel()
        features = compute_features(panel)
        close, bench = panel.close, panel.benchmark
        now = self._clock.now()
        updated = 0
        signals: dict[str, Any] = {}
        for r in rows:
            spec = StrategySpec.from_dict(r["spec"])
            score = scores(spec, features).iloc[-1].dropna().sort_values(ascending=False)
            async with self._db.session() as s:
                row = await self._row(s, spec.id, spec.version)
                entries = (
                    await s.scalars(
                        select(BrainStrategyRunRow)
                        .where(
                            BrainStrategyRunRow.strategy_row_id == row.id, BrainStrategyRunRow.kind == "paper"
                        )
                        .order_by(BrainStrategyRunRow.id)
                    )
                ).all()
                last_date = pd.Timestamp(entries[-1].result["date"]) if entries else None
                today = close.index[-1]
                due = last_date is None or (close.index > last_date).sum() >= spec.rebalance_days
                recorded = [e.result for e in entries]
                if due and len(score) >= spec.top_n:
                    holdings = [str(x) for x in score.index[: spec.top_n]]
                    result = {
                        "date": str(today.date()),
                        "holdings": holdings,
                        "prices": {h: float(close[h].iloc[-1]) for h in holdings},
                        "benchmark": float(bench.iloc[-1]),
                    }
                    s.add(
                        BrainStrategyRunRow(
                            strategy_row_id=row.id, kind="paper", result=result, created_at=now
                        )
                    )
                    recorded.append(result)
                    updated += 1
                row.paper = {**(row.paper or {}), **_paper_performance(recorded, close, bench)}
                row.updated_at = now
            if r["status"] == "promoted":
                n = len(score)
                signals[r["key"]] = {
                    "top": [str(x) for x in score.index[: spec.top_n]],
                    "bottom": [str(x) for x in score.index[max(n - spec.top_n, 0) :]],
                    "horizon": spec.rebalance_days,
                    "name": spec.name,
                }
        await self._state.set_state(SIGNAL_KEY, signals, now)
        return {"tracked": len(rows), "rebalanced": updated}

    async def publish_signals(self) -> None:
        promoted = await self.strategies("promoted")
        if not promoted:
            await self._state.set_state(SIGNAL_KEY, {}, self._clock.now())

    async def compare(self, keys: list[str]) -> list[dict[str, Any]]:
        rows = {r["key"]: r for r in await self.strategies()}
        out = []
        for k in keys:
            r = rows.get(k)
            if r is None:
                raise NotFoundError(f"strategy {k} not found")
            v = r["validation"] or {}
            wf = v.get("walk_forward") or {}
            bt = v.get("backtest") or {}
            out.append(
                {
                    "key": k,
                    "status": r["status"],
                    "verdict": v.get("verdict"),
                    "gates_passed": sum(1 for g in v.get("gates", []) if g["passed"]),
                    "gates": len(v.get("gates", [])),
                    "backtest_sharpe": bt.get("sharpe"),
                    "oos_active_sharpe": wf.get("oos_active_sharpe"),
                    "dsr": wf.get("dsr"),
                    "degradation": wf.get("degradation"),
                    "max_drawdown": bt.get("max_drawdown"),
                    "paper_excess": (r["paper"] or {}).get("excess_return"),
                    "params": StrategySpec.from_dict(r["spec"]).params(),
                }
            )
        return out


def _paper_performance(
    entries: list[dict[str, Any]], close: pd.DataFrame, bench: pd.Series
) -> dict[str, Any]:
    """Chain each recorded shadow portfolio's equal-weight return (to the next entry, or the latest close)."""
    if not entries:
        return {"entries": 0, "sessions": 0}
    total, bench_total, sessions = 1.0, 1.0, 0
    for i, e in enumerate(entries):
        start = pd.Timestamp(e["date"])
        end = pd.Timestamp(entries[i + 1]["date"]) if i + 1 < len(entries) else close.index[-1]
        if end <= start:
            continue
        rets = []
        for h in e["holdings"]:
            if h in close.columns and pd.notna(close.at[end, h]) and e["prices"].get(h):
                rets.append(float(close.at[end, h]) / e["prices"][h] - 1)
        if rets:
            total *= 1 + float(np.mean(rets))
        bench_total *= float(bench.at[end]) / e["benchmark"]
        sessions += int(((close.index > start) & (close.index <= end)).sum())
    return {
        "entries": len(entries),
        "sessions": sessions,
        "return": round(total - 1, 5),
        "benchmark_return": round(bench_total - 1, 5),
        "excess_return": round(total - bench_total, 5),
        "since": entries[0]["date"],
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not isinstance(v, pd.Series)}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, float | np.floating):
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def next_spec(spec: StrategySpec, **changes: Any) -> StrategySpec:
    return replace(spec, version=spec.version + 1, **changes)
