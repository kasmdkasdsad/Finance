"""The lab's evaluation of one strategy: every test the promotion gates ask for, in order, stopping at the
first gate that fails (later tests would only spend the research budget on a strategy that cannot advance).

::

    genome valid? → backtest under all five fill models → VALIDATION (enough trades; positive per dollar at
    risk under REALISTIC *and* PESSIMISTIC fills) → walk-forward + overfit risk → WALK_FORWARD → Monte Carlo,
    tail stress, baselines, the critic → PAPER_SHADOW

Pure computation on the data it is given (``LabData``) and labelled with that data's grade: on model-priced
chains (the free-data default: no historical option quotes) a pass earns shadow trading on live quotes at
most — :data:`~quantpulse.options.lab.promotion.Stage.PAPER_ACTIVE` needs real shadow evidence.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from quantpulse.options.fills import ExecutionModel
from quantpulse.options.lab import baselines, critic, overfit, scoring, stress, walkforward
from quantpulse.options.lab import montecarlo as mc
from quantpulse.options.lab.backtest import BacktestConfig, BacktestResult, run
from quantpulse.options.lab.chains import ChainSource, ModelChains
from quantpulse.options.lab.features import DayFeatures, history
from quantpulse.options.lab.genome import Genome
from quantpulse.options.lab.promotion import Evidence, Stage

MODELS = tuple(ExecutionModel)
WARMUP_SESSIONS = 260  # a year of history before the first entry (the 200-day average, IV rank)


@dataclass
class LabData:
    closes: dict[str, dict[date, float]]
    source: ChainSource
    features: dict[str, dict[date, DayFeatures]]
    grade: str  # "model" (model-priced chains) | "recorded" (stored real quotes)
    events: dict[str, list[date]] = field(default_factory=dict)

    @property
    def days(self) -> list[date]:
        return sorted({d for c in self.closes.values() for d in c})

    @property
    def underlyings(self) -> tuple[str, ...]:
        return tuple(sorted(self.closes))


def prepare(
    closes: Mapping[str, Mapping[date, float]], events: Mapping[str, Sequence[date]] | None = None
) -> LabData:
    """Model-priced chains over real underlying prices (labelled "model" everywhere they are used)."""
    c = {u: dict(v) for u, v in closes.items() if len(v) > WARMUP_SESSIONS + 60}
    src = ModelChains(c)
    ev = {u: sorted(v) for u, v in (events or {}).items()}
    feats = {u: history(v, {d: src.atm_iv(u, d) for d in v}, ev.get(u, ())) for u, v in c.items()}
    return LabData(c, src, feats, "model", ev)


def _cfg(data: LabData, model: ExecutionModel, equity: float, start: date | None = None,
         end: date | None = None) -> BacktestConfig:  # fmt: skip
    days = data.days
    return BacktestConfig(start or days[min(WARMUP_SESSIONS, len(days) - 1)], end or days[-1], data.underlyings,
                          equity=equity, model=model)  # fmt: skip


def evaluate(
    g: Genome,
    data: LabData,
    *,
    equity: float = 100_000.0,
    n_trials: int = 1,
    deadline: float | None = None,
    wf_variants: int = 2,
    random_seeds: Sequence[int] = (1, 2, 3, 4),
) -> dict[str, Any]:
    """Every test up to the first failed gate. ``n_trials`` is how many strategies the population has tried
    (the deflated Sharpe's multiple-testing correction); ``deadline`` (``time.monotonic()``) stops early,
    reported as ``incomplete``."""
    out: dict[str, Any] = {"genome": g.hash, "family": g.family, "grade": data.grade, "stopped_at": None,
                           "incomplete": False, "evidence": {}}  # fmt: skip
    ev = Evidence(genome_problems=g.problems(), family=g.family)
    out["evidence_obj"] = ev

    def over_time() -> bool:
        if deadline is not None and time.monotonic() > deadline:
            out["incomplete"] = True
            return True
        return False

    if ev.genome_problems:
        out["stopped_at"] = Stage.EXTRACTED.value
        return _finish(out, ev)
    if not data.closes:
        out["stopped_at"] = Stage.BACKTESTING.value
        out["note"] = "no underlying has enough price history to backtest"
        return _finish(out, ev)

    # 1. the backtest under every fill model (never judged on a flattering one alone)
    results: dict[str, BacktestResult] = {}
    for m in MODELS:
        results[m.value] = run(g, data.source, data.features, _cfg(data, m, equity))
        if over_time():
            break
    ev.backtests = len(results)
    ev.ror_by_model = {k: r.metrics.get("expectancy_on_risk") for k, r in results.items()}
    base = results.get(ExecutionModel.REALISTIC.value)
    out["backtests"] = {k: r.summary() for k, r in results.items()}
    if base is None:
        out["stopped_at"] = Stage.BACKTESTING.value
        return _finish(out, ev)
    ev.backtest_trades = int(base.metrics.get("trades") or 0)
    out["trades"] = base.trades
    regimes = _by(base.trades, "regime")
    out["regimes"] = regimes
    if ev.backtest_trades < 30 or not all(
        (ev.ror_by_model.get(m) or 0) > 0 for m in ("REALISTIC", "PESSIMISTIC")
    ):
        out["stopped_at"] = Stage.VALIDATION.value
        return _finish(out, ev, base=base, regimes=regimes)
    if over_time():
        return _finish(out, ev, base=base, regimes=regimes)

    # 2. walk-forward (parameters chosen on the past only) and the overfitting defences
    days = data.days
    first = days[min(WARMUP_SESSIONS, len(days) - 1)]
    span = (days[-1] - first).days
    wins = walkforward.windows(first, days[-1], train_days=max(int(span * 0.5), 180), validate_days=max(int(span * 0.15), 60),
                               test_days=max(int(span * 0.15), 60), step_days=max(int(span * 0.15), 60))  # fmt: skip
    wf = walkforward.walk_forward(g, data.source, data.features, data.underlyings, wins, n_variants=wf_variants,
                                  min_trades=15, equity=equity)  # fmt: skip
    wf_public = {k: v for k, v in wf.items() if k != "oos_trades"}
    out["walkforward"] = wf_public
    ev.validation_ror = wf["oos"].get("expectancy_on_risk")
    daily = base.metrics.get("sharpe")
    per_obs = (daily / math.sqrt(252)) if daily is not None else 0.0
    deflated = overfit.deflated_sharpe(per_obs, len(base.equity), max(n_trials, 1),
                                       base.metrics.get("daily_skew") or 0.0, base.metrics.get("daily_kurtosis") or 0.0)  # fmt: skip
    of = overfit.overfit_risk(
        parameter_count=g.parameter_count,
        trades=ev.backtest_trades,
        train_ror=wf.get("mean_train_ror"),
        test_ror=wf.get("mean_test_ror"),
        parameter_stability=wf.get("parameter_stability"),
        pnl_by_symbol=_by(base.trades, "underlying", total=True),
        pnl_by_regime=_by(base.trades, "regime", total=True),
        sharpe_annual=daily,
        win_rate=base.metrics.get("win_rate"),
        ror_by_model=ev.ror_by_model,
        variants_tried=max(n_trials, wf.get("variants_tried", 1)),
        deflated=deflated,
    )
    out["overfit"] = {**of, "deflated_sharpe": deflated}
    ev.overfit_risk = of["score"]
    ev.walkforward_passed = bool(wf["passed"]) and not of["promote_blocked"]
    if not ev.walkforward_passed or (ev.validation_ror or 0) <= 0:
        out["stopped_at"] = Stage.WALK_FORWARD.value
        return _finish(out, ev, base=base, regimes=regimes, wf=wf_public, of=of)
    if over_time():
        return _finish(out, ev, base=base, regimes=regimes, wf=wf_public, of=of)

    # 3. what shadow trading needs: the distribution, the tail, the benchmarks and the critic
    sim = mc.simulate(base.trades, equity=equity, paths=1000)
    out["montecarlo"] = sim
    ev.montecarlo_ruin = sim.get("worst_risk_of_ruin")
    tail = stress.strategy_tail(base.trades, equity)
    out["tail"] = tail
    ev.tail_passed = bool(tail["passed"])
    cmp = baselines.compare(g, base.metrics.get("expectancy_on_risk"), data.source, data.features, data.closes,
                            _cfg(data, ExecutionModel.REALISTIC, equity), random_seeds=random_seeds)  # fmt: skip
    out["baselines"] = cmp
    ev.beats_baselines = bool(cmp["beats_baselines"])
    rep = critic.critique(g, base, data.source, data.features, tail_breach=not tail["passed"])
    out["critic"] = rep
    blocking = [k for k in critic.BLOCKING if rep["attacks"].get(k, {}).get("passed") is False]
    ev.critic_survived = not blocking
    out["stopped_at"] = None
    return _finish(out, ev, base=base, regimes=regimes, wf=wf_public, of=of, sim=sim)


def _by(trades: Sequence[Mapping[str, Any]], key: str, *, total: bool = False) -> dict[str, float]:
    """P&L per dollar at risk (mean) — or total P&L — per value of ``key``."""
    groups: dict[str, list[float]] = {}
    for t in trades:
        k = str(t.get(key) or "?")
        groups.setdefault(k, []).append(
            float(t["pnl"]) if total else float(t["pnl"]) / max(float(t.get("max_loss") or 1), 1)
        )
    return {k: round(sum(v) if total else sum(v) / len(v), 5) for k, v in groups.items()}


def _finish(out: dict[str, Any], ev: Evidence, *, base: BacktestResult | None = None,
            regimes: dict[str, float] | None = None, wf: dict[str, Any] | None = None,
            of: dict[str, Any] | None = None, sim: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    out["evidence"] = dict(ev.__dict__)
    if base is not None:
        out["score"] = scoring.score(metrics=base.metrics, walkforward=wf, montecarlo=sim, overfit=of,
                                     regimes=regimes, grade=out["grade"])  # fmt: skip
        out["metrics"] = base.metrics
    return out


def t_to_p(t: float | None, n: int) -> float:
    return overfit.p_value_from_t(t, n)
