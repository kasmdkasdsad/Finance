"""A self-learning paper-trading agent built on the daily-picks factor model.

Decision
    Score the universe with the six screener factors using the agent's *current* factor weights, hold the
    top ``top_k`` names with a positive composite in equal weight (capped at ``max_position``), keep the
    rest in cash.

Learning (online, multiplicative weights on information coefficients)
    At every decision the agent remembers each stock's factor z-scores and price. At the next decision it
    measures, per factor, the Spearman rank correlation (information coefficient, IC) between those
    scores and the returns realised since. Weights update as ``w_f ← w_f · exp(η · IC_f)``, are shrunk a
    little towards the original prior (``prior_shrink``), floored at ``weight_floor`` so no factor is ever
    switched off by noise, and renormalised. Factors that keep ranking future winners gain influence.

Walk-forward training
    :func:`walk_forward` replays history one trading day at a time with no look-ahead: signals use closes
    up to day *t*, orders execute at the close of day *t+1* (with slippage and fees), and learning at a
    decision only uses returns that were already observable.

This is a transparent, deliberately simple learner. Daily/weekly ICs are noisy; learned weights describe
what worked in the replayed window and are not a forecast.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

from quantpulse.core.errors import DomainError
from quantpulse.domain import screener
from quantpulse.domain.paper_broker import ExecutionModel, Fill, PaperBook
from quantpulse.quant import risk

FACTORS: tuple[str, ...] = tuple(screener.WEIGHTS)
PRIOR: dict[str, float] = dict(screener.WEIGHTS)
IC_EMA_ALPHA = 0.3


@dataclass(frozen=True, slots=True)
class AgentConfig:
    top_k: int = 5
    max_position: float = 0.25
    cash_buffer: float = 0.02
    learning_rate: float = 0.5
    prior_shrink: float = 0.05
    weight_floor: float = 0.02
    min_trade_value: float = 50.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise DomainError("top_k must be >= 1")
        if not 0 < self.max_position <= 1:
            raise DomainError("max_position must be in (0, 1]")
        if not 0 <= self.cash_buffer < 1:
            raise DomainError("cash_buffer must be in [0, 1)")
        if self.learning_rate < 0 or not 0 <= self.prior_shrink <= 1:
            raise DomainError("learning_rate must be >= 0 and prior_shrink in [0, 1]")
        if not 0 <= self.weight_floor < 1 / len(FACTORS):
            raise DomainError(f"weight_floor must be in [0, {1 / len(FACTORS):.3f})")


def normalise(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise DomainError("weights must have a positive sum")
    return {f: weights[f] / total for f in FACTORS}


@dataclass
class LearningState:
    weights: dict[str, float] = field(default_factory=lambda: normalise(PRIOR))
    ic_ema: dict[str, float] = field(default_factory=dict)
    periods_learned: int = 0
    last_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    last_decision_on: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "ic_ema": self.ic_ema,
            "periods_learned": self.periods_learned,
            "last_scores": self.last_scores,
            "last_prices": self.last_prices,
            "last_decision_on": self.last_decision_on,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> LearningState:
        if not data:
            return cls()
        weights = data.get("weights") or PRIOR
        return cls(
            weights=normalise({f: float(weights.get(f, PRIOR[f])) for f in FACTORS}),
            ic_ema={k: float(v) for k, v in (data.get("ic_ema") or {}).items()},
            periods_learned=int(data.get("periods_learned") or 0),
            last_scores={
                s: {f: float(z) for f, z in v.items()} for s, v in (data.get("last_scores") or {}).items()
            },
            last_prices={s: float(p) for s, p in (data.get("last_prices") or {}).items()},
            last_decision_on=data.get("last_decision_on"),
        )


@dataclass(frozen=True, slots=True)
class Lesson:
    factor: str
    ic: float
    observations: int
    weight_before: float
    weight_after: float


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y) or np.std(x) == 0 or np.std(y) == 0:
        return None
    rho = float(spearmanr(x, y).statistic)
    return rho if math.isfinite(rho) else None


def learn(state: LearningState, config: AgentConfig, prices: Mapping[str, float]) -> list[Lesson]:
    """Update ``state.weights`` from the returns realised since the previous decision (in place)."""
    if not state.last_scores or not state.last_prices:
        return []
    returns = {
        s: prices[s] / state.last_prices[s] - 1.0
        for s in state.last_scores
        if s in prices and state.last_prices.get(s, 0) > 0 and prices[s] > 0
    }
    ics: dict[str, tuple[float, int]] = {}
    for f in FACTORS:
        pairs = [(state.last_scores[s][f], r) for s, r in returns.items() if f in state.last_scores[s]]
        ic = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            ics[f] = (ic, len(pairs))
    if not ics:
        return []
    before = dict(state.weights)
    raw = {f: before[f] * math.exp(config.learning_rate * ics.get(f, (0.0, 0))[0]) for f in FACTORS}
    raw = normalise(raw)
    prior = normalise(PRIOR)
    shrunk = {f: (1 - config.prior_shrink) * raw[f] + config.prior_shrink * prior[f] for f in FACTORS}
    floored = normalise({f: max(v, config.weight_floor) for f, v in shrunk.items()})
    state.weights = floored
    for f, (ic, _) in ics.items():
        prev = state.ic_ema.get(f)
        state.ic_ema[f] = ic if prev is None else (1 - IC_EMA_ALPHA) * prev + IC_EMA_ALPHA * ic
    state.periods_learned += 1
    return [Lesson(f, ic, n, before[f], floored[f]) for f, (ic, n) in ics.items()]


@dataclass(frozen=True, slots=True)
class Decision:
    targets: dict[str, float]
    ranked: list[screener.ScreenResult]
    scores: dict[str, dict[str, float]]


def decide(
    factors: Mapping[str, screener.RawFactors], weights: Mapping[str, float], config: AgentConfig
) -> Decision:
    if not factors:
        return Decision({}, [], {})
    ranked = screener.screen(factors, weights)
    chosen = [r for r in ranked if r.composite > 0][: config.top_k]
    each = min(config.max_position, 1.0 / config.top_k)
    return Decision(
        targets={r.symbol: each for r in chosen},
        ranked=ranked,
        scores={r.symbol: dict(r.z) for r in ranked},
    )


def factor_universe(
    closes: Mapping[str, Sequence[float] | NDArray[np.float64]],
) -> tuple[dict[str, screener.RawFactors], dict[str, str]]:
    factors: dict[str, screener.RawFactors] = {}
    skipped: dict[str, str] = {}
    for symbol, series in closes.items():
        try:
            factors[symbol] = screener.compute_factors(series)
        except ValueError as exc:
            skipped[symbol] = str(exc)
    return factors, skipped


# ----------------------------------------------------------------------------- walk-forward training
@dataclass
class BacktestResult:
    dates: list[date]
    equity: list[float]
    benchmark: list[float]
    fills: list[tuple[date, Fill]]
    weights_history: list[tuple[date, dict[str, float]]]
    ic_history: list[tuple[date, dict[str, float]]]
    state: LearningState
    decisions: int
    turnover: float

    def metrics(self, risk_free: float = 0.0) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        for name, curve in (("strategy", self.equity), ("benchmark", self.benchmark)):
            series = np.asarray(curve, dtype=float)
            rets = series[1:] / series[:-1] - 1.0
            out[name] = {
                "total_return": float(series[-1] / series[0] - 1.0),
                "annual_return": risk.annualized_return(rets) if rets.size else None,
                "annual_volatility": risk.annualized_volatility(rets) if rets.size > 1 else None,
                "sharpe": risk.sharpe_ratio(rets, risk_free) if rets.size > 1 else None,
                "max_drawdown": risk.max_drawdown(rets) if rets.size else None,
            }
        return out


def walk_forward(
    closes: pd.DataFrame,
    benchmark: pd.Series,
    config: AgentConfig,
    model: ExecutionModel,
    *,
    start_cash: float = 100_000.0,
    rebalance_every: int = 5,
    initial_weights: Mapping[str, float] | None = None,
    warmup: int = screener.MIN_BARS,
) -> BacktestResult:
    """Replay ``closes`` (rows = trading days, columns = symbols, no NaNs) without look-ahead."""
    if rebalance_every < 1:
        raise DomainError("rebalance_every must be >= 1")
    if closes.isna().to_numpy().any():
        raise DomainError("closes must be aligned with no missing values")
    n = len(closes)
    if n < warmup + rebalance_every + 1:
        raise DomainError(f"need at least {warmup + rebalance_every + 1} aligned trading days, got {n}")
    bench = benchmark.reindex(closes.index)
    if bench.isna().any():
        raise DomainError("benchmark must cover every trading day in the replay")

    book = PaperBook(cash=start_cash)
    state = LearningState(weights=normalise(initial_weights or PRIOR))
    arrays = {s: closes[s].to_numpy(dtype=float) for s in closes.columns}
    pending: Decision | None = None
    dates: list[date] = []
    equity: list[float] = []
    fills: list[tuple[date, Fill]] = []
    weights_history: list[tuple[date, dict[str, float]]] = []
    ic_history: list[tuple[date, dict[str, float]]] = []
    traded = 0.0
    decisions = 0
    first = warmup - 1
    for i in range(first, n):
        day = closes.index[i].date() if hasattr(closes.index[i], "date") else closes.index[i]
        prices = {s: float(arrays[s][i]) for s in arrays}
        if pending is not None:  # yesterday's signal executes at today's close
            executed = book.rebalance(
                prices,
                pending.targets,
                model,
                cash_buffer=config.cash_buffer,
                min_trade_value=config.min_trade_value,
            )
            fills.extend((day, f) for f in executed)
            traded += sum(f.notional for f in executed)
            pending = None
        if (i - first) % rebalance_every == 0 and i < n - 1:
            lessons = learn(state, config, prices)
            if lessons:
                ic_history.append((day, {lesson.factor: lesson.ic for lesson in lessons}))
            factors, _ = factor_universe({s: a[: i + 1] for s, a in arrays.items()})
            pending = decide(factors, state.weights, config)
            state.last_scores = pending.scores
            state.last_prices = prices
            state.last_decision_on = day.isoformat()
            weights_history.append((day, dict(state.weights)))
            decisions += 1
        dates.append(day)
        equity.append(book.equity(prices))

    bench_values = bench.to_numpy(dtype=float)[first:]
    bench_curve = [float(v / bench_values[0] * start_cash) for v in bench_values]
    avg_equity = float(np.mean(equity)) if equity else start_cash
    return BacktestResult(
        dates=dates,
        equity=equity,
        benchmark=bench_curve,
        fills=fills,
        weights_history=weights_history,
        ic_history=ic_history,
        state=state,
        decisions=decisions,
        turnover=traded / avg_equity if avg_equity > 0 else 0.0,
    )
