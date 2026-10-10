import math

import numpy as np
import pandas as pd
import pytest

from quantpulse.core.errors import DomainError
from quantpulse.domain.paper_broker import ExecutionModel
from quantpulse.domain.screener import RawFactors
from quantpulse.domain.trading_agent import (
    FACTORS,
    PRIOR,
    AgentConfig,
    LearningState,
    decide,
    learn,
    normalise,
    walk_forward,
)

CFG = AgentConfig()


def test_learning_rewards_predictive_factors_and_penalises_contrary_ones():
    symbols = [f"S{i}" for i in range(10)]
    returns = np.linspace(-0.05, 0.05, 10)
    state = LearningState(
        last_scores={
            s: {"momentum_3m": float(i), "reversal": float(-i), "trend": float((i * 7) % 10)}
            for i, s in enumerate(symbols)
        },
        last_prices=dict.fromkeys(symbols, 100.0),
    )
    prices = {s: 100.0 * (1 + r) for s, r in zip(symbols, returns, strict=True)}
    lessons = {lesson.factor: lesson for lesson in learn(state, CFG, prices)}
    assert lessons["momentum_3m"].ic == pytest.approx(1.0)
    assert lessons["reversal"].ic == pytest.approx(-1.0)
    assert state.weights["momentum_3m"] > PRIOR["momentum_3m"] / sum(PRIOR.values())
    assert state.weights["reversal"] < PRIOR["reversal"] / sum(PRIOR.values())
    assert sum(state.weights.values()) == pytest.approx(1.0)
    assert min(state.weights.values()) >= CFG.weight_floor / (1 + len(FACTORS) * CFG.weight_floor)
    assert state.periods_learned == 1 and state.ic_ema["momentum_3m"] == pytest.approx(1.0)


def test_learning_needs_history_and_rate_zero_only_shrinks_to_prior():
    assert learn(LearningState(), CFG, {"A": 1.0}) == []
    state = LearningState(
        last_scores={s: {"momentum_3m": float(i)} for i, s in enumerate("ABCDE")},
        last_prices=dict.fromkeys("ABCDE", 10.0),
    )
    learn(state, AgentConfig(learning_rate=0.0), {s: 10.0 + i for i, s in enumerate("ABCDE")})
    assert state.weights == pytest.approx(normalise(PRIOR))


def test_state_round_trip():
    state = LearningState(
        periods_learned=3,
        last_scores={"A": {"trend": 1.5}},
        last_prices={"A": 12.0},
        last_decision_on="2026-09-24",
    )
    state.weights = normalise({f: i + 1.0 for i, f in enumerate(FACTORS)})
    again = LearningState.from_json(state.to_json())
    a, b = again.to_json(), state.to_json()
    assert a.pop("weights") == pytest.approx(b.pop("weights"))
    assert a == b
    assert LearningState.from_json(None).weights == pytest.approx(normalise(PRIOR))


def test_decide_holds_top_k_positive_composites_equal_weight():
    factors = {
        f"S{i}": RawFactors(0.1 * i, 0.05 * i, 0.01 * i, 0.2 * i, 0.3 - 0.01 * i, 50.0) for i in range(8)
    }
    d = decide(factors, PRIOR, AgentConfig(top_k=3, max_position=0.25))
    assert list(d.targets) == ["S7", "S6", "S5"]
    assert all(w == pytest.approx(0.25) for w in d.targets.values())
    assert set(d.scores) == set(factors)
    assert decide({}, PRIOR, CFG).targets == {}


def test_config_validation():
    with pytest.raises(DomainError):
        AgentConfig(top_k=0)
    with pytest.raises(DomainError):
        AgentConfig(weight_floor=0.5)


def _trending_market(days=420, n=12, seed=4):
    """Stocks with persistent drifts: momentum/trend genuinely predict forward returns."""
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.0015, 0.0020, n)
    rets = drifts[None, :] + rng.normal(0, 0.012, (days, n))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2024-01-02", periods=days)
    closes = pd.DataFrame(prices, index=idx, columns=[f"S{i:02d}" for i in range(n)])
    bench = closes.mean(axis=1)  # equal-weight index
    return closes, bench


def test_walk_forward_learns_and_beats_equal_weight_in_trending_market():
    closes, bench = _trending_market()
    model = ExecutionModel(slippage_bps=5, commission_per_trade=0.0)
    res = walk_forward(closes, bench, AgentConfig(top_k=3, max_position=0.34), model, rebalance_every=5)
    assert res.decisions > 50 and res.fills and res.state.periods_learned == res.decisions - 1
    prior = normalise(PRIOR)
    assert res.state.weights["reversal"] < prior["reversal"]
    assert (
        res.state.weights["momentum_3m"] + res.state.weights["trend"] > prior["momentum_3m"] + prior["trend"]
    )
    m = res.metrics()
    assert m["strategy"]["total_return"] > m["benchmark"]["total_return"]
    assert len(res.dates) == len(res.equity) == len(res.benchmark)
    assert res.equity[0] == pytest.approx(100_000) and res.benchmark[0] == pytest.approx(100_000)
    assert all(math.isfinite(v) and v > 0 for v in res.equity)


def test_walk_forward_has_no_look_ahead():
    closes, bench = _trending_market(days=300)
    altered = closes.copy()
    cut = 200
    altered.iloc[cut:] = altered.iloc[cut:] * np.linspace(0.5, 2.0, altered.shape[1])  # rewrite the future
    alt_bench = altered.mean(axis=1)
    model = ExecutionModel()
    a = walk_forward(closes, bench, CFG, model, rebalance_every=3)
    b = walk_forward(altered, alt_bench, CFG, model, rebalance_every=3)
    cutoff = closes.index[cut - 1].date()
    past_a = [w for d, w in a.weights_history if d <= cutoff]
    past_b = [w for d, w in b.weights_history if d <= cutoff]
    assert past_a == past_b and len(past_a) > 10
    fills_a = [(d, f.symbol, f.quantity) for d, f in a.fills if d <= cutoff]
    fills_b = [(d, f.symbol, f.quantity) for d, f in b.fills if d <= cutoff]
    assert fills_a == fills_b


def test_walk_forward_input_validation():
    closes, bench = _trending_market(days=60)
    with pytest.raises(DomainError, match="at least"):
        walk_forward(closes, bench, CFG, ExecutionModel())
    closes, bench = _trending_market(days=120)
    broken = closes.copy()
    broken.iloc[5, 0] = np.nan
    with pytest.raises(DomainError, match="missing"):
        walk_forward(broken, bench, CFG, ExecutionModel())
