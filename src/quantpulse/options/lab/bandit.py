"""A contextual bandit for strategy selection — RESEARCH AND SIMULATION ONLY.

Linear Thompson sampling: for each strategy (arm), a Bayesian linear model of the reward (a risk-adjusted
result, penalised for drawdown, poor liquidity, slippage, tail loss and turnover) on the context (regime, IV
state, trend, liquidity, DTE, skew, event status, portfolio exposure). It is evaluated offline, by replaying
recorded outcomes (only rounds where the logged choice matches the bandit's are scored — the standard unbiased
replay estimator), against a uniform-random policy.

It never selects a live trade: :data:`EXECUTION_ENABLED` is False and there is no code path from here to an
order. An opaque learner does not get the keys to the account; if it ever proves itself in simulation, that
is a finding for a person to review, and a candidate like any other in the model registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EXECUTION_ENABLED = False  # a constant, not a setting: this module is research only


def reward(pnl_on_risk: float, *, drawdown: float = 0.0, liquidity_cost: float = 0.0, slippage: float = 0.0,
           tail_loss: float = 0.0, turnover: float = 0.0) -> float:  # fmt: skip
    """Risk-adjusted reward with explicit penalties (all per dollar at risk)."""
    return float(
        pnl_on_risk - 0.5 * abs(drawdown) - liquidity_cost - slippage - 0.5 * abs(tail_loss) - 0.01 * turnover
    )


@dataclass
class LinearThompson:
    arms: Sequence[str]
    dim: int
    noise: float = 0.5
    prior: float = 1.0
    seed: int = 0
    _A: dict[str, np.ndarray] = field(default_factory=dict)
    _b: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        for a in self.arms:
            self._A[a] = np.eye(self.dim) / self.prior
            self._b[a] = np.zeros(self.dim)

    def choose(self, x: Sequence[float]) -> str:
        v = np.asarray(x, dtype=float)
        best, score = self.arms[0], -np.inf
        for a in self.arms:
            cov = np.linalg.inv(self._A[a])
            mu = cov @ self._b[a]
            theta = self._rng.multivariate_normal(mu, self.noise**2 * cov)
            s = float(theta @ v)
            if s > score:
                best, score = a, s
        return best

    def update(self, arm: str, x: Sequence[float], r: float) -> None:
        v = np.asarray(x, dtype=float)
        self._A[arm] += np.outer(v, v)
        self._b[arm] += r * v

    def expected(self, x: Sequence[float]) -> dict[str, float]:
        v = np.asarray(x, dtype=float)
        return {a: float(np.linalg.solve(self._A[a], self._b[a]) @ v) for a in self.arms}


def replay(log: Sequence[Mapping[str, Any]], arms: Sequence[str], dim: int, seed: int = 0) -> dict[str, Any]:
    """Offline evaluation on logged rounds ``{'context': [...], 'arm': str, 'reward': float}`` (the logging
    policy chose ``arm``). Only matching rounds are scored."""
    bandit = LinearThompson(arms, dim, seed=seed)
    rng = np.random.default_rng(seed + 1)
    matched, rewards, random_rewards = 0, [], []
    for row in log:
        x, logged, r = row["context"], row["arm"], float(row["reward"])
        if bandit.choose(x) == logged:
            matched += 1
            rewards.append(r)
            bandit.update(logged, x, r)
        if arms[int(rng.integers(len(arms)))] == logged:
            random_rewards.append(r)
    return {
        "rounds": len(log),
        "matched": matched,
        "bandit_mean_reward": round(float(np.mean(rewards)), 5) if rewards else None,
        "random_mean_reward": round(float(np.mean(random_rewards)), 5) if random_rewards else None,
        "execution_enabled": EXECUTION_ENABLED,
        "note": "simulation only: never selects a live trade",
    }
