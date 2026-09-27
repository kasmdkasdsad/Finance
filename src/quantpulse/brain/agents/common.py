"""Small helpers shared by the specialist agents (no analysis lives here)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..context import BrainContext
from ..types import DataState, Evidence, Opinion, Stance, clamp, stance_of
from .base import Agent


def sgn(x: float) -> int:
    return int(x > 0) - int(x < 0)


def squash(x: float, scale: float) -> float:
    """``tanh(x / scale)``: a bounded, monotone mapping of a raw metric onto −1…1."""
    return math.tanh(x / scale)


def symbol_quality(ctx: BrainContext, symbol: str) -> DataState:
    """The data state behind an opinion built from daily data: the symbol's quote state, except that a
    missing quote still leaves valid daily bars (reported like a closed market)."""
    state = ctx.state(symbol)
    return state if state is not DataState.UNAVAILABLE else DataState.MARKET_CLOSED


def cross_section_z(frame: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Column-wise z-scores after winsorising at the 2nd/98th percentiles (robust to outliers)."""
    x = frame.apply(pd.to_numeric, errors="coerce")
    lo, hi = x.quantile(0.02), x.quantile(0.98)
    x = x.clip(lo, hi, axis=1)
    return ((x - x.mean()) / x.std().replace(0, np.nan)).clip(-clip, clip)


def agreement(parts: Iterable[float], score: float) -> float:
    """Share of the non-trivial components that point the same way as ``score`` (0 when there is none)."""
    signs = [v for v in parts if abs(v) > 0.05]
    if not signs or not score:
        return 0.0
    return sum(1 for v in signs if (v > 0) == (score > 0)) / len(signs)


def opinion(
    agent: Agent,
    subject: str,
    score: float,
    confidence: float,
    thesis: str,
    evidence: list[Evidence],
    *,
    quality: DataState,
    used: list[str],
    missing: list[str] | None = None,
    invalidation: str | None = None,
    meta: dict | None = None,
    horizon: int | None = None,
    directional: bool = True,
) -> Opinion:
    """``directional=False``: the agent reports evidence and context (kept in ``meta``) but casts no vote."""
    score = clamp(score)
    return Opinion(
        agent_id=agent.spec.id,
        agent_version=agent.spec.version,
        subject=subject,
        stance=stance_of(score) if directional else Stance.ABSTAIN,
        score=score,
        confidence=clamp(confidence, 0.0, 1.0),
        horizon_days=horizon or agent.spec.horizon_days,
        thesis=thesis,
        evidence=evidence,
        data_used=used,
        data_missing=list(missing or []),
        data_quality=quality,
        invalidation=invalidation,
        meta=meta or {},
    )


def top_details(evidence: list[Evidence], n: int = 3) -> str:
    return ", ".join(e.detail for e in sorted(evidence, key=lambda e: -e.strength)[:n])
