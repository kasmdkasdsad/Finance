"""Learning from evidence: strategy weights, calibration, confusion matrices, meta-learning.

* :func:`shrunk_weight` — a strategy's weight in a context (regime × structure × underlying class × volatility
  state) from a normal–normal (empirical Bayes) update: a prior from its family, pulled towards the observed
  mean in proportion to the evidence. Recent trades count more (exponential recency), but never so much that
  the long record is erased (a floor on old evidence's weight). Small samples stay near the prior.
* :func:`calibration` — do 70% predictions come true 70% of the time? Reliability bins, Brier score, log loss.
* :func:`confusion` — predicted profitable vs actually profitable, by any dimension (strategy, regime,
  underlying, DTE, IV rank, delta, structure).
* :func:`meta_learn` — which strategy families do best in which contexts (learned from recorded outcomes,
  shrunk, with the sample shown) — never hard-coded.
* :func:`structure_bias` — does the Brain keep choosing a structure that its own counterfactuals say is worse?
  (a model-selection lesson).
* :func:`relevance` — how relevant a lesson still is: recency, regime similarity, sample size, stability and
  agreement with current evidence. Old evidence is kept; only its relevance changes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Weight:
    mean: float  # posterior mean return per $ at risk
    sd: float  # posterior standard deviation
    n: float  # effective sample (after recency weighting)
    weight: float  # 0–1: the probability the true mean is positive
    prior_mean: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(getattr(self, k), 5) for k in self.__dataclass_fields__}


def recency_weights(ages_days: Sequence[float], half_life: float = 180.0, floor: float = 0.25) -> np.ndarray:
    """Exponential recency with a floor: a trade from long ago still counts at least ``floor``."""
    a = np.asarray(ages_days, dtype=float)
    return np.maximum(floor, 0.5 ** (a / half_life))


def shrunk_weight(returns: Sequence[float], ages_days: Sequence[float] | None = None, *, prior_mean: float = 0.0,
                  prior_sd: float = 0.1, noise_sd: float | None = None, half_life: float = 180.0) -> Weight:  # fmt: skip
    from scipy.stats import norm

    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return Weight(prior_mean, prior_sd, 0.0, float(norm.cdf(prior_mean / prior_sd)), prior_mean)
    w = recency_weights(ages_days if ages_days is not None else [0.0] * len(r), half_life)
    n_eff = float(w.sum() ** 2 / (w**2).sum())
    mean = float((w * r).sum() / w.sum())
    sigma = noise_sd if noise_sd is not None else (float(r.std(ddof=1)) if len(r) > 1 else prior_sd * 3)
    sigma = max(sigma, 1e-6)
    precision = 1 / prior_sd**2 + n_eff / sigma**2
    post_mean = (prior_mean / prior_sd**2 + mean * n_eff / sigma**2) / precision
    post_sd = math.sqrt(1 / precision)
    return Weight(post_mean, post_sd, n_eff, float(norm.cdf(post_mean / post_sd)), prior_mean)


def context_key(
    regime: str, structure: str, underlying_class: str, vol_state: str
) -> tuple[str, str, str, str]:
    return (
        regime or "UNKNOWN",
        structure or "UNKNOWN",
        underlying_class or "UNKNOWN",
        vol_state or "UNKNOWN_IV",
    )


def weights_by_context(trades: Iterable[Mapping[str, Any]], today: date, *, family_prior: Mapping[str, float] | None = None
                       ) -> dict[tuple[str, str, str, str], Weight]:  # fmt: skip
    """strategy_weight(regime, structure, underlying_class, volatility state) from recorded trades."""
    groups: dict[tuple[str, str, str, str], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for t in trades:
        k = context_key(t.get("regime", ""), t.get("family", ""), t.get("underlying_class", "liquid_large_cap"),
                        t.get("iv_regime", ""))  # fmt: skip
        age = (today - date.fromisoformat(t["exit_date"])).days if t.get("exit_date") else 0
        groups[k][0].append(float(t["pnl"]) / float(t.get("max_loss") or 1.0))
        groups[k][1].append(float(age))
    prior = family_prior or {}
    return {k: shrunk_weight(r, a, prior_mean=prior.get(k[1], 0.0)) for k, (r, a) in groups.items()}


def calibration(predicted: Sequence[float], actual: Sequence[bool], bins: int = 5) -> dict[str, Any]:
    p = np.clip(np.asarray(predicted, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(actual, dtype=float)
    if len(p) == 0:
        return {"n": 0}
    edges = np.linspace(0, 1, bins + 1)
    table: list[dict[str, Any]] = []
    for lo, hi in pairwise(edges):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "predicted": round(float(p[m].mean()), 3),
                          "observed": round(float(y[m].mean()), 3)})  # fmt: skip
    brier = float(((p - y) ** 2).mean())
    logloss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    gap = float(np.mean([abs(r["predicted"] - r["observed"]) for r in table])) if table else None
    return {"n": len(p), "reliability": table, "brier": round(brier, 4), "log_loss": round(logloss, 4),
            "mean_abs_gap": None if gap is None else round(gap, 4),
            "well_calibrated": gap is not None and gap < 0.1 and len(p) >= 30}  # fmt: skip


def confusion(rows: Iterable[Mapping[str, Any]], by: str) -> dict[str, dict[str, int]]:
    """rows: {'predicted_profit': bool, 'profit': bool, <by>: value} → per value: TP, FP, FN, TN."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    for r in rows:
        key = str(r.get(by))
        pred, act = bool(r.get("predicted_profit")), bool(r.get("profit"))
        out[key]["TP" if pred and act else "FP" if pred else "FN" if act else "TN"] += 1
    return dict(out)


def precision(c: Mapping[str, int]) -> float | None:
    d = c["TP"] + c["FP"]
    return c["TP"] / d if d else None


def meta_learn(
    trades: Iterable[Mapping[str, Any]], *, context: str = "iv_regime", min_n: int = 5
) -> dict[str, Any]:
    """Per context value: families ranked by shrunk expectancy per $ at risk (with n); the learned statement
    'in context X, family A has done better than B' only when both have ``min_n`` trades."""
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in trades:
        groups[str(t.get(context))][str(t.get("family"))].append(
            float(t["pnl"]) / float(t.get("max_loss") or 1.0)
        )
    out: dict[str, Any] = {}
    for ctx, fams in groups.items():
        ranked: list[dict[str, Any]] = []
        for fam, r in fams.items():
            w = shrunk_weight(r)
            ranked.append(
                {
                    "family": fam,
                    "n": len(r),
                    "posterior_mean": round(w.mean, 4),
                    "p_positive": round(w.weight, 3),
                }
            )
        ranked.sort(key=lambda x: x["posterior_mean"], reverse=True)
        eligible = [x for x in ranked if x["n"] >= min_n]
        statement = None
        if len(eligible) >= 2 and eligible[0]["posterior_mean"] > eligible[-1]["posterior_mean"]:
            statement = (f"when {context} = {ctx}, {eligible[0]['family']} has done better than "
                         f"{eligible[-1]['family']} ({eligible[0]['n']} vs {eligible[-1]['n']} trades; shrunk means "
                         f"{eligible[0]['posterior_mean']:+.3f} vs {eligible[-1]['posterior_mean']:+.3f})")  # fmt: skip
        out[ctx] = {"families": ranked, "learned": statement}
    return out


def structure_bias(records: Iterable[Mapping[str, Any]], min_n: int = 10) -> dict[str, Any] | None:
    """records: {'chosen': family, 'best_alternative': family, 'chosen_pnl': x, 'best_alternative_pnl': y}.
    A systematic bias: the chosen structure loses to the same alternative in most of ≥ ``min_n`` cases."""
    pairs: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in records:
        if r.get("best_alternative") and r["best_alternative"] != r.get("chosen"):
            pairs[(str(r["chosen"]), str(r["best_alternative"]))].append(
                float(r["best_alternative_pnl"]) > float(r["chosen_pnl"])
            )
    worst: dict[str, Any] | None = None
    for (chosen, alt), wins in pairs.items():
        if len(wins) >= min_n and sum(wins) / len(wins) >= 0.65:
            share = sum(wins) / len(wins)
            if worst is None or share > worst["share"]:
                worst = {"chosen": chosen, "better": alt, "share": round(share, 3), "n": len(wins),
                         "lesson": f"structure-selection bias: {chosen} was chosen where {alt} did better in "
                                   f"{share:.0%} of {len(wins)} comparable cases"}  # fmt: skip
    return worst


def relevance(*, created: date, today: date, regime_then: str | None, regime_now: str | None, sample_size: int,
              stability: float | None, agrees_now: bool | None, half_life_days: float = 365.0) -> float:  # fmt: skip
    """0–1. Old evidence is never deleted; its relevance decays and recovers with the current evidence."""
    recency = 0.5 ** (max((today - created).days, 0) / half_life_days)
    regime = 1.0 if regime_then is None or regime_now is None else (1.0 if regime_then == regime_now else 0.5)
    size = min(1.0, math.log1p(sample_size) / math.log1p(100))
    stab = 0.7 if stability is None else 0.4 + 0.6 * max(0.0, min(1.0, stability))
    agree = 1.0 if agrees_now is None else (1.2 if agrees_now else 0.5)
    return round(max(0.0, min(1.0, (0.35 + 0.65 * recency) * regime * (0.4 + 0.6 * size) * stab * agree)), 4)
