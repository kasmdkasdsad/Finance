"""Has the distribution changed? Distribution-shift and change-point tests, each with its statistic.

* :func:`compare` — a reference window against a recent one: the Kolmogorov–Smirnov and Anderson–Darling
  two-sample tests (shape), Welch's t (level), Brown–Forsythe (spread), the Wasserstein distance and the
  population stability index (how far the distribution moved), and the effect size in reference standard
  deviations;
* :func:`change_points` — where a series changed: a two-sided CUSUM on the standardized series and a
  likelihood search for the single most likely change in mean and in variance (binary segmentation to find
  several, with a minimum segment length);
* :func:`page_hinkley` — an online detector for a drift in the mean.

A change counts as *significant* only after a false-discovery-rate control across everything tested at once
(:func:`quantpulse.options.lab.overfit.benjamini_hochberg`) — the monitor tests many series, and some will
look different by chance.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np


def _clean(x: Sequence[float | None]) -> np.ndarray:
    a = np.asarray([v for v in x if v is not None], dtype=float)
    return a[np.isfinite(a)]


def psi(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float | None:
    """Population stability index over the reference's deciles (> 0.25 is conventionally a large shift)."""
    if len(reference) < bins * 2 or len(recent) < bins:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, edges)[0] / len(reference)
    b = np.histogram(recent, edges)[0] / len(recent)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def compare(
    reference: Sequence[float | None], recent: Sequence[float | None], *, min_n: int = 10
) -> dict[str, Any]:
    from scipy import stats

    a, b = _clean(reference), _clean(recent)
    out: dict[str, Any] = {"n_reference": len(a), "n_recent": len(b)}
    if len(a) < min_n or len(b) < max(5, min_n // 2):
        out["note"] = "too few observations to compare"
        return out
    ks = stats.ks_2samp(a, b)
    try:
        # SciPy caps this p-value to [0.001, 0.25] and warns about it: known, and clipped below
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ad = stats.anderson_ksamp([a, b])
        ad_p = float(min(max(ad.pvalue, 0.001), 0.25))
    except (ValueError, FloatingPointError):
        ad_p = None
    t = stats.ttest_ind(b, a, equal_var=False)
    lev = stats.levene(a, b, center="median")
    sd = float(a.std(ddof=1)) or 1e-12
    out.update({
        "reference_mean": round(float(a.mean()), 6),
        "recent_mean": round(float(b.mean()), 6),
        "reference_sd": round(sd, 6),
        "recent_sd": round(float(b.std(ddof=1)), 6),
        "effect_sd": round(float((b.mean() - a.mean()) / sd), 4),
        "variance_ratio": round(float(b.var(ddof=1) / max(a.var(ddof=1), 1e-24)), 4),
        "ks_stat": round(float(ks.statistic), 4),
        "ks_p": float(ks.pvalue),
        "ad_p": ad_p,
        "mean_p": float(t.pvalue),
        "spread_p": float(lev.pvalue),
        "wasserstein": round(float(stats.wasserstein_distance(a, b)), 6),
        "psi": None if (p := psi(a, b)) is None else round(p, 4),
    })  # fmt: skip
    out["p_value"] = float(
        min(1.0, 3 * min(out["ks_p"], out["mean_p"], out["spread_p"]))
    )  # Bonferroni over 3
    out["kind"] = _kind(out)
    return out


def _kind(r: dict[str, Any]) -> str:
    if r["mean_p"] < 0.05 and abs(r["effect_sd"]) >= 0.5:
        return "level_up" if r["effect_sd"] > 0 else "level_down"
    if r["spread_p"] < 0.05:
        return "more_dispersed" if r["variance_ratio"] > 1 else "less_dispersed"
    if r["ks_p"] < 0.05:
        return "shape"
    return "none"


def cusum(x: Sequence[float | None], k: float = 0.5, h: float = 5.0) -> dict[str, Any]:
    """Two-sided CUSUM on the series standardized by its first half: the first alarm (index) up or down."""
    a = _clean(x)
    if len(a) < 20:
        return {"alarm": None}
    ref = a[: len(a) // 2]
    mu, sd = float(ref.mean()), float(ref.std(ddof=1)) or 1e-12
    z = (a - mu) / sd
    hi = lo = 0.0
    for i, v in enumerate(z):
        hi = max(0.0, hi + v - k)
        lo = min(0.0, lo + v + k)
        if hi > h:
            return {"alarm": i, "direction": "up"}
        if lo < -h:
            return {"alarm": i, "direction": "down"}
    return {"alarm": None}


def page_hinkley(x: Sequence[float | None], delta: float = 0.005, threshold: float = 0.05) -> int | None:
    a = _clean(x)
    mean, m, big = 0.0, 0.0, 0.0
    for i, v in enumerate(a, start=1):
        mean += (v - mean) / i
        m += v - mean - delta
        big = min(big, m)
        if m - big > threshold:
            return i - 1
    return None


def _best_split(a: np.ndarray, min_seg: int) -> tuple[int | None, float]:
    """The split maximizing the Gaussian log-likelihood gain (mean and variance change)."""
    n = len(a)
    if n < 2 * min_seg:
        return None, 0.0

    def nll(seg: np.ndarray) -> float:
        v = max(float(seg.var()), 1e-24)
        return 0.5 * len(seg) * math.log(v)

    whole = nll(a)
    best, gain = None, 0.0
    for i in range(min_seg, n - min_seg + 1):
        g = whole - nll(a[:i]) - nll(a[i:])
        if g > gain:
            best, gain = i, g
    return best, gain


def change_points(x: Sequence[float | None], *, min_seg: int = 20, penalty: float | None = None,
                  max_points: int = 5) -> list[int]:  # fmt: skip
    """Binary segmentation with a BIC-like penalty (3·log n by default)."""
    a = _clean(x)
    pen = penalty if penalty is not None else 3 * math.log(max(len(a), 2))
    found: list[int] = []
    stack = [(0, len(a))]
    while stack and len(found) < max_points:
        s, e = stack.pop()
        i, gain = _best_split(a[s:e], min_seg)
        if i is not None and gain > pen:
            found.append(s + i)
            stack += [(s, s + i), (s + i, e)]
    return sorted(found)
