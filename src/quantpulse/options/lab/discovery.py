"""Analogues, feature discovery, feature interactions and the knowledge graph.

* :func:`analogues` — "which strategies worked in conditions like today's?": standardized state vectors
  (trend, volatility, IV, IV rank, IV/RV, term structure, skew, event proximity, ATR), the k nearest
  historical states, and the outcomes of the trades entered in them — with how many there were.
* :func:`importance` — how much each feature says about outcomes: the rank correlation with the result per
  dollar at risk, measured on a training half and *re-measured* on the held-out half; a feature only counts
  when both halves agree in sign, and a Benjamini–Hochberg control is applied across all of them.
  Importance is not causality, and the output says so.
* :func:`interactions` — products of feature pairs (IV rank × trend, IV/RV × event, delta × DTE, …) tested
  the same way: discovered in one half, validated in the other.
* :func:`graph_edges` — the knowledge graph: STRATEGY WORKS_IN/FAILS_IN REGIME, STRATEGY USES FEATURE,
  STRATEGY DERIVED_FROM SOURCE, TRADE TESTED STRATEGY, TRADE PRODUCED LESSON, LESSON MODIFIES STRATEGY,
  FEATURE INTERACTS_WITH FEATURE.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from quantpulse.options.lab.overfit import benjamini_hochberg

STATE = ("trend", "ret20", "rv20", "iv", "iv_rank", "iv_rv", "term", "skew", "event", "atr")


def _matrix(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> np.ndarray:
    return np.array([[np.nan if r.get(k) is None else float(r[k]) for k in keys] for r in rows], dtype=float)


def analogues(today: Mapping[str, Any], history: Sequence[Mapping[str, Any]], *, k: int = 25,
              keys: Sequence[str] = STATE) -> dict[str, Any]:  # fmt: skip
    """``history``: trades, each with ``features`` (its entry state), ``family`` and a result. The nearest
    ``k`` entry states to ``today`` (standardized; missing dimensions ignored) and what worked in them."""
    feats = [h.get("features") or {} for h in history]
    X = _matrix(feats, keys)
    x = np.array([np.nan if today.get(k_) is None else float(today[k_]) for k_ in keys])
    usable = ~np.isnan(x) & (np.sum(~np.isnan(X), axis=0) > 5 if len(X) else np.zeros(len(keys), bool))
    if not len(history) or not usable.any():
        return {"neighbours": 0, "by_family": {}, "note": "no comparable history"}
    mu = np.nanmean(X[:, usable], axis=0)
    sd = np.nanstd(X[:, usable], axis=0)
    sd[sd == 0] = 1
    Z = (X[:, usable] - mu) / sd
    z = (x[usable] - mu) / sd
    dist = np.sqrt(np.nansum((Z - z) ** 2, axis=1) / np.maximum(np.sum(~np.isnan(Z), axis=1), 1))
    order = np.argsort(dist)[:k]
    by: dict[str, list[float]] = {}
    for i in order:
        h = history[int(i)]
        by.setdefault(str(h.get("family")), []).append(float(h["pnl"]) / float(h.get("max_loss") or 1.0))
    out = {f: {"n": len(v), "mean_on_risk": round(float(np.mean(v)), 4), "win_rate": round(float(np.mean(np.array(v) > 0)), 3)}
           for f, v in by.items()}  # fmt: skip
    return {"neighbours": len(order), "dimensions": [k_ for k_, u in zip(keys, usable, strict=True) if u],
            "mean_distance": round(float(dist[order].mean()), 4), "by_family": out}  # fmt: skip


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    from scipy.stats import spearmanr

    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return None, None
    rho, p = spearmanr(a[m], b[m])
    return float(rho), float(p)


def importance(
    trades: Sequence[Mapping[str, Any]], keys: Sequence[str] | None = None, *, q: float = 0.10
) -> list[dict[str, Any]]:
    """Discovered on the first half of the trades (by time), validated on the second."""
    rows = sorted(trades, key=lambda t: t.get("entry_date") or "")
    if len(rows) < 30:
        return []
    feats = [r.get("features") or {} for r in rows]
    keys = keys or sorted({k for f in feats for k, v in f.items() if isinstance(v, int | float)})
    y = np.array([float(r["pnl"]) / float(r.get("max_loss") or 1.0) for r in rows])
    half = len(rows) // 2
    X = _matrix(feats, keys)
    out: list[dict[str, Any]] = []
    for j, k in enumerate(keys):
        rho1, _ = _spearman(X[:half, j], y[:half])
        rho2, p2 = _spearman(X[half:, j], y[half:])
        if rho1 is None:
            continue
        out.append({"feature": k, "importance": round(rho1, 4), "oos_importance": None if rho2 is None else round(rho2, 4),
                    "p_value": None if p2 is None else round(p2, 5), "_p": p2 if p2 is not None else 1.0,
                    "agrees": rho2 is not None and np.sign(rho1) == np.sign(rho2)})  # fmt: skip
    keep = benjamini_hochberg([r["_p"] for r in out], q)
    for r, discovery in zip(out, keep, strict=True):
        agrees = r.pop("agrees")
        r.pop("_p")
        r["validated"] = bool(discovery and agrees)
        r["note"] = "association, not causation"
    return sorted(out, key=lambda r: -abs(r["oos_importance"] or 0))


def interactions(trades: Sequence[Mapping[str, Any]], pairs: Sequence[tuple[str, str]] | None = None, *,
                 q: float = 0.10) -> list[dict[str, Any]]:  # fmt: skip
    rows = sorted(trades, key=lambda t: t.get("entry_date") or "")
    if len(rows) < 40:
        return []
    feats = [r.get("features") or {} for r in rows]
    base = ("iv_rank", "iv_rv", "ret20", "rv20", "event_days", "atr14_pct", "z5")
    pairs = pairs or list(
        itertools.combinations([b for b in base if any(f.get(b) is not None for f in feats)], 2)
    )
    synthetic = []
    for r, f in zip(rows, feats, strict=True):
        extra = {}
        for a, b in pairs:
            va, vb = f.get(a), f.get(b)
            extra[f"{a}×{b}"] = None if va is None or vb is None else float(va) * float(vb)
        synthetic.append({**r, "features": extra})
    found = importance(synthetic, [f"{a}×{b}" for a, b in pairs], q=q)
    for x in found:
        a, b = x["feature"].split("×")
        x["feature"], x["interaction_with"] = a, b
    return found


def graph_edges(*, strategy: str, regimes: Mapping[str, float] | None = None, features: Sequence[str] = (),
                source: str | None = None, trade: str | None = None, lesson: str | None = None,
                interactions_found: Sequence[tuple[str, str]] = ()) -> list[dict[str, Any]]:  # fmt: skip
    edges = []
    for reg, v in (regimes or {}).items():
        edges.append({"src_type": "STRATEGY", "src_id": strategy, "relation": "WORKS_IN" if v > 0 else "FAILS_IN",
                      "dst_type": "REGIME", "dst_id": reg, "weight": round(float(v), 4)})  # fmt: skip
    for f in features:
        edges.append(
            {
                "src_type": "STRATEGY",
                "src_id": strategy,
                "relation": "USES",
                "dst_type": "FEATURE",
                "dst_id": f,
                "weight": 1.0,
            }
        )
    if source:
        edges.append(
            {
                "src_type": "STRATEGY",
                "src_id": strategy,
                "relation": "DERIVED_FROM",
                "dst_type": "SOURCE",
                "dst_id": source,
                "weight": 1.0,
            }
        )
    if trade:
        edges.append(
            {
                "src_type": "TRADE",
                "src_id": trade,
                "relation": "TESTED",
                "dst_type": "STRATEGY",
                "dst_id": strategy,
                "weight": 1.0,
            }
        )
        if lesson:
            edges.append(
                {
                    "src_type": "TRADE",
                    "src_id": trade,
                    "relation": "PRODUCED",
                    "dst_type": "LESSON",
                    "dst_id": lesson,
                    "weight": 1.0,
                }
            )
    if lesson:
        edges.append(
            {
                "src_type": "LESSON",
                "src_id": lesson,
                "relation": "MODIFIES",
                "dst_type": "STRATEGY",
                "dst_id": strategy,
                "weight": 1.0,
            }
        )
    for a, b in interactions_found:
        edges.append(
            {
                "src_type": "FEATURE",
                "src_id": a,
                "relation": "INTERACTS_WITH",
                "dst_type": "FEATURE",
                "dst_id": b,
                "weight": 1.0,
            }
        )
    return edges


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
