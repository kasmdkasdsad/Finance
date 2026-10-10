"""Is today's data like the data the model learned from?

* :func:`reference` — per feature, the training data's percentiles and decile bins (kept with the model);
* :func:`psi` — the population stability index of recent candidates against those bins (by convention under
  0.1 stable, 0.1–0.25 a moderate shift, above 0.25 a major one);
* :func:`out_of_range` — for one candidate, the share of its known features outside the training data's 1st–99th
  percentile range. A model asked about a world it has not seen says so: the agent abstains when this is high.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

EPS = 1e-4


def reference(X: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(names):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if len(col) < 20:
            continue
        p = np.percentile(col, [1, 5, 50, 95, 99])
        edges = np.unique(np.percentile(col, np.linspace(0, 100, 11)))
        counts = np.histogram(col, bins=edges)[0] if len(edges) > 1 else np.array([len(col)])
        out[name] = {"p1": float(p[0]), "p5": float(p[1]), "p50": float(p[2]), "p95": float(p[3]),
                     "p99": float(p[4]), "edges": edges.tolist(), "share": (counts / counts.sum()).tolist(),
                     "missing": float(1 - len(col) / len(X))}  # fmt: skip
    return out


def psi(ref: dict[str, Any], values: np.ndarray) -> float | None:
    v = values[np.isfinite(values)]
    edges = np.asarray(ref.get("edges") or [], dtype=float)
    if len(v) < 20 or len(edges) < 2:
        return None
    clipped = np.clip(v, edges[0], edges[-1])
    actual = np.histogram(clipped, bins=edges)[0] / len(v)
    expected = np.asarray(ref["share"], dtype=float)
    a, e = np.maximum(actual, EPS), np.maximum(expected, EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(refs: dict[str, dict[str, Any]], X: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    scores = {}
    for j, name in enumerate(names):
        if name in refs:
            s = psi(refs[name], X[:, j])
            if s is not None:
                scores[name] = round(s, 4)
    worst = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return {"features": len(scores), "major": [k for k, s in scores.items() if s > 0.25],
            "moderate": [k for k, s in scores.items() if 0.1 < s <= 0.25], "worst": worst}  # fmt: skip


def out_of_range(
    refs: dict[str, dict[str, Any]], x: np.ndarray, names: Sequence[str]
) -> tuple[float, list[str]]:
    known, outside = 0, []
    for j, name in enumerate(names):
        r = refs.get(name)
        v = x[j]
        if r is None or not np.isfinite(v):
            continue
        known += 1
        if v < r["p1"] or v > r["p99"]:
            outside.append(name)
    return (len(outside) / known if known else 0.0), outside
