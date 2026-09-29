"""The Market Evolution Monitor: detect structural change, explain it only with evidence, re-test what it
touches.

Dimensions monitored (each a set of time series, per underlying or market-wide, at several timescales):

==================  ====================================================================================
volatility          daily realized volatility (5, 20, 60 days), its volatility
micro_volatility    realized volatility at 1–60 minutes, the noise ratio, the jump share
microstructure      the variance ratio, 1- and 5-minute autocorrelation, Roll and quoted spreads, Amihud
                    illiquidity, the open/close volume shares
options             ATM IV, IV rank, the IV/RV premium, skew, term slope, option spreads
correlation         the average pairwise correlation of the universe's daily returns
liquidity           dollar volume, quoted spreads, option open interest
execution           slippage (bps and dollars), fill rate, latency — from QuantPulse's own orders
strategy            each strategy's rolling result per dollar at risk (live evidence only)
==================  ====================================================================================

:func:`scan` compares each series' recent window with its reference window (:mod:`.shifts`), finds change
points, and applies one false-discovery-rate control across *everything* scanned, so a change is only
reported when it survives that. :func:`explain` asks whether a driver (e.g. micro-volatility) explains an
outcome (the IV/RV premium, slippage, strategy results, realized outcomes) — discovered on one half of the
data and confirmed on the other, or reported as not robust. :func:`revalidation_targets` names the strategies
a change touches; the lab re-runs their validation on recent data (nothing is assumed still valid).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from quantpulse.evolution import relationships as rel
from quantpulse.evolution.shifts import change_points, compare
from quantpulse.options.lab.overfit import benjamini_hochberg

DIMENSIONS = ("volatility", "micro_volatility", "microstructure", "options", "correlation", "liquidity", "execution",
              "strategy")  # fmt: skip
# which strategy genes each dimension bears on (for re-validation)
TOUCHES = {
    "volatility": ("iv_rank_min", "iv_rank_max", "iv_rv_min", "iv_rv_max", "regime_filter"),
    "micro_volatility": ("max_spread_pct",),
    "microstructure": ("max_spread_pct", "min_open_interest"),
    "options": ("iv_rank_min", "iv_rank_max", "iv_rv_min", "iv_rv_max", "term_filter", "skew_filter"),
    "correlation": ("risk_per_trade",),
    "liquidity": ("max_spread_pct", "min_open_interest"),
    "execution": ("max_spread_pct",),
    "strategy": (),
}


@dataclass
class Series:
    dimension: str
    subject: str  # an underlying, a strategy key, or "market"
    metric: str
    timescale: str  # "1m" … "60m", "1d", "20d", …
    points: list[tuple[date, float | None]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.dimension}:{self.subject}:{self.metric}:{self.timescale}"

    def values(self) -> list[float | None]:
        return [v for _, v in sorted(self.points, key=lambda p: p[0])]


def scan(series: Sequence[Series], *, recent: int = 20, reference: int = 120, q: float = 0.10,
         previous: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Every series tested; ``significant`` only after the FDR control across all of them. ``previous``:
    the last scan's results by key (to say whether a change *persisted*)."""
    rows: list[dict[str, Any]] = []
    for s in series:
        vals = s.values()
        if len(vals) < recent + 10:
            continue
        ref = vals[-(recent + reference) : -recent]
        rec = vals[-recent:]
        cmp = compare(ref, rec)
        if "p_value" not in cmp:
            continue
        pts = sorted(s.points, key=lambda p: p[0])
        cps = change_points(vals, min_seg=max(10, recent // 2))
        rows.append({
            "key": s.key, "dimension": s.dimension, "subject": s.subject, "metric": s.metric, "timescale": s.timescale,
            "reference_window": [pts[-(recent + len(ref))][0].isoformat(), pts[-recent - 1][0].isoformat()],
            "recent_window": [pts[-recent][0].isoformat(), pts[-1][0].isoformat()],
            "test": cmp, "p_value": cmp["p_value"], "kind": cmp["kind"],
            "change_points": [pts[i][0].isoformat() for i in cps if i < len(pts)],
        })  # fmt: skip
    keep = benjamini_hochberg([r["p_value"] for r in rows], q)
    m = len(rows)
    ranked = sorted(range(m), key=lambda i: rows[i]["p_value"])
    qvals = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):  # Benjamini–Hochberg adjusted p-values (q-values)
        i = ranked[rank - 1]
        running = min(running, rows[i]["p_value"] * m / rank)
        qvals[i] = running
    for r, k, qv in zip(rows, keep, qvals, strict=True):
        r["q_value"] = round(min(qv, 1.0), 6)
        r["significant"] = bool(k) and r["kind"] != "none"
        prev = (previous or {}).get(r["key"])
        r["persisted"] = None if prev is None else bool(prev.get("significant") and r["significant"])
    return rows


def explain(outcome: Sequence[float | None], driver: Sequence[float | None], *, name: str) -> dict[str, Any]:
    """Does ``driver`` explain ``outcome``? Estimated on each half; robust only if both halves agree in sign
    and are significant. Association, not causation."""
    n = min(len(outcome), len(driver))
    half = n // 2
    first = rel.estimate(driver[:half], outcome[:half])
    second = rel.estimate(driver[half:n], outcome[half:n])
    robust = first.significant and second.significant and (first.slope or 0) * (second.slope or 0) > 0
    return {"relationship": name, "first_half": first.as_dict(), "second_half": second.as_dict(),
            "explains": robust, "verdict": "explains (in both halves)" if robust else
            "not robust: discovered in one half, not confirmed in the other" if first.significant or second.significant
            else "no detectable relationship", "note": "association, not causation"}  # fmt: skip


def revalidation_targets(change: Mapping[str, Any], strategies: Sequence[Mapping[str, Any]]) -> list[str]:
    """Strategy keys a change bears on: those trading the changed underlying (or all, for a market-wide
    change) at a stage where they matter, and — for a strategy-performance change — that strategy itself."""
    if not change.get("significant"):
        return []
    dim, subject = change.get("dimension"), change.get("subject")
    genes = TOUCHES.get(str(dim), ())
    out = []
    for s in strategies:
        if s.get("stage") not in ("VALIDATION", "WALK_FORWARD", "PAPER_SHADOW", "PAPER_ACTIVE", "PROVEN"):
            continue
        if dim == "strategy":
            if s.get("key") == subject:
                out.append(str(s["key"]))
            continue
        underlyings = s.get("underlyings") or ()
        if subject != "market" and underlyings and subject not in underlyings:
            continue
        g = s.get("genome") or {}
        if not genes or any(g.get(x) not in (None, "any", ()) for x in genes) or subject == "market":
            out.append(str(s["key"]))
    return sorted(set(out))
