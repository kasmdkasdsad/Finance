"""Competing explanations for a detected change — each with what it predicts and how to test it.

A change is detected first. Then *every* plausible explanation is written down, including the dull ones
(chance; a data artifact), and each is judged only by evidence that tells it apart from the others. No
explanation is privileged. "Automated or AI-driven trading changed the market" is one candidate among many —
its observable implications (faster reversal at the shortest horizons, a changed variance ratio, spreads that
compress while depth becomes fragile) overlap with other liquidity changes, so price data alone rarely
identifies it; the verdict says so instead of guessing.

Verdicts: ``consistent`` (its prediction is observed), ``inconsistent`` (the opposite is observed),
``untestable`` (the data to test it is missing) — never "proven", and never more than one step from the data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

Evidence = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    name: str
    statement: str
    predicts: str
    test: Callable[[Evidence], tuple[str, str]]  # → (verdict, detail)
    identifiable_from_prices: bool = True


def _has(ev: Evidence, *keys: str) -> bool:
    return all(ev.get(k) is not None for k in keys)


def _chance(ev: Evidence) -> tuple[str, str]:
    q, persisted = ev.get("q_value"), ev.get("persisted")
    if q is None:
        return "untestable", "no multiple-testing-adjusted p-value"
    if q > 0.10:
        return "consistent", f"q = {q:.3f}: not distinguishable from chance across everything tested"
    if persisted is False:
        return "consistent", "significant once, not in the following window"
    return "inconsistent", f"q = {q:.3f}" + (" and it persisted" if persisted else "")


def _artifact(ev: Evidence) -> tuple[str, str]:
    if not _has(ev, "feed_changed"):
        return "untestable", "no record of the data feed around the change"
    if ev["feed_changed"] or (ev.get("data_gaps") or 0) > 0.1:
        return "consistent", "the feed or the provider changed, or data went missing, around the change"
    return "inconsistent", "same feed, no unusual gaps"


def _macro(ev: Evidence) -> tuple[str, str]:
    if not _has(ev, "benchmark_shift"):
        return "untestable", "no benchmark (market-wide) series for the same window"
    return ("consistent", "the market as a whole shifted the same way") if ev["benchmark_shift"] else (
        "inconsistent", "the market as a whole did not shift: the change is local")  # fmt: skip


def _events(ev: Evidence) -> tuple[str, str]:
    if not _has(ev, "shift_without_event_days"):
        return "untestable", "no event calendar to exclude event days"
    return ("inconsistent", "the shift persists with event days removed") if ev["shift_without_event_days"] else (
        "consistent", "the shift disappears once event days are removed")  # fmt: skip


def _liquidity(ev: Evidence) -> tuple[str, str]:
    if not _has(ev, "spread_change", "illiquidity_change"):
        return "untestable", "no spread or illiquidity series"
    if ev["spread_change"] > 0 and ev["illiquidity_change"] > 0:
        return "consistent", "spreads and price impact both rose"
    if ev["spread_change"] < 0 and ev["illiquidity_change"] < 0:
        return "inconsistent", "liquidity improved"
    return "consistent" if abs(ev["spread_change"]) > abs(
        ev["illiquidity_change"]
    ) else "inconsistent", "mixed signs"


def _structure(ev: Evidence) -> tuple[str, str]:
    if not _has(ev, "close_volume_share_change"):
        return "untestable", "no intraday volume profile"
    if abs(ev["close_volume_share_change"]) > 0.03:
        return (
            "consistent",
            f"the share of volume at the close moved by {ev['close_volume_share_change']:+.1%}",
        )
    return "inconsistent", "the intraday profile is unchanged"


def _composition(ev: Evidence) -> tuple[str, str]:
    if not _has(ev, "universe_changed"):
        return "untestable", "no universe history"
    return ("consistent", "the set of names changed") if ev["universe_changed"] else (
        "inconsistent", "same names throughout")  # fmt: skip


def _automated_liquidity(ev: Evidence) -> tuple[str, str]:
    if not _has(ev, "autocorr_1m_change", "variance_ratio_change"):
        return "untestable", "no intraday reversal or variance-ratio series"
    faster = ev["autocorr_1m_change"] < -0.03 and ev["variance_ratio_change"] < -0.03
    if faster:
        return "consistent", ("shortest-horizon reversal strengthened (consistent with more automated liquidity "
                              "provision — also with other liquidity changes: not identified by prices alone)")  # fmt: skip
    return "inconsistent", "short-horizon reversal did not strengthen"


CATALOGUE: tuple[Hypothesis, ...] = (
    Hypothesis("chance", "No real change: sampling variation across many tests", "the shift fails a false-discovery "
               "control or does not persist", _chance),
    Hypothesis("data_artifact", "A change in the data, not the market (feed, provider, gaps)", "coincides with a feed "
               "change or missing data", _artifact),
    Hypothesis("macro_regime", "A market-wide volatility or regime change", "the benchmark shifts the same way", _macro),
    Hypothesis("event_clustering", "Scheduled events (earnings season, central-bank meetings) cluster in the window",
               "the shift vanishes when event days are excluded", _events),
    Hypothesis("liquidity_change", "Liquidity was withdrawn (or supplied)", "spreads and price impact move together",
               _liquidity),
    Hypothesis("market_structure", "A change in market structure (trading hours, tick sizes, short-dated options "
               "activity, closing auctions)", "the intraday profile changes, concentrated at particular times",
               _structure),
    Hypothesis("composition", "The universe itself changed (index membership, new names)", "the names differ between "
               "windows", _composition),
    Hypothesis("automated_liquidity", "Faster automated liquidity provision (algorithmic or AI-driven trading) changed "
               "the shortest horizons", "stronger 1-minute reversal and a lower variance ratio — which other liquidity "
               "changes also produce", _automated_liquidity, identifiable_from_prices=False),
)  # fmt: skip


def evaluate(evidence: Evidence) -> list[dict[str, Any]]:
    """Every hypothesis in the catalogue judged against the evidence (none is dropped, none assumed)."""
    out = []
    for h in CATALOGUE:
        verdict, detail = h.test(evidence)
        out.append({"name": h.name, "statement": h.statement, "predicts": h.predicts, "verdict": verdict,
                    "detail": detail, "identifiable_from_prices": h.identifiable_from_prices})  # fmt: skip
    return out


def summary(results: list[dict[str, Any]]) -> str:
    consistent = [r["name"] for r in results if r["verdict"] == "consistent"]
    untestable = [r["name"] for r in results if r["verdict"] == "untestable"]
    if not consistent:
        return "no listed explanation is consistent with the evidence" + (
            f"; untestable: {', '.join(untestable)}" if untestable else ""
        )
    return (f"competing explanations still consistent: {', '.join(consistent)}"
            + (f"; untestable with current data: {', '.join(untestable)}" if untestable else "")
            + " — none is established")  # fmt: skip
