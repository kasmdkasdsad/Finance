"""The StrategyCritic: attack a strategy until it breaks, or fails to.

Each attack reruns or re-slices the evidence and answers one question:

* **costs** — does it survive pessimistic fills (the full spread on every leg)?
* **wider spreads** — the stress fills (spreads doubled, fees up, fills missed, exits late)?
* **volatility spikes and gaps** — the tail lab's worst scenarios within the maximum loss?
* **regime change** — is it positive in more than one regime? does it fail across transitions?
* **2008 / 2020 / 2022** — how did it do in those stretches — or does the data not cover them (said so)?
* **small sample** — enough trades for the mean to mean anything?
* **look-ahead** — does entering one day *later* destroy it (an edge that needs the same day's close)?
* **one ticker** — does it survive leaving out its best underlying?
* **one decade** — is it positive in both halves of the period?
* **realistic fills** — is the edge larger than the cost of trading it?
* **unavailable information / survivorship** — are the inputs point-in-time; was the universe chosen with
  hindsight? (Structural checks: the features are past-only by construction; a universe of today's
  constituents is flagged.)

A strategy that fails a *blocking* attack does not graduate; the others are warnings kept with it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import date
from typing import Any

from quantpulse.options.fills import ExecutionModel
from quantpulse.options.lab.backtest import BacktestResult, run
from quantpulse.options.lab.chains import ChainSource
from quantpulse.options.lab.features import DayFeatures
from quantpulse.options.lab.genome import Genome

STRESS_PERIODS = {
    "2008 crisis": (date(2008, 9, 1), date(2009, 3, 31)),
    "2020 COVID crash": (date(2020, 2, 15), date(2020, 4, 30)),
    "2022 rates and valuations": (date(2022, 1, 1), date(2022, 10, 31)),
}
BLOCKING = ("costs", "look_ahead", "small_sample", "realistic_fills")


class _Delayed:
    """Features shifted by one trading day: the signal decided today is acted on tomorrow."""

    def __init__(self, feats: Mapping[date, DayFeatures]) -> None:
        days = sorted(feats)
        self._map = {days[i + 1]: feats[days[i]] for i in range(len(days) - 1)}
        for d, f in self._map.items():
            self._map[d] = replace(f, day=d, spot=feats[d].spot)

    def get(self, d: date) -> DayFeatures | None:
        return self._map.get(d)

    def __iter__(self) -> Iterator[date]:
        return iter(self._map)

    def __getitem__(self, d: date) -> DayFeatures:
        return self._map[d]

    def __len__(self) -> int:
        return len(self._map)


def critique(
    g: Genome,
    base: BacktestResult,
    source: ChainSource,
    features: Mapping[str, Mapping[date, DayFeatures]],
    *,
    tail_breach: bool | None = None,
    point_in_time_universe: bool = False,
    regime_at: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    cfg = base.config
    ror = base.metrics.get("expectancy_on_risk")
    attacks: dict[str, dict[str, Any]] = {}

    def rerun(model: ExecutionModel, feats: Mapping[str, Mapping[date, DayFeatures]] = features,
              **cfg_changes: Any) -> dict[str, Any]:  # fmt: skip
        return run(g, source, feats, replace(cfg, model=model, **cfg_changes)).metrics

    pess = rerun(ExecutionModel.PESSIMISTIC)
    attacks["costs"] = {"passed": (pess.get("expectancy_on_risk") or 0) > 0,
                        "detail": f"pessimistic fills: {pess.get('expectancy_on_risk')} per $ at risk"}  # fmt: skip
    stress = rerun(ExecutionModel.STRESS)
    attacks["wider_spreads"] = {"passed": (stress.get("expectancy_on_risk") or 0) > 0,
                                "detail": f"stress fills: {stress.get('expectancy_on_risk')} per $ at risk"}  # fmt: skip
    attacks["tail"] = {"passed": tail_breach is False, "detail": "tail scenarios within the maximum loss"
                       if tail_breach is False else "tail stress not run" if tail_breach is None else "a scenario breached the maximum loss"}  # fmt: skip
    by_regime: dict[str, list[float]] = {}
    for t in base.trades:
        by_regime.setdefault(t.get("regime") or "?", []).append(t["pnl"] / (t.get("max_loss") or 1))
    positive_regimes = [k for k, v in by_regime.items() if len(v) >= 3 and sum(v) / len(v) > 0]
    attacks["regime_change"] = {"passed": len(positive_regimes) >= 2 or len(by_regime) <= 1,
                                "detail": f"positive in {positive_regimes or 'no regime'} of {sorted(by_regime)}"}  # fmt: skip
    periods = {}
    for name, (a, b) in STRESS_PERIODS.items():
        covered = cfg.start <= a and b <= cfg.end
        if not covered:
            periods[name] = "not covered by the data: untested"
            continue
        m = run(g, source, features, replace(cfg, start=a, end=b)).metrics
        periods[name] = (
            f"{m.get('trades', 0)} trades, {m.get('total_pnl')} P&L, drawdown {m.get('max_drawdown')}"
        )
    attacks["stress_periods"] = {"passed": None, "detail": periods}
    n = base.metrics.get("trades") or 0
    attacks["small_sample"] = {"passed": n >= 30, "detail": f"{n} trades"}
    delayed = {u: _Delayed(f) for u, f in features.items()}
    lag = run(g, source, delayed, cfg).metrics  # type: ignore[arg-type]
    lag_ror = lag.get("expectancy_on_risk")
    collapsed = ror is not None and ror > 0 and (lag_ror is None or lag_ror < 0.25 * ror)
    attacks["look_ahead"] = {"passed": not collapsed,
                             "detail": f"entering a day later: {lag_ror} vs {ror} per $ at risk"}  # fmt: skip
    pnl_by_sym: dict[str, float] = {}
    for t in base.trades:
        pnl_by_sym[t["underlying"]] = pnl_by_sym.get(t["underlying"], 0.0) + t["pnl"]
    if len(pnl_by_sym) > 1:
        best = max(pnl_by_sym, key=lambda k: pnl_by_sym[k])
        rest = [u for u in cfg.underlyings if u != best]
        loo = run(g, source, features, replace(cfg, underlyings=tuple(rest))).metrics
        attacks["one_ticker"] = {"passed": (loo.get("expectancy_on_risk") or 0) > 0,
                                 "detail": f"without {best}: {loo.get('expectancy_on_risk')} per $ at risk"}  # fmt: skip
    else:
        attacks["one_ticker"] = {"passed": False if len(cfg.underlyings) > 1 else None,
                                 "detail": "all profit from one underlying" if pnl_by_sym else "no trades"}  # fmt: skip
    mid = cfg.start + (cfg.end - cfg.start) / 2
    first = run(g, source, features, replace(cfg, end=mid)).metrics.get("expectancy_on_risk")
    second = run(g, source, features, replace(cfg, start=mid)).metrics.get("expectancy_on_risk")
    attacks["one_period"] = {"passed": (first or 0) > 0 and (second or 0) > 0,
                             "detail": f"first half {first}, second half {second} per $ at risk"}  # fmt: skip
    spread = base.metrics.get("spread_cost") or 0.0
    total = base.metrics.get("total_pnl") or 0.0
    attacks["realistic_fills"] = {"passed": total > 0 and total > 0.25 * spread,
                                  "detail": f"P&L {total} after paying {spread} in spreads"}  # fmt: skip
    attacks["information"] = {"passed": point_in_time_universe or None,
                              "detail": "features are past-only by construction; "
                              + ("the universe is point-in-time" if point_in_time_universe else
                                 "the universe is today's list: survivorship bias possible (flagged)")}  # fmt: skip
    blocking_failed = [k for k in BLOCKING if attacks.get(k, {}).get("passed") is False]
    warnings = [k for k, v in attacks.items() if v.get("passed") is False and k not in BLOCKING]
    return {"attacks": attacks, "survived": not blocking_failed, "blocking_failed": blocking_failed,
            "warnings": warnings}  # fmt: skip
