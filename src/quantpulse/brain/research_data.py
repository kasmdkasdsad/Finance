"""Extra evidence the brain gathers for its focus symbols only (cost control): option-chain metrics and the
earnings calendar with past earnings reactions. Everything comes from existing services and is computed
deterministically; a symbol without real data simply gets no entry (never a made-up one).

* **Options** — :meth:`OptionsService.chain` (cached; the trading loader asks for the same chains), only
  live or cached chains (never synthetic). At-the-money implied volatility reuses
  :func:`~quantpulse.services.trading_data.atm_implied_vol`.
* **Earnings** — :meth:`ReferenceService.events` (SEC 8-K item 2.02 filings) and
  :meth:`ReferenceService.next_earnings`; reactions reuse :func:`quantpulse.domain.earnings.reactions` and
  :func:`~quantpulse.domain.earnings.typical_move` on the cycle's own daily closes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from functools import partial
from typing import Any

import pandas as pd

from quantpulse.domain import earnings as earn
from quantpulse.schemas.common import DataStatus
from quantpulse.schemas.options import OptionChain
from quantpulse.services.options import OptionsService
from quantpulse.services.reference import ReferenceService
from quantpulse.services.trading_data import atm_implied_vol

logger = logging.getLogger(__name__)
USABLE = (DataStatus.LIVE, DataStatus.CACHED)


def _iv_near(chain: OptionChain, expiry: date, kind: str, strike_target: float) -> float | None:
    near = [
        c
        for c in chain.contracts
        if c.expiration == expiry and c.kind == kind and c.implied_volatility and c.strike > 0
    ]
    if not near:
        return None
    best = min(near, key=lambda c: abs(c.strike - strike_target))
    if abs(best.strike / strike_target - 1) > 0.06:  # nothing close enough to the target moneyness
        return None
    iv = float(best.implied_volatility or 0)
    return iv if 0.01 < iv < 5 else None


def chain_metrics(chain: OptionChain, spot: float, today: date) -> dict[str, Any]:
    """Implied volatility level and term structure, put/call skew, and put/call volume and open interest.
    ``None`` where the chain does not support a figure."""
    expiries = sorted({c.expiration for c in chain.contracts if (c.expiration - today).days >= 7})
    out: dict[str, Any] = {
        "atm_iv": atm_implied_vol(chain, spot, today, 30),
        "atm_iv_far": atm_implied_vol(chain, spot, today, 90),
        "expiries": len(expiries),
        "contracts": len(chain.contracts),
    }
    near_far = out["atm_iv"], out["atm_iv_far"]
    out["term_slope"] = (
        near_far[1] - near_far[0] if None not in near_far and len(expiries) > 1 else None
    )  # < 0: inverted (near-term stress or an event)
    if expiries:
        expiry = min(expiries, key=lambda e: abs((e - today).days - 30))
        put, call = _iv_near(chain, expiry, "put", 0.92 * spot), _iv_near(chain, expiry, "call", 1.08 * spot)
        out["skew"] = put - call if put is not None and call is not None else None  # > 0: puts richer
        out["skew_expiry"] = expiry.isoformat()
    else:
        out["skew"] = None
    calls = [c for c in chain.contracts if c.kind == "call"]
    puts = [c for c in chain.contracts if c.kind == "put"]
    cv, pv = sum(c.volume or 0 for c in calls), sum(c.volume or 0 for c in puts)
    coi, poi = sum(c.open_interest or 0 for c in calls), sum(c.open_interest or 0 for c in puts)
    out["call_volume"], out["put_volume"] = cv, pv
    out["pc_volume"] = pv / cv if cv > 0 else None
    out["pc_oi"] = poi / coi if coi > 0 else None
    out["volume_oi"] = (cv + pv) / (coi + poi) if coi + poi > 0 else None  # > 1: unusual turnover
    return out


async def _bounded(
    jobs: dict[str, Callable[[], Awaitable[Any]]], timeout: float
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run ``jobs`` concurrently; results that arrive in time, and why the others did not."""
    tasks = {k: asyncio.ensure_future(fn()) for k, fn in jobs.items()}
    if not tasks:
        return {}, {}
    _, pending = await asyncio.wait(tasks.values(), timeout=timeout)
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, task in tasks.items():
        if task in pending:
            task.cancel()
            errors[key] = f"no answer within {timeout:.0f}s"
        elif task.exception() is not None:
            exc = task.exception()
            errors[key] = f"{type(exc).__name__}: {exc}"
        elif task.result() is not None:
            results[key] = task.result()
    return results, errors


async def option_metrics(
    options: OptionsService, symbols: Sequence[str], spots: dict[str, float], today: date, timeout: float
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    async def one(sym: str) -> dict[str, Any] | None:
        chain_r, _ = await options.chain(sym, max_expirations=4)
        if chain_r.status not in USABLE:
            return None  # synthetic chains are never evidence
        m = chain_metrics(chain_r.value, spots[sym], today)
        m["source"], m["status"] = chain_r.provenance.provider, chain_r.status.value
        m["as_of"] = chain_r.value.as_of.isoformat()
        return m

    return await _bounded({s: partial(one, s) for s in symbols if spots.get(s)}, timeout)


def _series_map(s: pd.Series) -> dict[date, float]:
    s = s.dropna()
    return {pd.Timestamp(i).date(): float(v) for i, v in s.items()}


async def earnings_events(
    reference: ReferenceService,
    symbols: Sequence[str],
    close: pd.DataFrame,
    benchmark: pd.Series,
    today: date,
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    bench = _series_map(benchmark)

    async def one(sym: str) -> dict[str, Any] | None:
        ev = await reference.events(sym)
        if ev.status is DataStatus.SYNTHETIC:
            return None
        nxt, source = await reference.next_earnings(sym)
        closes = _series_map(close[sym]) if sym in close.columns else {}
        rs = earn.reactions(ev.value.earnings, closes, bench, until=today)
        last = rs[-1] if rs else None
        return {
            "next": nxt.isoformat() if nxt else None,
            "next_source": source,
            "days_to_next": (nxt - today).days if nxt else None,
            "typical_move": earn.typical_move(rs),
            "reactions": [
                {"date": r.reaction_date.isoformat(), "return": r.stock_return, "abnormal": r.abnormal}
                for r in rs[-8:]
            ],
            "last_date": last.reaction_date.isoformat() if last else None,
            "days_since_last": (today - last.reaction_date).days if last else None,
            "last_abnormal": last.abnormal if last else None,
            "releases_on_file": len(ev.value.earnings),
            "source": ev.provenance.provider,
        }

    return await _bounded({s: partial(one, s) for s in symbols}, timeout)
