"""What the Options Brain sees of one underlying: its live option chain, the chain's data quality, and the
day's point-in-time features — computed by the same functions the backtester uses
(:func:`quantpulse.options.lab.features.history`), so a signal means the same thing live as in research.

Implied-volatility history is QuantPulse's own: each day's 30-day constant-maturity ATM IV is recorded
(``options_iv_history``) and IV rank/percentile are computed from those records only. Until 60 days exist
the rank is unknown — never estimated — and strategies that filter on it simply do not enter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select

from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.options_models import OptionsChainSnapshotRow, OptionsIVHistoryRow
from quantpulse.db.session import Database
from quantpulse.options.analytics import Expiry, by_expiry, constant_maturity_iv, term_structure
from quantpulse.options.data import ChainSnapshot, OptionsMarketDataProvider
from quantpulse.options.lab.features import DayFeatures, history, iv_regime, iv_trend, trend_regime
from quantpulse.schemas.common import DataStatus

logger = logging.getLogger(__name__)
HISTORY_DAYS = 420  # the 200-day average and a year of 20-day returns


@dataclass
class UnderlyingView:
    underlying: str
    as_of: datetime
    spot: float | None = None
    chain: ChainSnapshot | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    features: DayFeatures | None = None
    expiries: list[Expiry] = field(default_factory=list)
    term: dict[str, Any] = field(default_factory=dict)
    atm_iv_30d: float | None = None
    closes: dict[date, float] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    # a single stock (an S&P 500 company) whose next earnings date could not be found: unlike an index fund it
    # reports, so a position that avoids or sells volatility cannot tell whether an announcement is inside it
    earnings_unknown: bool = False

    @property
    def usable(self) -> bool:
        return self.chain is not None and self.features is not None and bool(self.chain.quotes)

    @property
    def regime(self) -> str | None:
        return trend_regime(self.features) if self.features is not None else None

    @property
    def vol_regime(self) -> str | None:
        return iv_regime(self.features) if self.features is not None else None

    def summary(self) -> dict[str, Any]:
        f = self.features
        return {
            "underlying": self.underlying,
            "spot": self.spot,
            "atm_iv_30d": self.atm_iv_30d,
            "iv_rank": f.iv_rank if f else None,
            "iv_percentile": f.iv_percentile if f else None,
            "iv_rv": f.iv_rv if f else None,
            "rv20": f.rv20 if f else None,
            "term": self.term,
            "skew_25d": f.skew if f else None,
            "event_days": f.event_days if f else None,
            "regime": self.regime,
            "vol_regime": self.vol_regime,
            "iv_trend": iv_trend(f) if f else None,
            "quality": self.quality,
            "problems": self.problems,
        }


async def closes_of(market: Any, symbol: str) -> dict[date, float]:
    """Daily closes (real data only: synthetic prices are never used for a decision)."""
    r = await market.history(symbol, "1d", lookback_days=HISTORY_DAYS)
    if r.status is DataStatus.SYNTHETIC:
        return {}
    return {b.timestamp.astimezone(NEW_YORK).date(): float(b.close) for b in r.value.bars}


async def iv_history(db: Database, underlying: str, before: date) -> dict[date, float]:
    async with db.session() as s:
        rows = (await s.scalars(select(OptionsIVHistoryRow).where(
            OptionsIVHistoryRow.underlying == underlying, OptionsIVHistoryRow.day < before,
            OptionsIVHistoryRow.day >= before - timedelta(days=400)))).all()  # fmt: skip
    return {r.day: r.atm_iv_30d for r in rows if r.atm_iv_30d}


async def perceive(
    underlying: str,
    *,
    data: OptionsMarketDataProvider,
    market: Any,
    db: Database,
    now: datetime,
    min_dte: int,
    max_dte: int,
    next_earnings: date | None = None,
    closes: dict[date, float] | None = None,
) -> UnderlyingView:
    view = UnderlyingView(underlying, now)
    today = now.astimezone(NEW_YORK).date()
    try:
        chain = await data.chain(underlying, expiration_from=today + timedelta(days=1),
                                 expiration_to=today + timedelta(days=max_dte + 14))  # fmt: skip
    except Exception as exc:  # no chain: nothing option-related is decided on this underlying
        view.problems.append(f"option chain unavailable ({type(exc).__name__}: {str(exc)[:120]})")
        return view
    view.chain, view.spot = chain, chain.underlying_price
    view.quality = chain.quality(now)
    view.expiries = by_expiry(chain.quotes, chain.underlying_price, now)
    view.term = term_structure(view.expiries)
    view.atm_iv_30d = constant_maturity_iv(view.expiries, 30)
    if closes is None:
        try:
            closes = await closes_of(market, underlying)
        except Exception as exc:
            view.problems.append(f"price history unavailable ({type(exc).__name__})")
            closes = {}
    if not closes:
        view.problems.append("no real price history: features unknown")
        return view
    series = dict(closes)
    series[today] = chain.underlying_price  # today's point: the live price
    view.closes = series
    past_iv = await iv_history(db, underlying, today)
    if view.atm_iv_30d is not None:
        past_iv[today] = view.atm_iv_30d
    feats = history(series, past_iv, [next_earnings] if next_earnings else [])
    f = feats.get(today)
    if f is not None:
        f.term_shape = (
            str(view.term.get("shape")) if view.term.get("shape") not in (None, "unknown") else None
        )
        front = next((e for e in view.expiries if e.skew is not None and e.dte >= min_dte), None)
        f.skew = front.skew if front is not None else None
        f.extra["atm_iv_30d"] = view.atm_iv_30d
    view.features = f
    return view


async def record(db: Database, view: UnderlyingView, now: datetime) -> None:
    """Today's IV standing (one row per underlying per day, the latest reading wins) and a chain summary."""
    if view.chain is None:
        return
    today = now.astimezone(NEW_YORK).date()
    f = view.features
    async with db.session() as s:
        s.add(OptionsChainSnapshotRow(underlying=view.underlying, fetched_at=view.chain.fetched_at, feed=view.chain.feed,
                                      source=view.chain.source, underlying_price=view.chain.underlying_price,
                                      underlying_at=view.chain.underlying_at, contracts=len(view.chain.quotes),
                                      quality=_plain(view.quality)))  # fmt: skip
        if view.atm_iv_30d is None:
            return
        row = await s.scalar(select(OptionsIVHistoryRow).where(OptionsIVHistoryRow.underlying == view.underlying,
                                                                OptionsIVHistoryRow.day == today))  # fmt: skip
        if row is None:
            row = OptionsIVHistoryRow(underlying=view.underlying, day=today, feed=view.chain.feed,
                                      source=view.chain.source)  # fmt: skip
            s.add(row)
        row.atm_iv_30d = view.atm_iv_30d
        row.iv_rank = f.iv_rank if f else None
        row.iv_percentile = f.iv_percentile if f else None
        row.rv_20 = f.rv20 if f else None
        row.rv_60 = f.rv60 if f else None
        row.term_slope = view.term.get("slope_per_30d")
        row.term_shape = view.term.get("shape")
        row.skew_25d = f.skew if f else None
        front = next((e for e in view.expiries if e.implied_move_pct is not None), None)
        row.implied_move = front.implied_move_pct if front else None
        row.feed, row.details = view.chain.feed, {"expiries": len(view.expiries)}


def _plain(x: Any) -> Any:
    from quantpulse.services.options_lab import jsonable

    return jsonable(x)
