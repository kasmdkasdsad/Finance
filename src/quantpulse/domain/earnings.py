"""Earnings events: when the market reacts, how big the reactions are, and when the next one is due.

Timing
    An earnings release accepted by the SEC before that day's close (pre-market, or occasionally during
    the session) is priced at that day's close; one released after the close or on a non-trading day is
    priced by the next session. The **reaction return** is that session's close over the previous close.

Next date
    When a vendor calendar is configured its scheduled date wins; otherwise the next date is estimated
    as the last release plus the median gap between recent releases (companies keep a steady quarterly
    rhythm, usually within a week). Estimated dates are always labelled as such.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from quantpulse.core.market_calendar import (
    NEW_YORK,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    regular_close,
)

MAX_REACTIONS = 12


def reaction_day(announced_at: datetime) -> date:
    local = announced_at.astimezone(NEW_YORK)
    day = local.date()
    if is_trading_day(day) and local.time() < regular_close(day):
        return day
    return next_trading_day(day)


@dataclass(frozen=True, slots=True)
class Reaction:
    announced_at: datetime
    reaction_date: date
    stock_return: float
    benchmark_return: float | None

    @property
    def abnormal(self) -> float | None:
        return None if self.benchmark_return is None else self.stock_return - self.benchmark_return


def reactions(
    events: Sequence[datetime],
    closes: Mapping[date, float],
    benchmark: Mapping[date, float] | None = None,
    *,
    until: date | None = None,
) -> list[Reaction]:
    """Close-to-close returns on each reaction day that is covered by ``closes`` (and before ``until``)."""
    out: list[Reaction] = []
    for at in events:
        day = reaction_day(at)
        if until is not None and day > until:
            continue
        prev = previous_trading_day(day)
        if day not in closes or prev not in closes or closes[prev] <= 0:
            continue
        r = closes[day] / closes[prev] - 1.0
        b = None
        if benchmark is not None and day in benchmark and prev in benchmark and benchmark[prev] > 0:
            b = benchmark[day] / benchmark[prev] - 1.0
        out.append(Reaction(at, day, r, b))
    return out


def typical_move(rs: Sequence[Reaction], last: int = MAX_REACTIONS) -> float | None:
    """Root-mean-square reaction over the last ``last`` events (``None`` with fewer than 2)."""
    sample = [r.stock_return for r in rs[-last:]]
    if len(sample) < 2:
        return None
    return math.sqrt(sum(x * x for x in sample) / len(sample))


def estimate_next(events: Sequence[datetime], today: date) -> date | None:
    """Last release + the median gap of the last few releases, rolled forward past ``today``."""
    days = sorted({reaction_day(e) for e in events})
    if len(days) < 2:
        return None
    recent = days[-6:]
    gaps = [(b - a).days for a, b in itertools.pairwise(recent)]
    gaps = [g for g in gaps if 45 <= g <= 200]  # ignore amended or duplicate releases
    if not gaps:
        return None
    step = int(statistics.median(gaps))
    nxt = days[-1] + timedelta(days=step)
    while nxt < today - timedelta(days=7):  # the estimate went stale: project another cycle
        nxt += timedelta(days=step)
    return max(nxt, today)


def reports_before_close(events: Sequence[datetime]) -> bool:
    """Whether most past releases were priced on their own day (pre-market) rather than the next session."""
    if not events:
        return False
    same = sum(1 for e in events if reaction_day(e) == e.astimezone(NEW_YORK).date())
    return 2 * same > len(events)


def reaction_for_announcement(announced: date, events: Sequence[datetime]) -> date:
    """The session that will price a release scheduled for ``announced`` (vendor calendars give the date,
    not the time; the company's own habit decides between that day and the next session)."""
    if is_trading_day(announced) and reports_before_close(events):
        return announced
    return next_trading_day(announced)


def reaction_days_between(events: Sequence[datetime], start: date, end: date) -> list[date]:
    """Reaction days falling in ``[start, end]``."""
    return sorted({d for d in (reaction_day(e) for e in events) if start <= d <= end})
