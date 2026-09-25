"""NYSE trading calendar (holidays, early closes, sessions) used to pace polling.

Implements the NYSE's published holiday rules, including Saturday/Sunday observance and the
New Year's Day exception (when Jan 1 falls on a Saturday no holiday is observed on Friday Dec 31).
Unscheduled closures (e.g. national days of mourning) cannot be derived from rules; known ones are
listed in :data:`SPECIAL_CLOSURES`.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from enum import StrEnum
from functools import lru_cache
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
PRE_MARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)

SPECIAL_CLOSURES: frozenset[date] = frozenset(
    {
        date(2018, 12, 5),  # President George H.W. Bush national day of mourning
        date(2025, 1, 9),  # President Jimmy Carter national day of mourning
    }
)


class Session(StrEnum):
    PRE = "pre"
    REGULAR = "regular"
    POST = "post"
    CLOSED = "closed"


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month // 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:  # Saturday -> Friday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> Monday
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[date]:
    days: set[date] = set()
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        days.add(new_year + timedelta(days=1))
    elif new_year.weekday() != 5:  # Saturday New Year's is not observed on Dec 31
        days.add(new_year)
    days.add(_nth_weekday(year, 1, 0, 3))  # Martin Luther King Jr. Day
    days.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    days.add(easter_sunday(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    days.add(_observed(date(year, 7, 4)))  # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days.add(_observed(date(year, 12, 25)))  # Christmas
    days.update(d for d in SPECIAL_CLOSURES if d.year == year)
    return frozenset(days)


@lru_cache(maxsize=64)
def nyse_early_closes(year: int) -> frozenset[date]:
    days: set[date] = set()
    july3 = date(year, 7, 3)
    if july3.weekday() < 4:  # Mon-Thu, i.e. July 4 falls Tue-Fri
        days.add(july3)
    days.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))  # day after Thanksgiving
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 4:
        days.add(christmas_eve)
    return frozenset(days - nyse_holidays(year))


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


def regular_close(day: date) -> time:
    return EARLY_CLOSE if day in nyse_early_closes(day.year) else REGULAR_CLOSE


def session_at(moment: datetime) -> Session:
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    local = moment.astimezone(NEW_YORK)
    day = local.date()
    if not is_trading_day(day):
        return Session.CLOSED
    now = local.time()
    close = regular_close(day)
    if REGULAR_OPEN <= now < close:
        return Session.REGULAR
    if PRE_MARKET_OPEN <= now < REGULAR_OPEN:
        return Session.PRE
    if close <= now < AFTER_HOURS_CLOSE:
        return Session.POST
    return Session.CLOSED


def is_market_open(moment: datetime) -> bool:
    return session_at(moment) is Session.REGULAR


def previous_trading_day(day: date) -> date:
    candidate = day - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def next_trading_day(day: date) -> date:
    candidate = day + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def next_open(moment: datetime) -> datetime:
    """Next regular-session open at or after ``moment`` (aware, America/New_York)."""
    local = moment.astimezone(NEW_YORK)
    day = local.date()
    if is_trading_day(day) and local.time() < REGULAR_OPEN:
        return datetime.combine(day, REGULAR_OPEN, NEW_YORK)
    return datetime.combine(next_trading_day(day), REGULAR_OPEN, NEW_YORK)


def trading_days_between(start: date, end: date) -> int:
    """Number of NYSE trading days in the half-open interval (start, end]."""
    if end <= start:
        return 0
    count = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if is_trading_day(cursor):
            count += 1
        cursor += timedelta(days=1)
    return count


def upcoming_sessions(moment: datetime, n: int) -> list[date]:
    """The next ``n`` session dates whose close is still ahead of ``moment`` (today counts until its close)."""
    if n < 1:
        return []
    local = moment.astimezone(NEW_YORK)
    day = local.date()
    first = day if is_trading_day(day) and local.time() < regular_close(day) else next_trading_day(day)
    out = [first]
    while len(out) < n:
        out.append(next_trading_day(out[-1]))
    return out


def sessions_after(day: date, n: int) -> date:
    """The ``n``-th trading day after ``day``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    cursor = day
    for _ in range(n):
        cursor = next_trading_day(cursor)
    return cursor
