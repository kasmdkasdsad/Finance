from datetime import UTC, date, datetime

import pytest

from quantpulse.core.market_calendar import (
    Session,
    easter_sunday,
    is_trading_day,
    next_open,
    nyse_early_closes,
    nyse_holidays,
    previous_trading_day,
    session_at,
    trading_days_between,
)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2038, date(2038, 4, 25)),
    ],
)
def test_easter(year, expected):
    assert easter_sunday(year) == expected


def test_2026_nyse_holidays():
    assert nyse_holidays(2026) == {
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    }
    assert nyse_early_closes(2026) == {date(2026, 11, 27), date(2026, 12, 24)}


def test_observance_rules():
    # 2022: Jan 1 was a Saturday -> no holiday on Fri Dec 31, 2021
    assert date(2021, 12, 31) not in nyse_holidays(2021)
    assert date(2022, 1, 1) not in nyse_holidays(2022)
    # 2023: Jan 1 was a Sunday -> observed Monday Jan 2
    assert date(2023, 1, 2) in nyse_holidays(2023)
    # 2025: special closure for President Carter's national day of mourning
    assert date(2025, 1, 9) in nyse_holidays(2025)
    # 2024: July 3 early close, Christmas Eve early close
    assert nyse_early_closes(2024) == {date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)}


def test_sessions():
    assert session_at(datetime(2026, 9, 25, 14, 0, tzinfo=UTC)) is Session.REGULAR  # 10:00 EDT Friday
    assert session_at(datetime(2026, 9, 25, 12, 0, tzinfo=UTC)) is Session.PRE  # 08:00 EDT
    assert session_at(datetime(2026, 9, 25, 21, 0, tzinfo=UTC)) is Session.POST  # 17:00 EDT
    assert session_at(datetime(2026, 9, 26, 15, 0, tzinfo=UTC)) is Session.CLOSED  # Saturday
    # Early close at 13:00 EST on the day after Thanksgiving
    assert session_at(datetime(2026, 11, 27, 18, 30, tzinfo=UTC)) is Session.POST
    with pytest.raises(ValueError):
        session_at(datetime(2026, 9, 25, 14, 0))


def test_navigation_helpers():
    assert is_trading_day(date(2026, 9, 25)) and not is_trading_day(date(2026, 9, 7))
    assert previous_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)  # skips Labor Day weekend
    assert next_open(datetime(2026, 9, 25, 21, 0, tzinfo=UTC)).date() == date(2026, 9, 28)
    assert next_open(datetime(2026, 9, 25, 12, 0, tzinfo=UTC)).hour == 9
    assert trading_days_between(date(2026, 9, 4), date(2026, 9, 11)) == 4
