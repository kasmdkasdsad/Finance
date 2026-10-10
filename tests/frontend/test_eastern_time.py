"""Every time on the dashboard is New York (Eastern) time, the market's own clock, daylight saving included:
the API's UTC times become ET in text, in dates and in tables (which stay real times, so they still sort)."""

from datetime import UTC, datetime

import pandas as pd

from frontend import ui


def test_times_read_in_eastern_time_with_daylight_saving():
    assert ui.when("2026-10-01T14:05:09Z") == "Oct 1, 10:05 AM ET"  # EDT, UTC−4
    assert ui.when("2026-12-01T17:00:00+00:00") == "Dec 1, 12:00 PM ET"  # EST, UTC−5
    assert ui.when("2026-10-02T06:31:29.5+00:00", seconds=True) == "Oct 2, 2:31:29 AM ET"
    assert ui.when(datetime(2026, 10, 1, 20, 0, tzinfo=UTC)) == "Oct 1, 4:00 PM ET"  # the close
    assert ui.when("2026-10-01T14:05:09") == "Oct 1, 10:05 AM ET"  # no offset: the API's times are UTC
    assert ui.when(None) == "—" and ui.when("") == "—" and ui.when("not a time") == "not a time"


def test_dates_are_new_york_dates():
    assert ui.day("2026-10-02T01:30:00Z") == "2026-10-01"  # 9:30 PM the evening before in New York
    assert ui.day("2026-10-02T14:00:00Z") == "2026-10-02"
    assert ui.day(None) == "—"


def test_table_time_columns_become_eastern_and_say_so():
    df = pd.DataFrame(
        {
            "at": ["2026-10-01T14:05:09Z", None],
            "created_at": ["2026-10-01T13:30:00.123+00:00", "2026-12-01T14:30:00Z"],
            "symbol": ["UPA", "UPB"],
        }
    )
    config = ui.et_times(df)
    assert set(config) == {"at", "created_at"}  # "symbol" is not a time
    assert str(df["created_at"].dt.tz) == "America/New_York"
    assert [t.strftime("%H:%M") for t in df["created_at"]] == ["09:30", "09:30"]  # the open, EDT and EST
    assert pd.isna(df["at"].iloc[1])
    assert config["created_at"]["label"] == "Created (ET)" and config["at"]["label"] == "When (ET)"
