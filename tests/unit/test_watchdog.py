"""The supervisor watchdog's verdict: a restart only for a stalled supervisor, never for a fail-closed or
deliberate state."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.services.watchdog import (
    ATTEMPT_STALLED,
    LEADER_SILENT,
    TICK_HUNG_CLOSED,
    TICK_HUNG_OPEN,
    SupervisorView,
    supervisor_liveness,
)

NOW = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
MIN = timedelta(minutes=1)

HEALTHY = SupervisorView(
    now=NOW,
    started_at=NOW - timedelta(hours=3),
    enabled=True,
    polling_enabled=True,
    poller_running=True,
    stopping=False,
    paused=False,
    leader=True,
    standby=None,
    waiting=None,
    last_attempt_at=NOW - MIN,
    tick_started_at=None,
    last_tick_at=NOW - MIN,
    leader_tick_at=NOW - MIN,
    market_open=True,
)


def verdict(**changes):
    return supervisor_liveness(replace(HEALTHY, **changes))


def test_a_supervisor_that_ticks_is_ok():
    assert verdict() == supervisor_liveness(HEALTHY)
    assert (verdict().verdict, verdict().restart) == ("ok", False)


@pytest.mark.parametrize(
    ("changes", "why"),
    [
        ({"tick_started_at": NOW - TICK_HUNG_OPEN - MIN}, "hung: the current tick has run for 21 min"),
        ({"tick_started_at": NOW - TICK_HUNG_CLOSED - MIN, "market_open": False}, "hung"),
        ({"last_attempt_at": NOW - ATTEMPT_STALLED - MIN}, "has not asked for a tick for 11 min"),
        ({"poller_running": False}, "the background scheduler is not running"),
        ({"last_attempt_at": None, "started_at": NOW - ATTEMPT_STALLED - MIN}, "no tick was asked for"),
        # a stall outranks a pause or a fail-closed wait: those only count while ticks are still asked for
        ({"last_attempt_at": NOW - timedelta(hours=1), "paused": True}, "has not asked"),
        ({"last_attempt_at": NOW - timedelta(hours=1), "waiting": "Alpaca unreachable"}, "has not asked"),
    ],
)
def test_a_stalled_supervisor_is_restarted(changes, why):
    out = verdict(**changes)
    assert out.verdict == "stalled" and out.restart, out
    assert why in out.reason


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        # a long tick within its limit is work, not a hang (off-hours ticks may run longer)
        ({"tick_started_at": NOW - TICK_HUNG_OPEN + MIN}, "ok"),
        ({"tick_started_at": NOW - TICK_HUNG_OPEN - MIN, "market_open": False}, "ok"),
        ({"last_attempt_at": None, "started_at": NOW - 2 * MIN}, "starting"),
        # fail-closed: recovery refuses to resume (Alpaca down, the audit failed) — a restart would not fix it
        ({"waiting": "the execution audit failed: paper_endpoint"}, "blocked"),
        ({"leader": True, "last_tick_at": NOW - ATTEMPT_STALLED - MIN}, "blocked"),
        # another process holds the lease: restarting this one cannot help
        ({"leader": False, "standby": "another process supervises the Brain (h1)"}, "standby"),
        (
            {"leader": False, "standby": "another (h1)", "leader_tick_at": NOW - LEADER_SILENT - MIN},
            "standby",
        ),
        # a person's decision, a switch, or a stop in progress
        ({"paused": True}, "paused"),
        ({"enabled": False, "poller_running": False}, "not_applicable"),
        ({"polling_enabled": False, "poller_running": False}, "not_applicable"),
        ({"stopping": True, "tick_started_at": NOW - timedelta(hours=5)}, "not_applicable"),
    ],
)
def test_nothing_else_is_restarted(changes, expected):
    out = verdict(**changes)
    assert (out.verdict, out.restart) == (expected, False), out


def test_a_silent_leader_elsewhere_is_named_but_not_restarted_from_here():
    out = verdict(leader=False, standby="another process supervises the Brain (h1)",
                  leader_tick_at=NOW - LEADER_SILENT - MIN)  # fmt: skip
    assert "has not ticked for 11 min" in out.reason and "restarting this one does not help" in out.reason
