"""Reviews without a server: which findings become proposals, and that a protected control is only ever
recorded for a person's review."""

from datetime import UTC, date, datetime

from quantpulse.brain.improvement import ImprovementEngine
from quantpulse.brain.memory import MemoryStore
from quantpulse.brain.reviews import Reviewer, last_session_of_week
from quantpulse.config import Settings
from quantpulse.core.clock import FakeClock

NOW = datetime(2026, 9, 25, 21, 0, tzinfo=UTC)


def group(status, decisive=40):
    return {"status": status, "decisive": decisive, "verdicts": {"missed": 30, "avoided": 10},
            "avoided_share": 0.25, "ci95": [0.14, 0.4], "mean_favourable": 0.01, "meaning": "m", "needs": 0}  # fmt: skip


def test_the_weeks_last_session():
    assert last_session_of_week(date(2026, 9, 25))  # a Friday
    assert not last_session_of_week(date(2026, 9, 24))
    assert last_session_of_week(date(2026, 4, 2))  # Thursday: Good Friday (April 3, 2026) is a market holiday
    assert not last_session_of_week(date(2026, 11, 25))  # the Friday after Thanksgiving is a (half-day) session
    assert not last_session_of_week(date(2026, 9, 26))  # Saturday: not a session


async def test_costly_rejections_and_alerts_become_proposals_protected_ones_for_review_only(database):
    engine = ImprovementEngine(database, 30, 0.45)
    reviewer = Reviewer(database, Settings(_env_file=None), FakeClock(NOW), MemoryStore(database), engine)
    ideas = {"by_reason": {
        "low_confidence": group("costing opportunities"),  # the Brain's own threshold: a normal proposal
        "risk_engine": group("costing opportunities"),  # a protected control: review only
        "data_quality": group("costing opportunities"),
        "slot_limit": group("right more often than not"),  # working: nothing to propose
        "market_closed": group("costing opportunities"),  # nothing anyone could do
    }}  # fmt: skip
    findings = [
        {
            "code": "round_trips",
            "severity": "alert",
            "finding": "AAA (3) round trips",
            "sample": 8,
            "evidence": {},
        },
        {
            "code": "concentration",
            "severity": "warning",
            "finding": "largest at 29%",
            "sample": 3,
            "evidence": {},
        },
        {"code": "turnover", "severity": "info", "finding": "low", "sample": 5, "evidence": {}},
    ]
    proposals = reviewer._proposals(ideas, findings)
    assert {p["target"] for p in proposals} == {"low_confidence", "risk_engine", "data_quality", "round_trips",
                                                "concentration"}  # fmt: skip
    written = {p["target"]: p for p in await engine.submit(proposals, NOW)}
    assert written["low_confidence"]["status"] == "proposed"
    assert written["round_trips"]["status"] == "proposed"
    for protected in ("risk_engine", "data_quality", "concentration"):  # limits, freshness, position size
        assert written[protected]["status"] == "protected_review", protected
        assert written[protected]["evidence"]["protected_control"]
    assert {p["target"] for p in await engine.proposals("proposed")} == {"low_confidence", "round_trips"}
