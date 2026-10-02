"""Every paper trade traceable end to end, on the fake Alpaca paper API and the fake market feed:

opportunity → agents → evidence → disagreement → consensus → portfolio fit → risk preview → execution → fill
→ P&L → benchmark-relative outcome → prediction grade → decision-quality grade → lesson.

A trade is followed from the buy, through its position and a stop-loss exit, to the graded call, the
reflection and the trade lesson. The completeness report must show it complete: no link missing, nothing
still to come. Links that come later are *pending* until then, never reported as missing.
"""

from datetime import date

import pytest

from quantpulse.brain.opportunity_outcomes import REASONS
from quantpulse.core.clock import FakeClock
from tests.fakes.alpaca_paper import FakeAlpacaPaper

from .conftest import NOW
from .test_brain_cycle import brain_client, extend_feed, run_cycle, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS, sent


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def stages(trail: dict) -> dict[str, dict]:
    return {s["stage"]: s for s in trail["stages"]}


async def test_a_closed_and_graded_trade_is_traceable_from_the_idea_to_the_lesson(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        first = await run_cycle(api)
        buy = next(d for d in sent(first) if d["action"] == "buy")
        symbol = buy["subject"]
        clock.advance(31 * 60)
        await run_cycle(api)  # the fill becomes a position with its thesis
        fake.prices[symbol] = fake.positions[symbol]["avg"] * 0.85  # 15% down: past its stop
        clock.advance(31 * 60)
        exit_cycle = await run_cycle(api)
        exit_ = next(d for d in exit_cycle["decisions"] if d["subject"] == symbol)
        assert exit_["action"] == "close" and exit_["execution"]["sent"], exit_
        clock.advance(31 * 60)
        await run_cycle(api)  # the position is gone at Alpaca: its thesis closes with the Brain's exit

        extend_feed(api.feed, date(2027, 2, 5))
        clock.advance(130 * 86400)  # every horizon has matured (the fundamental agents look 63 sessions out)
        learned = (await api.post(f"{API}/learn")).json()
        assert learned["evaluated"] > 0 and learned["reflections"] > 0
        assert (await api.container.brain.trade_lessons())["lessons"] >= 1

        trail = (await api.get(f"{API}/decisions/{buy['id']}/audit")).json()
        st = stages(trail)
        assert trail["gaps"] == [] and trail["pending"] == [], (trail["gaps"], trail["pending"])
        for name in ("data", "agents", "opinions", "evidence", "disagreement", "consensus", "portfolio_fit",
                     "portfolio_decision", "risk_check", "order", "alpaca_response", "execution", "fill",
                     "position", "pnl", "benchmark_relative", "prediction_grade", "decision_quality", "lesson"):  # fmt: skip
            assert st[name]["status"] == "done", (name, st[name])
        assert (
            st["pnl"]["detail"]["kind"] == "realised" and st["pnl"]["detail"]["position_status"] == "closed"
        )
        rel = st["benchmark_relative"]["detail"]
        assert rel["final"] is True and rel["relative"] == pytest.approx(
            rel["return_pct"] - rel["benchmark_return"]
        )
        graded = [p for p in st["prediction_grade"]["detail"]["predictions"] if p["status"] == "evaluated"]
        assert graded and all(p["hit"] in (True, False) and p["relative"] is not None for p in graded)
        [r, *_] = st["decision_quality"]["detail"]["reflections"]
        assert r["decision_quality"] in ("good", "fair", "poor") and r["category"]
        trade_lesson = st["lesson"]["detail"]["trade_lesson"]
        assert (
            trade_lesson and trade_lesson["data"]["ended_by"] == "stop" and symbol in trade_lesson["summary"]
        )
        assert "ended by stop" in trade_lesson["summary"]
        assert st["position"]["detail"]["exit_reason"]

        exit_trail = (await api.get(f"{API}/decisions/{exit_['id']}/audit")).json()
        assert exit_trail["gaps"] == [] and stages(exit_trail)["pnl"]["detail"]["kind"] == "realised"

        # the learning report: everything measured, nothing judged on one day's calls
        lr = (await api.get(f"{API}/learning-report")).json()
        assert lr["graded_calls"] == learned["evaluated"] and lr["min_observations"] == 30
        assert lr["consensus"]["record"]["verdict"] == "unproven"
        assert (
            lr["consensus"]["calibration"]["status"] == "unproven"
            and lr["consensus"]["calibration"]["needs"] > 0
        )
        assert {"technical", "momentum"} <= set(lr["agents"])
        for a in lr["agents"].values():
            assert a["record"]["verdict"] == "unproven" and a["needs"] > 0
            assert a["versus_consensus"]["status"] == "unproven"
        assert lr["agents"]["technical"]["versus_consensus"]["pairs"] > 0
        cells = [c for source in lr["regimes"].values() for c in source.values()]
        assert cells and all(c["verdict"] == "unproven" for c in cells)
        assert lr["data"]["on_usable_data"]["calls"] > 0 and lr["caveats"]
        graded = [p for p in await api.container.brain.store.predictions() if p.status == "evaluated"]
        assert graded and all("market_event" in p.context for p in graded)

        behaviour = (await api.get(f"{API}/behavior", params={"days": 365})).json()
        assert len(behaviour["checked"]) == 9 and behaviour["headline"]
        for f in behaviour["findings"]:
            assert f["severity"] in ("info", "warning", "alert") and f["finding"] and f["sample"] >= 0
        assert any(
            f["code"] == "concentration" or f["code"] == "consensus_instability"
            for f in behaviour["findings"]
        )

        report = (await api.get(f"{API}/traces")).json()
        assert report["with_gaps"] == 0 and report["trades"] >= 2
        mine = next(t for t in report["trades_detail"] if t["decision_id"] == buy["id"])
        assert mine["complete"] and mine["gaps"] == [] and mine["pending"] == []


async def test_ideas_not_taken_are_recorded_once_a_day_and_graded_later(tmp_path, monkeypatch):
    """The Brain learns from the trades it rejected: every idea it considered is kept with why it was not
    taken, folded to one record per idea per day, and graded against the benchmark after its horizon."""
    from quantpulse.core.market_calendar import NEW_YORK

    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        first = await run_cycle(api)
        assert first["summary"]["ideas_recorded"] > 0
        rows = (await api.get(f"{API}/opportunity-outcomes/rows")).json()
        assert len(rows) == first["summary"]["ideas_recorded"]
        assert all(r["state"] == "open" and r["entry_price"] and r["horizon_days"] > 0 for r in rows)
        assert all(
            r["direction"] in (-1, 1) and r["day"] == NOW.astimezone(NEW_YORK).date().isoformat()
            for r in rows
        )
        reasons = {r["reason"] for r in rows}
        assert reasons and reasons <= set(REASONS)
        assert all(r["reason_detail"] for r in rows if r["reason"] != "taken")
        clock.advance(31 * 60)
        second = await run_cycle(api)  # the same ideas again today: folded in, not new observations
        again = (await api.get(f"{API}/opportunity-outcomes/rows", params={"limit": 2000})).json()
        assert len(again) == len(rows) + second["summary"]["ideas_recorded"]
        assert sum(r["repeats"] for r in again) > 0
        report = (await api.get(f"{API}/opportunity-outcomes")).json()
        assert report["graded"] == 0 and report["open"] == len(again) and report["by_reason"] == {}

        extend_feed(api.feed, date(2027, 2, 5))
        clock.advance(130 * 86400)
        learned = (await api.post(f"{API}/learn")).json()
        assert learned["ideas_graded"] == len(again)
        report = (await api.get(f"{API}/opportunity-outcomes")).json()
        assert report["graded"] == len(again) and report["open"] == 0
        for g in report["by_reason"].values():  # far too few ideas to judge a rejection rule
            assert g["status"] == "unproven" and g["needs"] > 0 and g["meaning"]
        assert report["taken_vs_rejected"]["status"].startswith("unproven")
        graded = (await api.get(f"{API}/opportunity-outcomes/rows", params={"limit": 2000})).json()
        verdicts = {"missed", "avoided", "noise", "worked", "failed", "signal_right", "signal_wrong"}
        assert all(r["verdict"] in verdicts and r["relative"] is not None for r in graded)
        assert all(r["favourable"] == pytest.approx(r["direction"] * r["relative"], abs=1e-6) for r in graded)
        patterns = (
            await api.get(f"{API}/memory", params={"kind": "pattern", "subject": "@rejections"})
        ).json()
        assert patterns and all("rejected for" in p["summary"] for p in patterns)
