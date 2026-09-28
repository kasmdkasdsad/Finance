"""Every paper trade traceable end to end, on the fake Alpaca paper API and the fake market feed:

opportunity → agents → evidence → disagreement → consensus → portfolio fit → risk preview → execution → fill
→ P&L → benchmark-relative outcome → prediction grade → decision-quality grade → lesson.

A trade is followed from the buy, through its position and a stop-loss exit, to the graded call, the
reflection and the trade lesson. The completeness report must show it complete: no link missing, nothing
still to come. Links that come later are *pending* until then, never reported as missing.
"""

from datetime import date

import pytest

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

        report = (await api.get(f"{API}/traces")).json()
        assert report["with_gaps"] == 0 and report["trades"] >= 2
        mine = next(t for t in report["trades_detail"] if t["decision_id"] == buy["id"])
        assert mine["complete"] and mine["gaps"] == [] and mine["pending"] == []
