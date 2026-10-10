"""Evaluating the Brain, end to end on the fakes: the replaced strategy as a shadow (it never trades),
execution quality from real paper fills, the separated scorecard, and the 60-session evaluation."""

from datetime import date, timedelta

import pytest
from sqlalchemy import func, select

from quantpulse.core.clock import FakeClock
from quantpulse.db.models import BrainSessionRow, BrokerOrderRow, TradingCycleRow

from .conftest import NOW
from .test_brain_cycle import brain_client, run_cycle, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS, posts


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


async def test_the_replaced_strategy_runs_as_a_shadow_and_never_sends_an_order(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        assert (await api.get(f"{API}/shadow")).json()["started"] is False
        done = await api.container.brain.supervisor.tick()
        assert "strategy_shadow" in done
        shadow = (await api.get(f"{API}/shadow")).json()
        assert shadow["started"] and shadow["capital"] == api.fake.equity()
        assert (
            shadow["trades"] == len(shadow["fills"]) and shadow["fills"]
        )  # the strategy bought in its shadow
        spent = sum(f["qty"] * f["price"] + f["cost"] for f in shadow["fills"] if f["side"] == "buy")
        assert shadow["cash"] == pytest.approx(
            shadow["capital"] - spent, abs=0.05
        )  # every dollar accounted for
        assert all("spread" in f["priced"] for f in shadow["fills"])  # modelled fills, and they say so
        assert posts(api.fake) == []  # the shadow never reaches Alpaca …
        async with api.container.db.session() as s:  # … nor the trading records
            assert await s.scalar(select(func.count()).select_from(BrokerOrderRow)) == 0
            cycles = (await s.scalars(select(TradingCycleRow))).all()
        assert all(c.trigger.startswith("brain") for c in cycles)
        assert set(shadow["positions"]) == {f["symbol"] for f in shadow["fills"] if f["side"] == "buy"}


async def test_execution_quality_is_measured_from_real_paper_fills(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        cycle = await run_cycle(api)
        q = (await api.get(f"{API}/execution-quality")).json()
        assert q["sent"] == q["filled"] == cycle["summary"]["orders_sent"] > 0
        assert q["fill_rate"] == 1.0 and q["fills_measured"] == q["filled"] and q["rejected"] == 0
        assert q["slippage_bps_mean"] is not None and abs(q["slippage_bps_mean"]) < 100
        card = (await api.get(f"{API}/scorecard")).json()
        assert card["execution_quality"]["filled"] == q["filled"]
        for part in ("prediction_accuracy", "decision_quality", "benchmark_relative"):
            assert card[part]["status"].startswith("unproven")  # nothing graded yet: nothing claimed
        assert (
            card["prediction_accuracy"]["hit_rate"] is None and card["agent_reliability"]["established"] == []
        )


async def seed_sessions(api, returns: list[tuple[float, float, float]]) -> None:
    """Recorded trading days (test data): (Brain, benchmark, shadow) daily returns."""
    day = date(2026, 6, 1)
    equity = shadow = 100_000.0
    async with api.container.db.session() as s:
        for brain, bench, shadow_ret in returns:
            while day.weekday() >= 5:
                day += timedelta(days=1)
            before, equity = equity, equity * (1 + brain)
            shadow *= 1 + shadow_ret
            s.add(BrainSessionRow(day=day, owner="brain", equity_open=before, equity_close=equity, day_return=brain,
                                  benchmark_return=bench, exposure=0.8, positions=5, orders_sent=2, orders_filled=2,
                                  traded_notional=10_000.0, cycles=13, data_blocked_cycles=0, halts={}, premarket={},
                                  close={"strategy_shadow": {"equity": shadow, "day_return": shadow_ret,
                                                             "turnover": 50_000.0}},
                                  updated_at=api.container.clock.now()))  # fmt: skip
            day += timedelta(days=1)


async def test_the_evaluation_reports_progress_toward_sixty_sessions(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS):
        empty = (await api.get(f"{API}/evaluation")).json()
        assert empty["sessions"] == 0 and empty["status"].startswith("in progress: 0 of 60")
        await seed_sessions(api, [(0.01, 0.005, 0.002), (-0.004, -0.002, 0.001), (0.006, 0.004, -0.003)] * 5)
        ev = (await api.get(f"{API}/evaluation")).json()
        assert ev["sessions"] == 15 and "too few to judge" in ev["status"]
        brain = ev["brain"]
        assert brain["sessions"] == 15
        assert brain["total_return"] == pytest.approx((1.01 * 0.996 * 1.006) ** 5 - 1, abs=1e-5)
        assert brain["excess_return_annual"] > 0 and brain["beta"] > 0 and brain["max_drawdown"] <= 0
        assert ev["benchmark"]["sessions"] == 15 and ev["previous_strategy"]["sessions"] == 15
        assert ev["previous_strategy"]["total_return"] == pytest.approx(
            (1.002 * 1.001 * 0.997) ** 5 - 1, abs=1e-5
        )
        assert ev["turnover"] > 0 and ev["previous_strategy"]["turnover"] > 0
        assert "no optimisation targets this report: it exists for a person's review" in ev["caveats"]
        await seed_sessions_more(api)
        ev = (await api.get(f"{API}/evaluation")).json()
        assert ev["sessions"] == 60 and ev["status"].startswith("ready for review")


async def seed_sessions_more(api) -> None:
    async with api.container.db.session() as s:
        last = await s.scalar(select(func.max(BrainSessionRow.day)))
    day = last + timedelta(days=1)
    async with api.container.db.session() as s:
        for _ in range(45):
            while day.weekday() >= 5:
                day += timedelta(days=1)
            s.add(BrainSessionRow(day=day, owner="brain", equity_open=1.0, equity_close=1.0, day_return=0.0,
                                  benchmark_return=0.0, orders_sent=0, orders_filled=0, traded_notional=0.0, cycles=0,
                                  data_blocked_cycles=0, halts={}, premarket={}, close={},
                                  updated_at=api.container.clock.now()))  # fmt: skip
            day += timedelta(days=1)
