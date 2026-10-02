"""The 20/40/60-session evaluation over recorded trading days (test data through the real tables), on the
fake Alpaca paper account: fixed checkpoints, the Brain beside the benchmark and the replaced strategy, and
never a verdict of success or failure."""

import json
from datetime import date

import numpy as np
import pytest

from quantpulse.brain.checkpoints import significance
from quantpulse.core.clock import FakeClock
from quantpulse.core.market_calendar import next_trading_day
from quantpulse.db.models import BrainSessionRow

from .conftest import NOW
from .test_brain_cycle import brain_client
from .test_brain_execution import API, OWNS


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def test_the_significance_statement_never_declares_a_verdict():
    rng = np.random.default_rng(3)
    bench = list(rng.normal(0.0004, 0.01, 20))
    same = [b + float(rng.normal(0, 0.002)) for b in bench]
    s = significance(same, bench)
    assert s["verdict"] == "none" and s["finding"] == "indistinguishable from the benchmark"
    assert "cannot establish skill" in s["statement"] and s["ci95_annual"][0] < 0 < s["ci95_annual"][1]
    ahead = [b + 0.004 + float(rng.normal(0, 0.001)) for b in bench]  # far ahead, every day
    s = significance(ahead, bench)
    assert s["finding"].startswith("notably ahead") and s["verdict"] == "none"
    assert "cannot establish skill" in s["statement"]  # 20 sessions: still no conclusion
    assert significance([0.01], [0.0])["statement"] == "too few sessions for any statistic"


async def test_checkpoints_report_every_dimension_and_judge_nothing(tmp_path):
    clock = FakeClock(NOW)
    rng = np.random.default_rng(7)
    async for api in brain_client(tmp_path, clock, **OWNS):
        days, day, equity = [], date(2026, 7, 1), 100_000.0
        async with api.container.db.session() as s:
            for i in range(45):
                ret, bench, shadow = (float(x) for x in rng.normal([0.0006, 0.0004, 0.0003], 0.01))
                start_eq, equity = equity, equity * (1 + ret)
                s.add(BrainSessionRow(day=day, owner="brain", equity_open=start_eq, equity_close=equity, day_return=ret,
                                      benchmark_return=bench, exposure=0.6, positions=4, orders_sent=2,
                                      orders_filled=2, traded_notional=8_000.0, cycles=13,
                                      data_blocked_cycles=1 if i % 9 == 0 else 0,
                                      halts={"data_quality": 1} if i % 9 == 0 else {},
                                      premarket={}, near_close={},
                                      close={"strategy_shadow": {"day_return": shadow, "turnover": 5_000.0}},
                                      updated_at=NOW))  # fmt: skip
                days.append(day)
                day = next_trading_day(day)
        out = (await api.get(f"{API}/checkpoints")).json()
        assert out["sessions"] == 45 and "no window declares" in out["principle"]
        cp = out["checkpoints"]
        assert cp["20"]["status"] == "reached — reported, not judged" and cp["20"]["sessions"] == 20
        assert cp["40"]["status"].startswith("reached") and cp["40"]["last_day"] == days[39].isoformat()
        assert cp["60"] == {"status": "not reached: 45 of 60 sessions", "progress": 0.75}
        for w in (cp["20"], cp["40"], out["rolling"], out["so_far"]):
            for section in ("brain", "benchmark", "shadow", "significance", "pnl", "turnover", "execution",
                            "calibration", "decision_quality", "risk", "agent_reliability"):  # fmt: skip
                assert section in w, section
            assert (
                w["significance"]["verdict"] == "none"
                and "cannot establish skill" in w["significance"]["statement"]
            )
            assert w["brain"]["max_drawdown"] <= 0 and w["brain"]["volatility"] > 0
            assert w["shadow"]["brain_vs_shadow"]["excess_return_annual"] is not None
            assert 0.06 < w["turnover"]["per_session"] < 0.11  # 8,000 a day against ~100,000 of equity
            assert w["risk"]["data_blocked_cycles"] >= 1 and w["risk"]["halts"]["data_quality"] >= 1
            assert w["calibration"]["status"] == "unproven" and w["decision_quality"]["status"] == "unproven"
        assert (
            cp["20"]["pnl"]["unrealised"] is None and cp["20"]["pnl"]["unrealised_note"]
        )  # the past is not re-marked
        assert out["so_far"]["pnl"]["unrealised"] == 0.0  # nothing open now
        assert out["rolling"]["first_day"] == days[25].isoformat()
        text = json.dumps({k: v for k, v in out.items() if k != "principle"}).lower()
        assert "successful" not in text  # (the principle itself says no window declares it)
