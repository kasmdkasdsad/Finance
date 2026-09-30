"""The Options Brain inside a real Brain cycle, end to end (fake Alpaca paper API, fake options market): a
validated strategy's candidate is deliberated, recorded with its thesis, traded in the shadow book and — only
at PAPER_ACTIVE — sent as a paper order through the trading service; later exits close both, and every
closed position feeds learning. Strategies still in research never trade."""

import pytest
from sqlalchemy import select

from quantpulse.core.clock import FakeClock
from quantpulse.db.options_models import (
    OptionsExecutionLedgerRow,
    OptionsLearningEventRow,
    OptionsPositionRow,
    OptionsStrategyVersionRow,
    OptionsTradeCandidateRow,
    OptionsTradeThesisRow,
)
from quantpulse.options.lab.genome import Genome
from quantpulse.options.lab.promotion import Stage, stage_record
from tests.fakes.options_market import FakeOptionsMarket

from .conftest import NOW
from .test_brain_cycle import brain_client, run_cycle

ENABLED = {"alpaca_trading_enabled": True, "trading_dry_run": False}
OWNS = {"brain_mode": "paper_execution"}
LONG_CALL = Genome("long_call", "bullish", entry_signal="always", event_filter="ignore", dte_min=14, dte_max=40,
                   delta_target=0.5, take_profit=0.5, stop_loss=0.5, exit_dte=5, risk_per_trade=0.01)  # fmt: skip


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


async def client(tmp_path, clock, **overrides):
    settings = {**OWNS, **ENABLED, "options_universe": "MIDA", "options_max_loss_per_trade": 1000.0,
                "options_max_loss_pct_per_trade": 0.02, "options_max_underlying_risk_pct": 0.05, **overrides}  # fmt: skip
    async for api in brain_client(tmp_path, clock, **settings):
        market = FakeOptionsMarket(clock, api.feed.live_price, broker=api.fake)
        market.vol_by["MIDA"] = 0.09  # fairly priced: about the stock's own realized volatility
        c = api.container
        c.trading.options_data = market
        c.options_brain.data = market
        api.market = market
        yield api


async def promote(api, genome: Genome, stage: Stage) -> int:
    """A strategy version as if the lab had validated it (test data through the lab's own writer)."""
    lab = api.container.options_lab
    now = api.container.clock.now()
    async with api.container.db.session() as s:
        v = await lab._add_version(
            s, genome, key="test-long-call", origin="seed", reason="test", generation=0, now=now
        )
        v.stage = stage.value
        v.stage_history = [*v.stage_history, stage_record(stage, now.isoformat(), "test",
                                                          {"latest": {"validation_ror": 0.08}})]  # fmt: skip
        return v.id


async def rows(api, model, **where):
    async with api.container.db.session() as s:
        q = select(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return (await s.scalars(q)).all()


async def test_a_validated_strategy_trades_shadow_and_paper_then_exits(tmp_path):
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        vid = await promote(api, LONG_CALL, Stage.PAPER_ACTIVE)
        cycle = await run_cycle(api)
        assert cycle["status"] == "completed", cycle.get("error")
        cands = await rows(api, OptionsTradeCandidateRow)
        assert cands and cands[0].version_id == vid and cands[0].status in ("submitted", "proposed"), [
            (c.status, c.gate, c.reject_reason) for c in cands
        ]
        assert await rows(api, OptionsTradeThesisRow)
        shadow = await rows(api, OptionsPositionRow, mode="shadow")
        paper = await rows(api, OptionsPositionRow, mode="paper")
        assert len(shadow) == 1 and shadow[0].status == "open"
        assert (
            len(paper) == 1 and paper[0].status == "open" and paper[0].client_order_id.startswith("qp-brain-")
        )
        opts = [b for b in api.fake.bodies if b.get("position_intent")]
        assert len(opts) == 1 and opts[0]["position_intent"] == "buy_to_open" and opts[0]["type"] == "limit"
        assert await rows(api, OptionsExecutionLedgerRow)
        # the underlying rallies: the call is worth far more — take profit on both books
        api.feed.live_move["MIDA"] = 0.12
        api.fake.prices["MIDA"] = api.feed.live_price("MIDA")
        clock.advance(35 * 60)
        await run_cycle(api)
        paper = await rows(api, OptionsPositionRow, mode="paper")
        shadow = await rows(api, OptionsPositionRow, mode="shadow")
        closed = [p for p in [*paper, *shadow] if p.status == "closed"]
        assert {p.mode for p in closed} == {"paper", "shadow"}, [
            (p.mode, p.status, p.exit_reason) for p in [*paper, *shadow]
        ]
        for p in closed:
            assert p.realized_pnl > 0 and p.exit_reason and p.critique
        sells = [b for b in api.fake.bodies if b.get("position_intent") == "sell_to_close"]
        assert len(sells) == 1
        events = await rows(api, OptionsLearningEventRow)
        assert {e.evidence for e in events} == {"paper", "shadow"}
        perf = await api.container.options_brain.performance()
        assert perf["paper"]["trades"] == 1 and perf["shadow"]["trades"] == 1


async def test_research_strategies_never_trade_and_shadow_never_sends_orders(tmp_path):
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock, options_exploration=False):
        await promote(api, LONG_CALL, Stage.WALK_FORWARD)  # validated in research only: nothing live
        await run_cycle(api)
        assert await rows(api, OptionsPositionRow) == []
        async with api.container.db.session() as s:
            v = (await s.scalars(select(OptionsStrategyVersionRow))).first()
            v.stage = Stage.PAPER_SHADOW.value
        clock.advance(35 * 60)
        await run_cycle(api)
        positions = await rows(api, OptionsPositionRow)
        assert positions and {p.mode for p in positions} == {"shadow"}
        assert not [
            b for b in api.fake.bodies if b.get("position_intent")
        ]  # a shadow trade is never an order


async def test_nothing_is_sent_when_the_brain_does_not_own_the_account(tmp_path):
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock, brain_mode="paper_recommendation"):
        await promote(api, LONG_CALL, Stage.PAPER_ACTIVE)
        await run_cycle(api)
        positions = await rows(api, OptionsPositionRow)
        assert {p.mode for p in positions} <= {"shadow"}
        assert not [b for b in api.fake.bodies if b.get("position_intent")]


async def test_expiration_is_recorded_as_the_occ_settles_it_and_learning_follows(tmp_path):
    from datetime import datetime, time

    from quantpulse.core.market_calendar import NEW_YORK, next_trading_day
    from quantpulse.db.options_models import OptionsExerciseEventRow, OptionsStrategyWeightRow

    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock, options_close_dte=0):
        await promote(api, LONG_CALL, Stage.PAPER_ACTIVE)
        await run_cycle(api)
        (paper,) = await rows(api, OptionsPositionRow, mode="paper")
        leg = paper.structure["legs"][0]["symbol"]
        # the position is still held at expiration (as if every exit had failed): the OCC settles it
        expiry_day = paper.first_expiration
        spot = api.feed.live_price("MIDA") * 1.2
        api.fake.expire(expiry_day, {"MIDA": spot})
        assert (
            leg not in api.fake.positions and api.fake.positions.get("MIDA", {}).get("qty", 0) > 0
        )  # exercised
        after = next_trading_day(expiry_day)
        clock.advance((datetime.combine(after, time(10, 30), NEW_YORK) - clock.now()).total_seconds())
        api.feed.live_move["MIDA"] = 0.2
        await run_cycle(api)
        (paper,) = await rows(api, OptionsPositionRow, mode="paper")
        assert (
            paper.status == "closed"
            and "settled by the OCC" in paper.exit_reason
            and "never exercises" in paper.exit_reason
        )
        events = await rows(api, OptionsExerciseEventRow)
        assert events and events[0].share_delivery > 0 and events[0].symbol == leg
        report = await api.container.options_brain.learn()
        assert report["weights"] >= 1 and report["trades"]["paper"] >= 1
        assert await rows(api, OptionsStrategyWeightRow)


async def test_option_failures_are_handled_and_alerted(tmp_path):
    """Failure simulation: Alpaca rejects the option order (no position, the candidate says why); a restart
    with an order still working picks it up on the next cycle; contracts opened outside QuantPulse and a
    position left close to expiration are alerted — nothing is traded from an alert."""
    from quantpulse.db.options_models import OptionsTradeCandidateRow as Cand

    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        await promote(api, LONG_CALL, Stage.PAPER_ACTIVE)
        api.fake.default_mode = "reject"
        await run_cycle(api)
        assert await rows(api, OptionsPositionRow, mode="paper") == []
        (cand,) = await rows(api, Cand)
        assert cand.status not in ("submitted",) and cand.reject_reason
        # the order rests at Alpaca (unfilled); the next cycle after a restart finds it filled
        api.fake.default_mode = "accept"
        clock.advance(35 * 60)
        await run_cycle(api)
        (pending,) = await rows(api, OptionsPositionRow, mode="paper")
        assert pending.status == "pending"
        api.fake.complete(pending.client_order_id)
        api.container.options_brain.last = None  # as after a restart: nothing held in memory
        clock.advance(35 * 60)
        await run_cycle(api)
        (opened,) = await rows(api, OptionsPositionRow, mode="paper")
        assert opened.status == "open" and opened.entry_value > 0
        # an option contract nobody in QuantPulse opened, and alerts for it
        api.fake.hold("MIDA261218P00150000", 1, 2.0)
        health = api.container.health
        await health._options_alerts()
        kinds = {a["kind"] for a in api.container.alerts.sent}
        assert "unexpected_option" in kinds
        assert not [
            b for b in api.fake.bodies if "MIDA261218P00150000" in str(b)
        ]  # never traded from an alert
        # a close that does not complete is alerted an hour after it began — not an hour after the opening
        async with api.container.db.session() as s:
            row = await s.get(OptionsPositionRow, opened.id)
            row.status = "closing"
            await s.commit()
        clock.advance(6 * 60)
        await health._options_alerts()
        assert "option_close_pending" not in {a["kind"] for a in api.container.alerts.sent}
        clock.advance(61 * 60)
        await health._options_alerts()
        assert "option_close_pending" in {a["kind"] for a in api.container.alerts.sent}
