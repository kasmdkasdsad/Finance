"""The Options Brain inside a real Brain cycle, end to end (fake Alpaca paper API, fake options market): a
validated strategy's candidate is deliberated, recorded with its thesis, traded in the shadow book and — only
at PAPER_ACTIVE — sent as a paper order through the trading service; later exits close both, and every
closed position feeds learning. Strategies still in research never trade."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
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
    # one underlying unless a test turns the scan on (the scenarios below are written around MIDA alone)
    settings = {**OWNS, **ENABLED, "options_universe": "MIDA", "options_max_loss_per_trade": 1000.0,
                "options_max_loss_pct_per_trade": 0.02, "options_max_underlying_risk_pct": 0.05,
                "options_scan_size": 0, **overrides}  # fmt: skip
    async for api in brain_client(tmp_path, clock, **settings):
        market = FakeOptionsMarket(clock, api.feed.live_price, broker=api.fake)
        market.vol_by["MIDA"] = 0.09  # fairly priced: about the stock's own realized volatility
        c = api.container
        c.trading.options_data = market
        c.options_brain.data = market
        api.market = market
        yield api


async def promote(
    api, genome: Genome, stage: Stage, key: str = "test-long-call", latest: dict | None = None
) -> int:
    """A strategy version as if the lab had validated it (test data through the lab's own writer)."""
    lab = api.container.options_lab
    now = api.container.clock.now()
    async with api.container.db.session() as s:
        v = await lab._add_version(s, genome, key=key, origin="seed", reason="test", generation=0, now=now)
        v.stage = stage.value
        v.stage_history = [*v.stage_history, stage_record(stage, now.isoformat(), "test",
                                                          {"latest": latest or {"validation_ror": 0.08}})]  # fmt: skip
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


PASSED = {"backtest_trades": 40, "ror_by_model": {"REALISTIC": 0.05, "PESSIMISTIC": 0.02}, "validation_ror": 0.08,
          "overfit_risk": 0.2, "walkforward_passed": True, "montecarlo_ruin": 0.001, "tail_passed": True,
          "beats_baselines": True, "critic_survived": True}  # fmt: skip


async def test_a_strategy_the_lab_promotes_is_traded_on_its_validated_edge(tmp_path):
    """The lab's own promotion carries the validated edge to the new stage record, so a strategy it promotes to
    PAPER_SHADOW is traded: shadow on live quotes and one exploration contract on paper. (The edge used to stay
    on the earlier record: every promoted strategy read as having none, and none ever traded.)"""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        lab = api.container.options_lab
        vid = await promote(api, LONG_CALL, Stage.WALK_FORWARD)
        async with api.container.db.session() as s:  # the lab's evaluation: every gate to PAPER_SHADOW passed
            v = await s.get(OptionsStrategyVersionRow, vid)
            last = v.stage_history[-1]
            v.stage_history = [
                *v.stage_history[:-1],
                {**last, "evidence": {"latest": PASSED, "fdr": {"discovery": True}}},
            ]
        promoted, _ = await lab._promote_all(clock.now())
        assert [(p["from"], p["to"]) for p in promoted] == [("WALK_FORWARD", "PAPER_SHADOW")]
        [eligible] = await lab.eligible_versions()
        assert eligible["stage"] == "PAPER_SHADOW" and eligible["expected_ror"] == 0.08
        await run_cycle(api)
        cands = await rows(api, OptionsTradeCandidateRow)
        assert cands and all(c.gate != "no validated edge" for c in cands), [
            (c.status, c.gate) for c in cands
        ]
        shadow = await rows(api, OptionsPositionRow, mode="shadow")
        paper = await rows(api, OptionsPositionRow, mode="paper")
        assert len(shadow) == 1 and shadow[0].version_id == vid
        assert len(paper) == 1 and paper[0].structure["exploration"] and paper[0].quantity == 1
        assert paper[0].max_loss <= api.container.settings.options_exploration_max_loss


async def test_a_promoted_strategy_with_no_positive_edge_still_never_trades(tmp_path):
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        lab = api.container.options_lab
        vid = await promote(api, LONG_CALL, Stage.WALK_FORWARD)
        async with api.container.db.session() as s:
            v = await s.get(OptionsStrategyVersionRow, vid)
            v.stage = Stage.PAPER_SHADOW.value  # promoted, but what the lab validated was not positive
            v.stage_history = [*v.stage_history[:-1], {**v.stage_history[-1], "evidence": {"latest": {**PASSED, "validation_ror": -0.01}}},
                               stage_record(Stage.PAPER_SHADOW, clock.now().isoformat(), "gate for PAPER_SHADOW passed",
                                            {"walkforward_passed": True})]  # fmt: skip
        [eligible] = await lab.eligible_versions()  # a history written before promotions carried the evidence
        assert eligible["expected_ror"] == -0.01
        await run_cycle(api)
        assert await rows(api, OptionsPositionRow) == []
        assert {c.gate for c in await rows(api, OptionsTradeCandidateRow)} == {"no validated edge"}
        assert not [b for b in api.fake.bodies if b.get("position_intent")]


async def test_with_exploration_a_validated_strategy_trades_one_capped_contract(tmp_path):
    """Exploration starts at VALIDATION: shadow on live quotes, and one contract on paper within the exploration
    limit, labelled exploration, on the evidence it has (here its backtest at pessimistic fills: the held-out
    test came out negative on model prices). A strategy with no positive backtest yet never trades."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        await promote(api, replace(LONG_CALL, delta_target=0.4), Stage.BACKTESTING, key="still-research")
        backtest_only = {"validation_ror": -0.02, "ror_by_model": {"REALISTIC": 0.05, "PESSIMISTIC": 0.03}}
        vid = await promote(api, LONG_CALL, Stage.VALIDATION, latest=backtest_only)
        [eligible] = await api.container.options_lab.eligible_versions()
        assert eligible["version_id"] == vid and eligible["expected_ror"] == 0.03
        assert eligible["edge_basis"] == "backtest at pessimistic fills"
        await run_cycle(api)
        positions = await rows(api, OptionsPositionRow)
        assert {p.version_id for p in positions} == {vid}  # nothing from the strategy still in research
        shadow = [p for p in positions if p.mode == "shadow"]
        paper = [p for p in positions if p.mode == "paper"]
        assert len(shadow) == 1 and len(paper) == 1
        assert paper[0].structure["exploration"] and paper[0].quantity == 1
        assert paper[0].max_loss <= api.container.settings.options_exploration_max_loss
        orders = [b for b in api.fake.bodies if b.get("position_intent")]
        assert len(orders) == 1 and orders[0]["position_intent"] == "buy_to_open"
        [cand] = [c for c in await rows(api, OptionsTradeCandidateRow) if c.gate is None]
        assert cand.audit["version"]["edge_basis"] == "backtest at pessimistic fills"


async def test_up_to_twenty_strategies_explore_and_the_untried_get_their_turn(tmp_path):
    """At most twenty strategies below PAPER_SHADOW are handed over: the furthest along first, then by an upper
    confidence bound (the expected edge plus a bonus that shrinks as a strategy is tried), so the same few never
    take every slot. Strategies at PAPER_SHADOW and beyond are always handed over."""
    from datetime import timedelta

    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        lab = api.container.options_lab
        ids = {}
        for i in range(22):  # twenty-two validated strategies, edges 1%..22%
            ids[i] = await promote(api, replace(LONG_CALL, delta_target=0.20 + i / 100), Stage.VALIDATION,
                                   key=f"v{i}", latest={"validation_ror": 0.01 * (i + 1)})  # fmt: skip
        walked = await promote(api, replace(LONG_CALL, delta_target=0.6), Stage.WALK_FORWARD, key="walked",
                               latest={"validation_ror": 0.005})  # fmt: skip
        settled = await promote(api, replace(LONG_CALL, delta_target=0.65), Stage.PAPER_SHADOW, key="settled")
        handed = [v["version_id"] for v in await lab.eligible_versions()]
        assert handed[0] == settled  # strategies at PAPER_SHADOW and beyond always
        assert handed[1] == walked  # then the furthest along ...
        assert handed[2:] == [ids[i] for i in range(21, 2, -1)]  # ... then, untried alike, the largest edges
        assert len(handed) == 1 + 20
        # the strongest one has been tried thirty times: untried strategies with smaller edges now come first
        now = clock.now()
        async with api.container.db.session() as s:
            for k in range(30):
                s.add(OptionsPositionRow(version_id=ids[21], underlying="MIDA", family="long_call", direction="bullish",
                                         mode="shadow", structure={}, quantity=1, status="closed",
                                         opened_at=now - timedelta(days=40 - k), entry_value=300.0,
                                         entry_underlying=100.0, max_loss=300.0, closed_at=now - timedelta(days=39 - k),
                                         realized_pnl=30.0))  # fmt: skip
        again = [v["version_id"] for v in await lab.eligible_versions()]
        assert again.index(ids[21]) > again.index(ids[14])  # rotated down ...
        assert ids[21] in again  # ... but its edge still keeps it in


async def test_each_strategy_builds_its_own_shadow_record(tmp_path):
    """Shadow trades are each strategy's own forward test: one strategy's shadow position on an underlying never
    keeps another from trading it in shadow (it used to: every strategy shared one shadow book, so a handful
    of positions starved the rest of the evidence PAPER_ACTIVE needs). A strategy still never stacks."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock, options_exploration=False):
        a = await promote(api, LONG_CALL, Stage.PAPER_SHADOW)
        b = await promote(api, replace(LONG_CALL, delta_target=0.4), Stage.PAPER_SHADOW, key="test-call-40")
        await run_cycle(api)
        shadow = await rows(api, OptionsPositionRow, mode="shadow")
        assert sorted(p.version_id for p in shadow) == [a, b]  # both on MIDA, the only underlying
        assert {p.underlying for p in shadow} == {"MIDA"}
        clock.advance(35 * 60)
        await run_cycle(api)
        assert len(await rows(api, OptionsPositionRow, mode="shadow")) == 2  # neither stacks a second
        gates = {c.gate for c in await rows(api, OptionsTradeCandidateRow) if c.gate}
        assert gates <= {"this strategy already holds this underlying", "OptionsPortfolioAgent"}, gates
        assert not [b for b in api.fake.bodies if b.get("position_intent")]  # shadow is never an order


async def test_the_scan_comes_round_to_hundreds_of_names(tmp_path):
    """Options are looked for across the stock universe, not just the core list: each cycle reads a batch of the
    most liquid names besides the core — up to half on the stock Brain's strongest views, the rest the longest
    unread — so all 300 names on the list are read within a day's cycles, within Alpaca's free data rate."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock, options_scan_size=300, options_scan_per_cycle=16):
        brain = api.container.options_brain
        names = [f"S{i:03d}" for i in range(500)]  # most liquid first
        ctx = SimpleNamespace(liquid=names, focus=["S400"], universe=names[:120])
        consensus = {"S010": {"stance": "bullish", "score": 0.4, "confidence": 0.5},
                     "S250": {"stance": "bearish", "score": -0.9, "confidence": 0.9},
                     "S450": {"stance": "bullish", "score": 0.9, "confidence": 0.9}}  # fmt: skip
        assert brain.scan_list(ctx) == names[:300]  # S400 and S450 are beyond the list
        batch, info = await brain._scan_batch(ctx, consensus, {"MIDA", "S001"})
        assert len(batch) == 16 and info["list"] == 300 and info["never_read"] == 299
        assert info["signals"] == ["S250", "S010"]  # the strongest stock views first
        # then the most liquid, none read yet
        assert info["rotation"] == [n for n in names[:16] if n not in ("S001", "S010")]
        seen: set[str] = set()
        cycles = 0
        while len(seen) < 299 and cycles < 40:  # S001 is held: read every cycle anyway
            batch, info = await brain._scan_batch(ctx, consensus, {"MIDA", "S001"})
            assert "S001" not in batch and len(batch) == 16
            for u in batch:
                brain._read_at[u] = clock.now()
            seen |= set(batch)
            clock.advance(5 * 60)
            cycles += 1
        # a session has 78 five-minute cycles: every name is read about four times a day
        assert len(seen) == 299 and cycles <= 22, cycles
        assert info["signals"] == ["S250", "S010"]  # strong views are read every cycle
        assert brain.scan_list(None) == [] and (await brain._scan_batch(None, {}, set()))[0] == []


async def test_a_scanned_name_is_read_and_traded_like_the_core(tmp_path):
    """A name from the scan is perceived (chain, features, IV record), deliberated and traded in shadow exactly as
    a core underlying is, through the same checks; its daily prices come from the price warehouse."""
    from quantpulse.db.options_models import OptionsChainSnapshotRow

    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock, options_scan_size=40, options_scan_per_cycle=6,
                            options_exploration=False):  # fmt: skip
        api.market.vol = 0.09  # every name fairly priced, as MIDA is
        await promote(api, LONG_CALL, Stage.PAPER_SHADOW)
        await run_cycle(api)
        last = api.container.options_brain.last
        read = last["scan"]["read"]
        assert len(read) == 6 and "MIDA" not in read, last["scan"]
        assert set(last["underlyings"]) == {"MIDA", *read}
        snapped = {r.underlying for r in await rows(api, OptionsChainSnapshotRow)}
        assert {"MIDA", *read} <= snapped
        cands = await rows(api, OptionsTradeCandidateRow)
        assert {c.underlying for c in cands} & set(read), [(c.underlying, c.gate) for c in cands]
        shadow = await rows(api, OptionsPositionRow, mode="shadow")
        assert {p.underlying for p in shadow} & set(read)
        # the next cycle reads the next names (the strongest stock views aside)
        clock.advance(35 * 60)
        await run_cycle(api)
        assert set(api.container.options_brain.last["scan"]["rotation"]).isdisjoint(last["scan"]["rotation"])
        closes = await api.container.options_brain._closes(["MIDA", "NOPE"])
        assert len(closes["MIDA"]) > 60 and "NOPE" not in closes


async def test_a_scanned_company_gets_its_earnings_date_looked_up_once_a_day(tmp_path):
    """Earnings dates are looked up for scanned companies (a fund has none); one that cannot be found is marked
    unknown on the view, never guessed (the earnings agent then vetoes strategies that avoid events)."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock, brain_catalyst_analysis=True):
        brain = api.container.options_brain
        asked: list[str] = []
        found = {"MIDA": None, "UPF": NOW.date() + timedelta(days=20)}

        async def next_earnings(sym, asked=asked, found=found):
            asked.append(sym)
            return found.get(sym), "estimated"

        brain._reference = SimpleNamespace(next_earnings=next_earnings)
        ctx = SimpleNamespace(close=pd.DataFrame(), earnings={}, sectors={"MIDA": "Industrials", "UPF": "Energy",
                                                                           "SPY": "ETF"})  # fmt: skip
        views = await brain._perceive(["MIDA", "UPF", "SPY"], clock.now(), ctx)
        assert views["MIDA"].earnings_unknown and not views["UPF"].earnings_unknown
        # a fund has no earnings: it is not looked up, and not flagged
        assert not views["SPY"].earnings_unknown and sorted(asked) == ["MIDA", "UPF"]
        assert views["UPF"].features is not None and views["UPF"].features.event_days is not None
        await brain._perceive(["MIDA", "UPF"], clock.now(), ctx)
        assert sorted(asked) == ["MIDA", "UPF"]  # once a day


async def test_open_interest_comes_from_the_contract_list_once_a_day(tmp_path):
    """Alpaca's chain snapshot carries no open interest (only the contract list does). Without it every leg failed
    the risk preview's open-interest check, the risk agent vetoed every candidate, and no option ever traded —
    paper or shadow. The chain now gets it from the contract list, read once a day per underlying."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        market, brain = api.market, api.container.options_brain
        assert not market.oi_on_quotes  # like Alpaca
        ctx = SimpleNamespace(close=pd.DataFrame(), earnings={}, sectors={})
        views = await brain._perceive(["MIDA"], clock.now(), ctx)
        assert views["MIDA"].chain is not None
        assert {q.open_interest for q in views["MIDA"].chain.quotes} == {2500.0}
        read = market.calls.count("contracts")
        assert read == 1
        await brain._perceive(["MIDA"], clock.now(), ctx)
        assert market.calls.count("contracts") == read  # the last close's figure: once a day
        clock.advance(24 * 3600)
        await brain._perceive(["MIDA"], clock.now(), ctx)
        assert market.calls.count("contracts") == read + 1


async def test_a_paper_order_the_risk_engine_refuses_still_trades_in_shadow(tmp_path):
    """A paper order the execution checks refuse (here open interest 50, under the 100 minimum) is not sent, but
    the strategy's shadow test still runs: shadow evidence is what a strategy needs to earn PAPER_ACTIVE, and a
    veto on the order used to throw it away. Why there is no paper order is recorded."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        api.market.open_interest = 50.0
        await promote(api, replace(LONG_CALL, min_open_interest=10), Stage.PAPER_ACTIVE)
        await run_cycle(api)
        assert not [b for b in api.fake.bodies if b.get("position_intent")]  # the refused order is never sent
        assert await rows(api, OptionsPositionRow, mode="paper") == []
        shadow = await rows(api, OptionsPositionRow, mode="shadow")
        assert len(shadow) == 1 and shadow[0].status == "open"
        [cand] = await rows(api, OptionsTradeCandidateRow)
        assert cand.mode == "shadow" and cand.gate is None
        assert (
            "open interest 50" in cand.audit["paper_blocked"]
            and "OptionsRiskAgent" in cand.audit["paper_blocked"]
        )


async def test_an_open_option_position_does_not_block_paper_orders_on_other_underlyings(tmp_path):
    """The risk preview measures the whole option book's delta and vega. It was given quotes for the candidate's
    own chain only, so once one option position was open, every paper order on any other underlying failed
    closed ("the book's delta or vega cannot be measured") and the risk agent vetoed it. The account's legs are
    now priced from the chains they were read in (every held underlying is read each cycle)."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        m = api.market
        held = m.pick("UPF", "call", moneyness=1.3)  # far out of the money: little delta, within every limit
        spot = api.feed.live_price("UPF")
        leg = {"symbol": held.symbol, "side": "long", "ratio": 1, "kind": "call", "strike": held.strike,
               "expiration": held.expiration.isoformat()}  # fmt: skip
        async with api.container.db.session() as s:
            s.add(OptionsPositionRow(underlying="UPF", family="long_call", direction="bullish", mode="paper",
                                     structure={"legs": [leg]}, quantity=1, status="open", expiry_state="OPEN",
                                     first_expiration=held.expiration, opened_at=clock.now() - timedelta(days=1),
                                     entry_value=300.0, entry_underlying=spot, max_loss=300.0, marks=[]))  # fmt: skip
        api.fake.hold(held.symbol, 1, 3.0)
        await promote(api, LONG_CALL, Stage.PAPER_ACTIVE)
        await run_cycle(api)
        opened = [b for b in api.fake.bodies if b.get("position_intent") == "buy_to_open"]
        assert any(str(b.get("symbol", "")).startswith("MIDA") for b in opened), [
            (c.underlying, c.gate, c.reject_reason) for c in await rows(api, OptionsTradeCandidateRow)
        ]
        cands = [c for c in await rows(api, OptionsTradeCandidateRow) if c.underlying == "MIDA"]
        assert cands and not any("cannot be measured" in (c.reject_reason or "") for c in cands)


async def test_a_chain_read_late_in_the_cycle_is_judged_when_it_was_read(tmp_path):
    """Chains are read one after another, up to a minute or more into the options pass. Their quality used to be
    judged at the pass's start, so the freshest quotes of a chain read a minute later looked a minute "in the
    future" (clock skew) and the chain was graded unusable for execution. It is judged when it was read."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        market, brain = api.market, api.container.options_brain
        start = clock.now()
        real_chain = market.chain

        async def slow_chain(underlying, real_chain=real_chain, clock=clock, **kw):
            clock.advance(60)  # the chain is read a minute into the pass
            return await real_chain(underlying, **kw)

        market.chain = slow_chain
        ctx = SimpleNamespace(close=pd.DataFrame(), earnings={}, sectors={})
        views = await brain._perceive(["MIDA"], start, ctx)
        q = views["MIDA"].quality
        assert q["OPTIONS_DATA_QUALITY"] == "execution", q["blockers"]
        assert not any("future" in b for b in q["blockers"])


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


async def test_an_early_assignment_freezes_the_rest_and_asks_a_person(tmp_path):
    """American options can be assigned any day. When the short leg of a spread is assigned the long leg is the
    hedge of the delivered shares: QuantPulse must not close it alone (that could leave naked stock), nor keep
    sending exits for legs it no longer holds. It freezes the position, records the likely assignment, keeps
    new trades off the underlying and alerts a person."""
    from datetime import timedelta

    from quantpulse.db.options_models import OptionsAssignmentEventRow

    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        m, spot = api.market, api.feed.live_price("MIDA")
        long = m.pick("MIDA", "call", moneyness=0.85)
        short = min((c for c in m.listed("MIDA") if c.kind == "call" and c.expiration == long.expiration
                     and c.strike > long.strike), key=lambda c: c.strike)  # fmt: skip
        assert short.strike < spot  # both in the money: the short call is a candidate for early assignment
        legs = [{"symbol": c.symbol, "side": side, "ratio": 1, "kind": "call", "strike": c.strike,
                 "expiration": c.expiration.isoformat()} for c, side in ((long, "long"), (short, "short"))]  # fmt: skip
        async with api.container.db.session() as s:
            pos = OptionsPositionRow(underlying="MIDA", family="bull_call_spread", direction="bullish", mode="paper",
                                     structure={"legs": legs}, quantity=1, status="open", expiry_state="OPEN",
                                     first_expiration=long.expiration, opened_at=clock.now() - timedelta(days=3),
                                     entry_value=250.0, entry_underlying=spot, max_loss=250.0, marks=[])  # fmt: skip
            s.add(pos)
            await s.commit()
            pid = pos.id
        api.fake.hold(long.symbol, 1, 5.0)
        api.fake.hold(short.symbol, -1, 3.0)
        event = api.fake.assign(short.symbol, spot)
        assert event["shares"] == -100 and api.fake.positions["MIDA"]["qty"] == -100  # shares delivered
        await promote(api, LONG_CALL, Stage.PAPER_ACTIVE)  # a strategy that would otherwise trade MIDA
        await run_cycle(api)
        (pos,) = [p for p in await rows(api, OptionsPositionRow, mode="paper") if p.id == pid]
        assert pos.status == "assigned" and pos.expiry_state == "ASSIGNED"
        (ev,) = await rows(api, OptionsAssignmentEventRow)
        assert (
            ev.symbol == short.symbol
            and ev.share_delivery == -100
            and ev.detail["inferred"]
            and ev.detail["early"]
        )
        assert not [b for b in api.fake.bodies if long.symbol in str(b)]  # the hedge is never closed alone
        assert len(await rows(api, OptionsPositionRow, mode="paper")) == 1  # no new trade on MIDA meanwhile
        assert {c.gate for c in await rows(api, OptionsTradeCandidateRow)} == {"OptionsPortfolioAgent"}
        await api.container.health._options_alerts()
        assert "option_assigned" in {a["kind"] for a in api.container.alerts.sent}
        # a person closes the shares and the long call in the paper account: the record follows
        del api.fake.positions[long.symbol], api.fake.positions["MIDA"]
        clock.advance(35 * 60)
        await run_cycle(api)
        (pos,) = [p for p in await rows(api, OptionsPositionRow, mode="paper") if p.id == pid]
        assert pos.status == "closed" and "resolved outside QuantPulse" in pos.exit_reason


def _order(cid, symbol, side, qty, status, filled, price):
    from quantpulse.db.models import BrokerOrderRow

    return BrokerOrderRow(client_order_id=cid, symbol=symbol, side=side, quantity=qty, asset_class="us_option",
                          order_type="limit", limit_price=price, status=status, filled_quantity=filled,
                          average_fill_price=price, strategy="brain")  # fmt: skip


async def _sync(api, pid, closing_cid=None, *orders):
    """Record the broker orders, mark the position closing on ``closing_cid``, and let the Options Brain sync it."""
    from quantpulse.brain.options.brain import OptionsCycle

    db, now = api.container.db, api.container.clock.now()
    async with db.session() as s:
        p = await s.get(OptionsPositionRow, pid)
        for o in orders:
            s.add(o)
        if closing_cid:
            p.status, p.exit_client_order_id = "closing", closing_cid
        await s.commit()
    await api.container.options_brain._sync_paper(OptionsCycle(cycle_id=None, at=now), p, None, {}, now)
    async with db.session() as s:
        return await s.get(OptionsPositionRow, pid)


async def test_partial_fills_keep_the_record_equal_to_what_is_held(tmp_path):
    """A DAY order can end part-filled at the close. Opening: the position is what was bought (not "never
    filled" while the contracts sit in the account). Closing: the closed part's P&L is booked and the rest stays
    open at its share of the entry, so the next exit asks for exactly what is held."""
    clock = FakeClock(NOW)
    async for api in client(tmp_path, clock):
        c = api.market.pick("MIDA", "call", moneyness=1.0)
        legs = [{"symbol": c.symbol, "side": "long", "ratio": 1, "kind": "call", "strike": c.strike,
                 "expiration": c.expiration.isoformat()}]  # fmt: skip
        async with api.container.db.session() as s:
            s.add(_order("qp-test-open", c.symbol, "buy", 3, "expired", 2, 1.9))
            pos = OptionsPositionRow(underlying="MIDA", family="long_call", direction="bullish", mode="paper",
                                     structure={"legs": legs}, quantity=3, status="pending", expiry_state="OPEN",
                                     first_expiration=c.expiration, opened_at=clock.now(), entry_value=600.0,
                                     entry_underlying=100.0, max_loss=600.0, marks=[], client_order_id="qp-test-open")  # fmt: skip
            s.add(pos)
            await s.commit()
            pid = pos.id
        p = await _sync(api, pid)
        assert (p.status, p.quantity, p.entry_value, p.max_loss) == (
            "open",
            2,
            380.0,
            400.0,
        )  # 2 of 3 at 1.90
        p = await _sync(api, pid, "qp-c1", _order("qp-c1", c.symbol, "sell", 2, "canceled", 1, 2.5))
        assert (p.status, p.quantity, p.entry_value, p.max_loss) == ("open", 1, 190.0, 200.0)
        assert p.structure["partial_exits"][0]["pnl"] == 60.0  # 250 received for what cost 190
        p = await _sync(api, pid, "qp-c2", _order("qp-c2", c.symbol, "sell", 1, "filled", 1, 3.0))
        assert p.status == "closed" and p.realized_pnl == 170.0  # 110 on the last contract + 60 on the first
