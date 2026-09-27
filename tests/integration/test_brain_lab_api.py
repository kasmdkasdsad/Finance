"""The strategy lab through the API on the fake market: propose, validate, refuse what fails, paper-track,
refuse early promotion, promote after enough paper sessions, and the promoted strategy's voice in a cycle.
No step reaches Alpaca's order endpoint."""

from datetime import date

from quantpulse.core.clock import FakeClock
from tests.fakes.market import TrendFeed

from .conftest import NOW, _client, make_settings
from .test_brain_cycle import BRAIN, WIDE, _no_network, brain_client, extend_feed, only_reads  # noqa: F401

LAB = f"{BRAIN}/lab"


async def test_the_strategy_lifecycle(tmp_path):
    clock = FakeClock(NOW)
    feed = TrendFeed(clock, drifts=WIDE, sessions=1200)
    async for api in brain_client(tmp_path, clock, feed=feed, brain_lab_history_days=1700):
        proposed = (await api.post(f"{LAB}/propose")).json()
        assert {p["key"] for p in proposed} == {
            f"{t}@v1" for t in (await api.get(f"{LAB}/templates")).json()["templates"]
        }
        assert (await api.post(f"{LAB}/propose")).json() == []  # nothing new to propose

        # a proposal cannot be paper-tracked or promoted before validation
        r = await api.post(f"{LAB}/strategies/momentum_12_1/1/status", json={"status": "paper"})
        assert r.status_code == 422 and "validate it first" in r.json()["detail"]

        good = (await api.post(f"{LAB}/strategies/momentum_12_1/1/validate", params={"wait": 300})).json()
        assert good["verdict"] == "validated", [g for g in good["gates"] if not g["passed"]]
        assert good["data"]["survivorship_bias"] and good["walk_forward"]["trials"] == 6
        bad = (
            await api.post(f"{LAB}/strategies/short_term_reversal/1/validate", params={"wait": 300})
        ).json()
        assert bad["verdict"] == "rejected" and any(not g["passed"] for g in bad["gates"])
        r = await api.post(f"{LAB}/strategies/short_term_reversal/1/status", json={"status": "paper"})
        assert r.status_code == 422

        paper = (await api.post(f"{LAB}/strategies/momentum_12_1/1/status", json={"status": "paper"})).json()
        assert paper["status"] == "paper" and paper["decided_by"] == "user"
        first = (await api.post(f"{LAB}/paper")).json()
        assert first == {"tracked": 1, "rebalanced": 1}
        r = await api.post(f"{LAB}/strategies/momentum_12_1/1/status", json={"status": "promoted"})
        assert r.status_code == 422 and "paper sessions" in r.json()["detail"]  # too early

        extend_feed(api.feed, date(2026, 11, 3))
        clock.advance(35 * 86400)
        (await api.post(f"{LAB}/paper")).json()
        detail = (await api.get(f"{LAB}/strategies/momentum_12_1/1")).json()
        assert detail["paper"]["sessions"] >= 20 and detail["paper"]["entries"] >= 2
        assert {r["kind"] for r in detail["runs"]} == {"validation", "paper"}
        promoted = (
            await api.post(f"{LAB}/strategies/momentum_12_1/1/status", json={"status": "promoted"})
        ).json()
        assert promoted["status"] == "promoted" and promoted["promoted_at"]

        # a new version is a separate record; the old one never changes
        v2 = (
            await api.post(
                f"{LAB}/strategies", json={"template": "momentum_12_1", "top_n": 5, "search_grid": False}
            )
        ).json()
        assert v2["key"] == "momentum_12_1@v2" and v2["spec"]["top_n"] == 5 and v2["parent_version"] == 1
        compared = (
            await api.get(f"{LAB}/compare", params={"keys": "momentum_12_1@v1,momentum_12_1@v2"})
        ).json()
        assert [c["verdict"] for c in compared] == ["validated", None]

        # the promoted strategy is now a voice in the brain's consensus
        await api.post(f"{LAB}/paper")  # publishes the promoted rankings
        cycle = (await api.post(f"{BRAIN}/run")).json()
        runs = {r["agent_id"]: r["status"] for r in cycle["runs"]}
        assert runs["strategy_lab"] == "ok"
        votes = [o for o in cycle["opinions"] if o["agent_id"] == "strategy_lab" and o["stance"] != "abstain"]
        assert votes and all("momentum_12_1@v1" in o["meta"]["strategies"] for o in votes)
        assert only_reads(api.fake)


async def test_the_lab_refuses_synthetic_prices(tmp_path, clock):
    async for api in _client(make_settings(tmp_path), clock):  # live data off: only synthetic prices exist
        await api.post(f"{LAB}/propose")
        r = await api.post(f"{LAB}/strategies/momentum_12_1/1/validate", params={"wait": 120})
        assert r.status_code == 422 and "synthetic" in r.json()["detail"]
        strategies = (await api.get(f"{LAB}/strategies", params={"status": "proposed"})).json()
        assert "momentum_12_1@v1" in {s["key"] for s in strategies}  # still untested, not "rejected"
