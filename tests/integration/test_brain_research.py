"""The Brain's 24/7 operating model, end to end on the fake Alpaca paper API.

* the learning ledger judges every conclusion from its evidence (UNPROVEN until the sample suffices);
* the improvement lifecycle moves one tested stage at a time and only a person promotes;
* the research queue is persistent, deduplicated, atomically claimed and recovered after a restart;
* the scheduler runs research only while the market is closed and only in the supervising process, stops it at
  the open, under memory pressure, on a pause and at shutdown (each job queued again), and records every result;
* execution waits for the day's readiness pipeline, which only ever adds a gate in front of the existing ones;
* every research job, run on the fake world with the Brain owning the paper account, sends no order and changes
  no setting, switch, limit or production strategy.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from quantpulse.brain.research import catalog
from quantpulse.brain.research.catalog import CATALOG, JobSpec
from quantpulse.brain.research.ledger import Finding, LearningLedger
from quantpulse.brain.research.lifecycle import STAGES, Lifecycle, LifecycleError
from quantpulse.brain.research.queue import ResearchQueue
from quantpulse.brain.research.resources import Snapshot
from quantpulse.core.clock import FakeClock
from quantpulse.db.research_models import BrainHypothesisRow, BrainLearningRow
from tests.fakes.alpaca_paper import FakeAlpacaPaper

from .conftest import NOW
from .test_brain_cycle import BRAIN, _no_network, brain_client, run_cycle, with_stock_model  # noqa: F401
from .test_brain_execution import ENABLED, OWNS, brain_orders, posts

RESEARCH = f"{BRAIN}/research"
FRIDAY_CLOSE = datetime(2026, 9, 25, 20, 50, tzinfo=UTC)  # Friday 16:50 New York
SATURDAY = datetime(2026, 9, 26, 15, 30, tzinfo=UTC)  # 11:30 New York
PREP = datetime(2026, 9, 28, 13, 10, tzinfo=UTC)  # Monday 09:10 New York: pre-market, light jobs still start
ROOMY = Snapshot(40.0, "cgroup", 0.1, 300.0)
TIGHT = Snapshot(90.0, "cgroup", 0.1, 300.0)


def finding(**kw) -> Finding:
    base = dict(topic="agent:momentum", claim="momentum's calls beat a coin flip", sample_size=60, min_sample=30,
                regime="all", benchmark="a coin flip", method="block binomial test",
                limitations=["one market regime"], p_value=0.01, effect=0.08)  # fmt: skip
    base.update(kw)
    return Finding(**base)  # type: ignore[arg-type]


async def lead(api, snapshot: Snapshot = ROOMY):
    """This process supervises the Brain (holds the lease) and has room to run research."""
    research = api.container.brain.research
    research.governor._read = lambda: snapshot
    assert await api.container.lease.acquire()
    return research


async def drain(research) -> None:
    await research.scheduler.idle()


# ================================================================================================ the ledger
async def test_the_ledger_judges_records_and_supersedes(database):
    ledger = LearningLedger(database)
    early = await ledger.record(
        "agent_calibration", finding(sample_size=10, p_value=0.0001), NOW, job_id=None
    )
    assert early["status"] == "UNPROVEN" and early["confidence"] == 0 and early["supersedes_id"] is None
    later = await ledger.record(
        "agent_calibration", finding(sample_size=80), NOW + timedelta(hours=1), job_id=7
    )
    assert (
        later["status"] == "SUPPORTED"
        and 0 < later["confidence"] < 1
        and later["supersedes_id"] == early["id"]
    )
    assert later["statistics"]["p_value"] == 0.01 and later["job_id"] == 7 and later["limitations"]
    assert [r["id"] for r in await ledger.learnings(current_only=True)] == [later["id"]]
    assert len(await ledger.learnings()) == 2  # what it believed before is kept
    assert (await ledger.summary())["by_status"] == {"SUPPORTED": 1}
    nan = await ledger.record("x", finding(topic="x", p_value=float("nan")), NOW)
    assert nan["status"] == "UNPROVEN" and nan["statistics"]["p_value"] is None
    with pytest.raises(ValueError, match="limitations"):
        await ledger.record("x", finding(limitations=[]), NOW)
    async with database.session() as s:
        assert len((await s.scalars(select(BrainLearningRow))).all()) == 3  # the refused one was not written


# ================================================================================================ the lifecycle
async def test_an_improvement_moves_one_tested_stage_at_a_time_and_only_a_person_promotes(database):
    lc = Lifecycle(database)
    h = await lc.discover(kind="feature", key="feature:x:+", title="use x as a ranking signal", source="research",
                          detail={"feature": "x"}, now=NOW)  # fmt: skip
    seen = [h["stage"]]
    for _ in range(6):
        h = await lc.advance(h["id"], passed=True, evidence={"gate": "passed"}, by="research", now=NOW)
        seen.append(h["stage"])
    assert seen == list(STAGES[:-1])  # DISCOVERED → … → EVALUATION, never a stage skipped
    assert h["awaiting_person"] and h["next_stage"] == "PRODUCTION"
    with pytest.raises(LifecycleError, match="only a person"):
        await lc.advance(h["id"], passed=True, evidence={"x": 1}, by="research", now=NOW)
    for who in ("brain", "research", "system", "supervisor", "lab", "automatic"):
        with pytest.raises(LifecycleError, match="only a person"):
            await lc.promote(h["id"], by=who, note="enough", now=NOW)
    with pytest.raises(LifecycleError, match="note"):
        await lc.promote(h["id"], by="Kim", note=" ", now=NOW)
    assert (await lc.get(h["id"]))["stage"] == "EVALUATION"
    done = await lc.promote(h["id"], by="Kim", note="the forward IC held for 30 sessions", now=NOW)
    assert done["stage"] == "PRODUCTION" and done["decided_by"] == "Kim" and done["promoted_at"]
    assert done["history"][-1] == {"from": "EVALUATION", "to": "PRODUCTION", "at": NOW.isoformat(), "by": "Kim",
                                   "passed": True, "evidence": {"note": "the forward IC held for 30 sessions"}}  # fmt: skip
    assert await lc.unapproved_in_production() == []
    with pytest.raises(LifecycleError):
        await lc.advance(h["id"], passed=True, evidence={"x": 1}, by="Kim", now=NOW)

    s = await lc.discover(
        kind="strategy", key="strategy:y", title="strategy y", source="lab", detail={}, now=NOW
    )
    with pytest.raises(LifecycleError, match="evidence"):
        await lc.advance(s["id"], passed=True, evidence={}, by="research", now=NOW)
    s = await lc.advance(s["id"], passed=True, evidence={"spec": 1}, by="research", now=NOW)
    s = await lc.advance(s["id"], passed=False, evidence={"gate": "better than random portfolios"}, by="research",
                         now=NOW)  # fmt: skip
    assert s["stage"] == "REJECTED"  # a failed stage ends it
    for call in (lc.advance(s["id"], passed=True, evidence={"x": 1}, by="r", now=NOW),
                 lc.promote(s["id"], by="Kim", note="n", now=NOW)):  # fmt: skip
        with pytest.raises(LifecycleError):
            await call
    again = await lc.discover(
        kind="strategy", key="strategy:y", title="strategy y", source="lab", detail={}, now=NOW
    )
    assert again["id"] == s["id"] and again["stage"] == "REJECTED"  # rediscovery does not resurrect it

    p = await lc.discover(kind="process", key="improvement:limits", title="raise QP_TRADING_MAX_POSITION_PCT",
                          source="improvement", detail={}, now=NOW)  # fmt: skip
    assert p["stage"] == "PROTECTED_REVIEW" and p["protected_control"] == "QP_TRADING_MAX_"
    with pytest.raises(LifecycleError, match="does not advance"):
        await lc.advance(p["id"], passed=True, evidence={"x": 1}, by="research", now=NOW)
    with pytest.raises(LifecycleError):
        await lc.promote(p["id"], by="Kim", note="n", now=NOW)
    with pytest.raises(LifecycleError, match="unknown kind"):
        await lc.discover(kind="risk_limit", key="k", title="t", source="research", detail={}, now=NOW)

    async with database.session() as session:  # something put in production without a person: detected
        session.add(BrainHypothesisRow(key="forced", kind="process", title="t", source="x", stage="PRODUCTION",
                                       detail={}, history=[], decided_by="brain", created_at=NOW, updated_at=NOW))  # fmt: skip
    assert await lc.unapproved_in_production() == ["forced"]
    assert (await lc.counts())["PRODUCTION"] == 2


# ================================================================================================ the queue
async def test_the_queue_dedupes_orders_claims_once_and_recovers_after_a_restart(database):
    q = ResearchQueue(database)

    async def ask(kind, cost="medium", prio=5.0, **kw):
        return await q.enqueue(kind=kind, question="?", params={}, cost=cost, priority_=prio, priority_detail={},
                               source="system", now=NOW, **kw)  # fmt: skip

    a = await ask("trade_review", prio=5)
    again = await ask("trade_review", prio=8)
    assert again["deduplicated"] and again["id"] == a["id"] and (await q.get(a["id"]))["priority"] == 8
    heavy = await ask("feature_research", cost="heavy", prio=9)
    later = await ask("watchlist_prep", cost="light", prio=1, not_before=NOW + timedelta(hours=1))
    assert [j["kind"] for j in await q.next_jobs(NOW)] == ["feature_research", "trade_review"]
    assert [j["kind"] for j in await q.next_jobs(NOW, costs=("light", "medium"))] == ["trade_review"]
    assert [j["id"] for j in await q.next_jobs(NOW + timedelta(hours=2), costs=("light",))] == [later["id"]]

    assert await q.claim(a["id"], "host-a:1", NOW)
    assert not await q.claim(a["id"], "host-b:2", NOW)  # two processes never run the same job
    assert await q.is_open(a["key"])
    # host-a stopped (a reboot): host-b finds its job and queues it again
    assert await q.recover(NOW + timedelta(minutes=1), "host-b:2", timedelta(minutes=10)) == [a["id"]]
    job = await q.get(a["id"])
    assert job["status"] == "queued" and job["attempts"] == 1 and "interrupted" in job["error"]
    assert await q.claim(a["id"], "host-b:2", NOW + timedelta(minutes=1))
    assert (
        await q.recover(NOW + timedelta(minutes=2), "host-b:2", timedelta(minutes=10)) == []
    )  # its own, alive
    assert await q.recover(NOW + timedelta(minutes=20), "host-b:2", timedelta(minutes=10)) == [
        a["id"]
    ]  # silent
    assert await q.claim(a["id"], "host-b:2", NOW + timedelta(minutes=21))
    keep = frozenset({a["id"]})
    assert await q.recover(NOW + timedelta(minutes=40), "x", timedelta(minutes=10), keep=keep) == []
    assert (
        await q.requeue(a["id"], reason="boom", now=NOW + timedelta(minutes=41), delay=timedelta(0))
        == "failed"
    )
    job = await q.get(a["id"])
    assert job["status"] == "failed" and job["finished_at"] and "attempt 3" in job["error"]
    assert not await q.is_open(a["key"]) and (await q.last_finished("trade_review"))["id"] == a["id"]
    assert await q.cancel(heavy["id"], NOW, "by a person") and not await q.cancel(heavy["id"], NOW, "again")
    assert await q.counts() == {"failed": 1, "cancelled": 1, "queued": 1}


# ================================================================================================ the scheduler
async def test_no_research_runs_while_the_market_is_open(tmp_path):
    clock = FakeClock(NOW)  # Friday 10:00 New York
    async for api in brain_client(tmp_path, clock):
        research = await lead(api)
        r = await api.post(f"{RESEARCH}/questions", json={"kind": "trade_review", "question": "why?"})
        assert r.status_code == 200 and r.json()["status"] == "queued" and r.json()["source"] == "person"
        out = await research.tick()
        assert out == "EXECUTION: no research (execution and safety first)"
        assert (
            research.scheduler.running() == [] and (await research.job(r.json()["id"]))["status"] == "queued"
        )
        op = (await api.get(f"{RESEARCH}/operating")).json()
        assert op["mode"] == "EXECUTION" and op["loop"] == ["EXECUTE", "MONITOR", "RECONCILE", "LEARN"]
        assert op["research_allowed"] == []


async def test_closed_market_research_seeds_runs_and_records_every_job(tmp_path):
    clock = FakeClock(SATURDAY)
    async for api in brain_client(tmp_path, clock):
        research = api.container.brain.research
        assert (await research.tick()).startswith("standby")  # not the supervising process: no research
        await lead(api)
        out = await research.tick()
        assert "started grade_predictions#" in out  # the most valuable question first
        queued = {j["kind"] for j in await research.jobs(status="queued", limit=500)}
        standing = {k for k, s in CATALOG.items() if s.refresh is not None and not s.owner_only}
        assert queued | {"grade_predictions"} == standing  # every standing question; nothing owner-only
        await drain(research)
        done = (await research.jobs(status="done"))[0]
        assert done["kind"] == "grade_predictions" and done["duration_ms"] is not None
        assert set(done["result"]) == {"result", "learned", "follow_ups"} and done["holder"]
        assert done["priority_detail"]["staleness"] == 3.0  # never answered before
        # the next tick does not seed again (every 10 minutes at most) and starts the next job
        out = await research.tick()
        assert "started" in out and "grade_predictions" not in out
        await drain(research)
        clock.advance(11 * 60)
        await lead(api)  # the lease is renewed by the supervisor in production
        assert "started" in await research.tick()
        await drain(research)
        assert not await research.queue.is_open("grade_predictions:-")  # answered recently: not asked again
        status = (await api.get(f"{RESEARCH}/status")).json()
        assert status["operating"]["mode"] == "RESEARCH" and status["queue"]["done"] >= 3
        assert status["resources"]["limits"]["max_concurrent"] == 1 and status["rules"]


async def test_research_stops_for_the_open_memory_a_pause_and_standby_and_resumes(tmp_path, monkeypatch):
    started = asyncio.Event()
    runs: list[str] = []

    async def probe(ctx):
        runs.append(ctx.job["question"])
        started.set()
        await asyncio.sleep(3600)
        return {}

    monkeypatch.setitem(
        CATALOG, "probe", JobSpec("probe", "a slow light job", "PREPARE", "light", 10, None, probe)
    )
    clock = FakeClock(PREP)  # Monday 09:10 New York: light preparation may start
    async for api in brain_client(tmp_path, clock):
        research = await lead(api)
        sched = research.scheduler

        async def start(research=research, sched=sched) -> int:
            started.clear()
            out = await research.tick()
            assert "started probe#" in out, out
            await asyncio.wait_for(started.wait(), 5)
            return sched.running()[0]

        job_id = (await research.ask("probe", None, {}))["id"]
        assert await start() == job_id
        clock.advance(6 * 60)  # 09:16: research must have stopped before the bell
        await lead(api)
        out = await research.tick()
        assert out == "PRE_MARKET: no research (execution and safety first); 1 stopped"
        job = await research.job(job_id)
        assert job["status"] == "queued" and job["error"] == "stopped: PRE_MARKET: execution has priority"
        assert sched.running() == []

        clock.advance(8 * 3600)  # after the close
        await lead(api)
        assert await start() == job_id
        research.governor._read = lambda: TIGHT  # memory nearly gone: execution comes first
        out = await research.tick()
        assert "research stopped to protect execution" in out and sched.running() == []
        assert "protect execution" in (await research.job(job_id))["error"]

        research.governor._read = lambda: ROOMY
        clock.advance(6 * 60)
        await lead(api)
        assert await start() == job_id
        await api.post(f"{BRAIN}/supervisor", json={"paused": True})
        assert await research.tick() == "paused with the supervisor" and sched.running() == []
        await api.post(f"{BRAIN}/supervisor", json={"paused": False})

        clock.advance(6 * 60)
        await lead(api)
        assert await start() == job_id
        await api.container.lease.release()  # another process took over supervision
        out = await research.tick()
        assert out.startswith("standby") and sched.running() == []
        job = await research.job(job_id)
        assert (
            job["status"] == "queued" and job["error"] == "stopped: this process does not supervise the Brain"
        )
        # four runs, one charged attempt: only the memory stop counts against the job (the open, a pause or a
        # standby are not its doing), so being interrupted never fails it
        assert job["attempts"] == 1
    assert len(runs) == 4


async def test_a_job_that_fails_times_out_or_asks_more_is_recorded(tmp_path, monkeypatch):
    async def broken(ctx):
        raise RuntimeError("bad data")

    async def slow(ctx):
        await asyncio.sleep(60)
        return {}

    async def curious(ctx):
        ctx.ask("trade_review", "what did the losers have in common?", {"since": "2026-09-01"})
        await ctx.learn(finding(topic="probe:x", sample_size=3))
        return {"answer": 42}

    monkeypatch.setitem(CATALOG, "broken", JobSpec("broken", "?", "ANALYZE", "light", 9, None, broken))
    monkeypatch.setitem(CATALOG, "slow", JobSpec("slow", "?", "ANALYZE", "light", 9, None, slow,
                                                 timeout=timedelta(seconds=0.2)))  # fmt: skip
    monkeypatch.setitem(CATALOG, "curious", JobSpec("curious", "?", "ANALYZE", "light", 9, None, curious))
    clock = FakeClock(SATURDAY)
    async for api in brain_client(tmp_path, clock):
        research = await lead(api)
        sched = research.scheduler
        ids = {k: (await research.ask(k, None, {}))["id"] for k in ("broken", "slow", "curious")}
        for kind, job_id in ids.items():
            assert (await sched.run_now(job_id))["kind"] == kind
        broken_job = await research.job(ids["broken"])
        assert broken_job["status"] == "queued" and broken_job["error"] == "RuntimeError: bad data"
        assert datetime.fromisoformat(broken_job["not_before"]) == SATURDAY + timedelta(minutes=30)
        slow_job = await research.job(ids["slow"])
        assert slow_job["status"] == "queued" and slow_job["error"].startswith("timed out")
        assert datetime.fromisoformat(slow_job["not_before"]) == SATURDAY + timedelta(hours=2)
        curious_job = await research.job(ids["curious"])
        assert curious_job["status"] == "done" and curious_job["result"]["result"] == {"answer": 42}
        assert curious_job["result"]["learned"][0]["status"] == "UNPROVEN"  # three observations prove nothing
        follow = (await research.jobs(kind="trade_review"))[0]
        assert follow["source"] == "follow_up" and follow["parent_id"] == ids["curious"]
        assert follow["params"] == {"since": "2026-09-01"} and follow["question"].startswith("what did")
        for _ in range(2):  # attempts run out: failed, with the reason kept
            clock.advance(31 * 60)
            await sched.run_now(ids["broken"])
        assert (await research.job(ids["broken"]))["status"] == "failed"


async def test_restart_recovery_and_shutdown(tmp_path, monkeypatch):
    started = asyncio.Event()

    async def probe(ctx):
        started.set()
        await asyncio.sleep(3600)
        return {}

    monkeypatch.setitem(CATALOG, "probe", JobSpec("probe", "?", "PREPARE", "light", 10, None, probe))
    clock = FakeClock(SATURDAY)
    async for api in brain_client(tmp_path, clock):
        research = await lead(api)
        orphan = (await research.ask("probe", None, {}))["id"]
        # a process that stopped mid-job (a reboot) left it running under its own name
        assert await research.queue.claim(orphan, "old-host:41:abc", clock.now())
        out = await research.tick()
        assert "recovered 1 interrupted job(s)" in out and f"probe#{orphan}" in out
        await asyncio.wait_for(started.wait(), 5)
        job = await research.job(orphan)
        assert (
            job["status"] == "running" and job["holder"] == research.scheduler.holder and job["attempts"] == 2
        )
        # the process stops: research is stopped and its job queued again for the next start
        await api.container.shutdown()
        job = await research.job(orphan)
        assert job["status"] == "queued" and job["error"] == "stopped: the process is stopping"
        assert research.scheduler.running() == []


# ================================================================================================ readiness
async def test_execution_waits_for_the_days_readiness_and_resumes_when_it_passes(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        brain = api.container.brain
        assert (await api.get(f"{RESEARCH}/operating")).json()["readiness"] is None
        real = brain.sessions.premarket

        async def no_data(real=real):
            report = await real()
            return {**report, "ok": False, "failed": [*report.get("failed", []), "market_data"]}

        monkeypatch.setattr(brain.sessions, "premarket", no_data)
        cycle = await run_cycle(api)
        held = [d for d in cycle["decisions"] if d["quantity"] and d["action"] in ("buy", "increase")]
        assert held and cycle["summary"]["orders_sent"] == 0 and brain_orders(api.fake) == []
        for d in held:
            assert (
                d["execution"]["reason"]
                == "execution readiness has not passed today: DATA HEALTH; EXECUTION READINESS"
            )
        ready = (await api.get(f"{RESEARCH}/operating")).json()["readiness"]
        assert ready["passed"] is False and ready["failed"] == ["DATA HEALTH", "EXECUTION READINESS"]
        audit = (await api.get(f"{BRAIN}/execution-audit")).json()["latest"]
        assert audit["ok"]  # the existing audit passed: readiness is an extra gate, not a replacement

        monkeypatch.setattr(brain.sessions, "premarket", real)  # data is back
        cycle = await run_cycle(api)  # within 5 minutes: the held result stands (no hammering)
        assert cycle["summary"]["orders_sent"] == 0
        clock.advance(6 * 60)
        cycle = await run_cycle(api)
        assert cycle["summary"]["orders_sent"] > 0 and brain_orders(api.fake)
        ready = (await api.get(f"{RESEARCH}/operating")).json()["readiness"]
        assert ready["passed"] and ready["owns_account"]
        assert [s["step"] for s in ready["steps"]] == ["PRE-MARKET AUDIT", "DATA HEALTH", "PORTFOLIO RECONCILIATION",
                                                         "WATCHLIST", "STRATEGY STATUS", "EXECUTION READINESS"]  # fmt: skip


async def test_something_in_production_without_a_person_holds_execution(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        async with api.container.db.session() as s:
            s.add(BrainHypothesisRow(key="strategy:sneaky", kind="strategy", title="t", source="lab",
                                     stage="PRODUCTION", detail={}, history=[], decided_by="research",
                                     created_at=NOW, updated_at=NOW))  # fmt: skip
        cycle = await run_cycle(api)
        assert cycle["summary"]["orders_sent"] == 0 and posts(api.fake) == []
        ready = (await api.get(f"{RESEARCH}/operating")).json()["readiness"]
        assert "STRATEGY STATUS" in ready["failed"]
        step = next(s for s in ready["steps"] if s["step"] == "STRATEGY STATUS")
        assert "strategy:sneaky" in step["detail"]


async def test_the_supervisor_runs_readiness_before_the_open_and_records_the_modes(tmp_path):
    clock = FakeClock(datetime(2026, 9, 28, 12, 50, tzinfo=UTC))  # Monday 08:50 New York
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        sup = api.container.brain.supervisor
        done = set((await sup.tick()).split(", "))
        assert "premarket_check" in done
        op = (await api.get(f"{RESEARCH}/operating")).json()
        assert op["mode"] == "PRE_MARKET" and op["readiness"]["passed"], op["readiness"]
        assert op["transitions"][0]["mode"] == "PRE_MARKET"
        clock.advance(45 * 60)  # 09:35: the session
        fake.market_open = True
        await sup.tick()
        op = (await api.get(f"{RESEARCH}/operating")).json()
        assert op["mode"] == "EXECUTION" and [t["mode"] for t in op["transitions"][:2]] == [
            "EXECUTION",
            "PRE_MARKET",
        ]
        assert posts(api.fake) == [] or brain_orders(
            api.fake
        )  # anything sent was the Brain's, through the gates


# ================================================================================================ safety
async def test_every_research_job_leaves_trading_untouched(tmp_path, monkeypatch):
    """The Brain owns the paper account with trading enabled (the riskiest configuration): a session of trading,
    then every research job in the catalogue. Research sends no order, changes no setting, switch or limit,
    and puts nothing into production."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        c = api.container
        await run_cycle(api)
        assert brain_orders(api.fake)  # it traded, so there is something to research
        clock.advance((FRIDAY_CLOSE - NOW).total_seconds())
        research = await lead(api)
        sent_before = list(posts(api.fake))
        settings_before = c.settings.model_dump()
        trading_before = (await api.get("/api/v1/trading/status")).json()
        ks_before = (await api.get(f"{BRAIN}/kill-switch")).json()
        for kind in CATALOG:
            job = await research.ask(kind, None, {})
            out = await research.scheduler.run_now(job["id"])
            assert out["status"] == "done", (kind, out["error"])
        assert posts(api.fake) == sent_before  # not a single order
        assert c.settings.model_dump() == settings_before
        trading_after = (await api.get("/api/v1/trading/status")).json()
        for key in ("mode", "owner", "kill_switch", "paper", "endpoint"):
            assert trading_after.get(key) == trading_before.get(key), key
        assert (await api.get(f"{BRAIN}/kill-switch")).json() == ks_before
        assert await c.brain.lab.strategies("promoted") == []
        assert await research.lifecycle.hypotheses(stage="PRODUCTION") == []
        assert await research.lifecycle.unapproved_in_production() == []
        learned = await research.learnings(current_only=False, limit=1000)
        hypotheses = await research.lifecycle.hypotheses()
        assert learned and hypotheses  # it did research something
        for row in learned:
            assert row["limitations"] and row["benchmark"] and row["method"]
            if row["sample_size"] < row["min_sample"]:
                assert row["status"] == "UNPROVEN" and row["confidence"] == 0, row
        assert (await api.get(f"{RESEARCH}/learnings")).status_code == 200


def test_research_code_has_no_order_path():
    """A static check: the research package never calls an order or settings path, and the only lab status it
    sets is paper (shadow) tracking."""
    import inspect
    import re

    from quantpulse.brain.research import handlers, lifecycle, operating, queue, scheduler, service

    for module in (handlers, scheduler, queue, lifecycle, operating):
        src = inspect.getsource(module)
        for banned in ("submit_order", "run_brain", "place_order", ".arm(", "set_kill_switch", "close_all",
                       "flatten", "setattr(", "os.environ"):  # fmt: skip
            assert banned not in src, (module.__name__, banned)
    from quantpulse.brain.lab.service import STATUSES

    def lab_statuses(module) -> list[str]:
        calls = re.findall(r"set_status\(((?:[^()]|\([^()]*\))*)\)", inspect.getsource(module))
        return [w for call in calls for w in re.findall(r'"(\w+)"', call) if w in STATUSES]

    assert lab_statuses(handlers) == ["paper"]  # shadow tracking only
    # the one call that sets "promoted" is a person's decision through the API, behind the lab's own gates
    assert lab_statuses(service) == ["promoted"]
    assert catalog.CATALOG is CATALOG


# ================================================================================================ the API
async def test_the_research_api(tmp_path, monkeypatch):
    clock = FakeClock(SATURDAY)
    async for api in brain_client(tmp_path, clock):
        cat = (await api.get(f"{RESEARCH}/catalog")).json()
        assert {c["kind"] for c in cat} == set(CATALOG) and [c["phase"] for c in cat] == sorted(
            (c["phase"] for c in cat), key=catalog.PHASES.index
        )
        r = await api.post(f"{RESEARCH}/questions", json={"kind": "no_such_job"})
        assert r.status_code == 422 and "unknown research job" in r.json()["detail"]
        job = (await api.post(f"{RESEARCH}/questions", json={"kind": "trade_review"})).json()
        assert (await api.get(f"{RESEARCH}/jobs/{job['id']}")).json()["kind"] == "trade_review"
        assert (await api.get(f"{RESEARCH}/jobs", params={"status": "queued"})).json()[0]["id"] == job["id"]
        assert (await api.post(f"{RESEARCH}/jobs/{job['id']}/cancel")).json()["status"] == "cancelled"
        assert (await api.post(f"{RESEARCH}/jobs/{job['id']}/cancel")).status_code == 422
        assert (await api.get(f"{RESEARCH}/jobs/999999")).status_code in (404, 422)

        lc = api.container.brain.research.lifecycle
        h = await lc.discover(
            kind="feature", key="feature:z:+", title="z", source="research", detail={}, now=clock.now()
        )
        r = await api.post(
            f"{RESEARCH}/hypotheses/{h['id']}/promote", json={"by": "Kim", "note": "looks good"}
        )
        assert r.status_code == 422 and "only an evaluated" in r.json()["detail"]
        for _ in range(6):
            h = await lc.advance(h["id"], passed=True, evidence={"ok": True}, by="research", now=clock.now())
        r = await api.post(
            f"{RESEARCH}/hypotheses/{h['id']}/promote", json={"by": "brain", "note": "I learned it"}
        )
        assert r.status_code == 422 and "only a person" in r.json()["detail"]
        hyps = (await api.get(f"{RESEARCH}/hypotheses")).json()
        assert hyps[0]["awaiting_person"] and hyps[0]["stage"] == "EVALUATION"
        r = await api.post(
            f"{RESEARCH}/hypotheses/{h['id']}/promote", json={"by": "Kim", "note": "forward IC held"}
        )
        assert r.status_code == 200 and r.json()["stage"] == "PRODUCTION" and r.json()["decided_by"] == "Kim"
        h2 = await lc.discover(
            kind="feature", key="feature:w:+", title="w", source="research", detail={}, now=clock.now()
        )
        r = await api.post(
            f"{RESEARCH}/hypotheses/{h2['id']}/reject", json={"by": "Kim", "note": "not plausible"}
        )
        assert r.json()["stage"] == "REJECTED"
        assert (await api.get(f"{RESEARCH}/learnings", params={"status": "SUPPORTED"})).json() == []
        status = (await api.get(f"{RESEARCH}/status")).json()
        assert status["lifecycle"] == {"PRODUCTION": 1, "REJECTED": 1} and status["enabled"]
    async for api in brain_client(tmp_path / "remote", clock, client_host="203.0.113.9"):
        h = await api.container.brain.research.lifecycle.discover(
            kind="feature", key="feature:q:+", title="q", source="research", detail={}, now=clock.now()
        )
        for path, body in ((f"{RESEARCH}/questions", {"kind": "trade_review"}),
                           (f"{RESEARCH}/hypotheses/{h['id']}/promote", {"by": "Kim", "note": "yes please"}),
                           (f"{RESEARCH}/hypotheses/{h['id']}/reject", {"by": "Kim", "note": "no thanks"})):  # fmt: skip
            assert (await api.post(path, json=body)).status_code == 403, path
        assert (await api.get(f"{RESEARCH}/status")).status_code == 200  # reading is open


async def test_promoting_a_strategy_goes_through_the_labs_own_gates(tmp_path):
    clock = FakeClock(SATURDAY)
    async for api in brain_client(tmp_path, clock):
        research = api.container.brain.research
        lc = research.lifecycle
        h = await lc.discover(kind="strategy", key="strategy:nope@v1", title="nope", source="lab", detail={},
                              now=clock.now(), source_ref="nope@v1")  # fmt: skip
        for _ in range(6):
            h = await lc.advance(h["id"], passed=True, evidence={"ok": True}, by="research", now=clock.now())
        r = await api.post(
            f"{RESEARCH}/hypotheses/{h['id']}/promote", json={"by": "Kim", "note": "promote it"}
        )
        assert r.status_code in (404, 422)  # the lab has no such strategy: nothing is promoted
        assert (await lc.get(h["id"]))["stage"] == "EVALUATION"
        assert await api.container.brain.lab.strategies("promoted") == []


async def test_a_strategy_climbs_to_evaluation_on_evidence_and_only_a_person_puts_it_in_production(tmp_path):
    from quantpulse.brain.research import handlers
    from quantpulse.db.models import BrainStrategyRow

    clock = FakeClock(SATURDAY)
    async for api in brain_client(tmp_path, clock):
        brain = api.container.brain
        research = brain.research
        await brain.lab.propose()
        good, bad = (await brain.lab.strategies("proposed"))[:2]
        every = [*handlers.BACKTEST_GATES, *handlers.WALK_FORWARD_GATES, *handlers.STRESS_GATES]

        async def set_row(st, db=api.container.db, **values):
            async with db.session() as s:
                row = await s.scalar(select(BrainStrategyRow).where(BrainStrategyRow.strategy_id == st["strategy_id"],
                                                                     BrainStrategyRow.version == st["version"]))  # fmt: skip
                for k, v in values.items():
                    setattr(row, k, v)

        gates = [{"gate": g, "passed": True} for g in every]
        await set_row(
            good, status="validated", validation={"verdict": "validated", "at": "x", "gates": gates}
        )
        failing = [{**g, "passed": g["gate"] != "edge survives realistic costs"} for g in gates]
        await set_row(
            bad, status="validated", validation={"verdict": "validated", "at": "x", "gates": failing}
        )
        ctx = handlers.JobContext(job={"id": None, "kind": "strategy_research"}, brain=brain, settings=api.container.settings,
                                  clock=clock, ledger=research.ledger, lifecycle=research.lifecycle)  # fmt: skip

        await handlers.sync_strategy_hypotheses(ctx)
        h = await research.lifecycle.by_key(f"strategy:{good['key']}")
        assert [e["to"] for e in h["history"]] == list(STAGES[:6])  # one evidenced stage at a time
        assert all(e["evidence"] for e in h["history"])
        lab = {s["key"]: s for s in await brain.lab.strategies()}
        assert (
            lab[good["key"]]["status"] == "paper" and lab[good["key"]]["decided_by"] == "research"
        )  # shadow only
        rejected = await research.lifecycle.by_key(f"strategy:{bad['key']}")
        last = rejected["history"][-1]
        assert rejected["stage"] == "REJECTED" and last["from"] == "HYPOTHESIS" and last["passed"] is False
        assert (
            last["evidence"]["gates"]["edge survives realistic costs"]["passed"] is False
        )  # the backtest failed
        assert lab[bad["key"]]["status"] == "validated"  # a failed gate never reaches paper tracking

        await handlers.sync_strategy_hypotheses(ctx)  # not enough paper sessions yet: it waits
        assert (await research.lifecycle.by_key(f"strategy:{good['key']}"))["stage"] == "PAPER_SHADOW"
        days = api.container.settings.brain_lab_paper_days
        await set_row(good, paper={"sessions": days, "excess_return": 0.01, "started": "x"})
        await handlers.sync_strategy_hypotheses(ctx)
        h = await research.lifecycle.by_key(f"strategy:{good['key']}")
        assert h["stage"] == "EVALUATION" and h["awaiting_person"]
        assert await brain.lab.strategies("promoted") == []  # research stops here, always

        r = await api.post(
            f"{RESEARCH}/hypotheses/{h['id']}/promote", json={"by": "research", "note": "it passed"}
        )
        assert r.status_code == 422 and await brain.lab.strategies("promoted") == []
        r = await api.post(f"{RESEARCH}/hypotheses/{h['id']}/promote",
                           json={"by": "Kim", "note": "validated, and 20 paper sessions ahead of the benchmark"})  # fmt: skip
        assert r.status_code == 200 and r.json()["stage"] == "PRODUCTION"
        promoted = await brain.lab.strategies("promoted")
        assert [p["key"] for p in promoted] == [good["key"]] and promoted[0]["decided_by"] == "Kim"


def test_the_poller_runs_the_research_queue():
    import inspect

    from quantpulse.workers import poller

    src = inspect.getsource(poller.Poller.start)
    assert '"research", lambda: BRAIN_CHECK_SECONDS, self.run_research_queue' in src
