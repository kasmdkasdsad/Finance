"""The language-model layer without a real model: calculations are refused, nothing runs without a provider,
model and budget, and budget, cache, output cap, concurrency, timeouts, structured answers, refusals and
failures behave as documented. The provider here is a scripted fake — the tests never reach a model."""

import asyncio
import json
from datetime import UTC, datetime

import numpy as np
import pytest

from quantpulse.brain.agents.briefing import BriefingAgent, facts
from quantpulse.brain.llm import (
    DeterministicTaskError,
    MissingProvider,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelTask,
    NoModelProvider,
    build_provider,
    register_provider,
    schema_errors,
    unregister_provider,
)
from quantpulse.brain.types import Evidence, ModelTier, Opinion, Stance
from quantpulse.config import Settings
from quantpulse.core.clock import FakeClock

from .test_brain_agents import make_ctx, path

NOW = datetime(2026, 9, 24, 15, 0, tzinfo=UTC)


class Scripted:
    """A fake provider: answers from a script (text, a callable, or an exception), records every call."""

    name = "scripted"

    def __init__(self, *answers, usage=(100, 50), stop="end_turn", delay=0.0, missing=None):
        self.answers = list(answers)
        self.calls: list[tuple[ModelRequest, str]] = []
        self.usage, self.stop, self.delay, self.missing = usage, stop, delay, missing
        self.in_flight = self.max_in_flight = 0

    def unavailable(self):
        return self.missing

    async def complete(self, request, model):
        self.calls.append((request, model))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            if isinstance(answer, BaseException):
                raise answer
            text = answer(request) if callable(answer) else answer
            return ModelResponse(text, model, self.usage[0], self.usage[1], self.stop, cost_usd=0.001)
        finally:
            self.in_flight -= 1


def settings(**kw) -> Settings:
    base = {
        "brain_llm_fast_model": "fast-model",
        "brain_llm_strong_model": "strong-model",
        "brain_llm_daily_token_budget": 10_000,
    }
    return Settings(_env_file=None, **{**base, **kw})


def router(provider=None, clock=None, store=None, **kw) -> ModelRouter:
    return ModelRouter(settings(**kw), clock or FakeClock(NOW), provider or Scripted("ok"), store)


def ask(prompt="Summarise.", task=ModelTask.SUMMARIZE, **kw) -> ModelRequest:
    return ModelRequest.ask(task, "You summarise.", prompt, **kw)


SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "points": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "points"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------------------------- the boundary
def test_calculations_are_never_asked_of_a_model():
    for calc in ("rsi", "ATR", "beta", "correlation", "position_weight", "spread", "quote_age", "risk_limit"):
        with pytest.raises(DeterministicTaskError, match="calculated deterministically"):
            ModelRequest.ask(calc, "s", "p")
    with pytest.raises(ValueError, match="unknown model task"):
        ModelRequest.ask("predict_price", "s", "p")
    with pytest.raises(ValueError, match="end with a user message"):
        ModelRequest(ModelTask.SUMMARIZE, "s", (("user", "a"), ("assistant", "b")))
    assert ask().tier is ModelTier.FAST and ask(task="debate").tier is ModelTier.STRONG


async def test_nothing_runs_by_default():
    s = Settings(_env_file=None)
    assert s.brain_llm_provider == "none" and s.brain_llm_daily_token_budget == 0
    r = ModelRouter(s, FakeClock(NOW))
    assert isinstance(build_provider(s), NoModelProvider)
    out = await r.complete(ask())
    assert out.status == "skipped" and "QP_BRAIN_LLM_PROVIDER=none" in (out.reason or "")
    assert r.status()["available"] is False and r.status()["usage"]["skipped"] == 1

    fake = Scripted("ok")
    no_model = ModelRouter(settings(brain_llm_fast_model=""), FakeClock(NOW), fake)
    assert "no fast model" in (await no_model.complete(ask())).reason
    no_budget = ModelRouter(settings(brain_llm_daily_token_budget=0), FakeClock(NOW), fake)
    assert "daily token budget is 0" in (await no_budget.complete(ask())).reason
    missing_key = ModelRouter(settings(), FakeClock(NOW), Scripted("ok", missing="API key is not set"))
    assert (await missing_key.complete(ask())).reason == "API key is not set"
    assert fake.calls == []  # none of these reached the provider


def test_providers_are_plugged_in_by_name():
    assert "not installed" in (MissingProvider("acme").unavailable() or "")
    assert isinstance(build_provider(settings(brain_llm_provider="acme")), MissingProvider)
    fake = Scripted("ok")
    register_provider("scripted", lambda _s: fake)
    try:
        assert build_provider(settings(brain_llm_provider="Scripted")) is fake
    finally:
        unregister_provider("scripted")
    with pytest.raises(ValueError, match="reserved"):
        register_provider("none", lambda _s: fake)


# ---------------------------------------------------------------------------------------------- cost control
async def test_the_daily_budget_is_checked_before_and_charged_after():
    clock = FakeClock(NOW)
    fake = Scripted("ok", usage=(700, 200))
    r = router(fake, clock, brain_llm_daily_token_budget=2_000, brain_llm_max_output_tokens=300)
    first = await r.complete(ask("one"))
    assert first.status == "ok" and r.status()["usage"]["tokens"] == 900  # what the provider reported
    second = await r.complete(ask("two"))
    assert second.status == "ok" and r.remaining() == 200
    third = await r.complete(ask("three"))  # the estimate (≥ the 300-token output cap) no longer fits
    assert third.status == "skipped" and "would exceed today's token budget" in third.reason
    assert len(fake.calls) == 2
    clock.advance(24 * 3600)  # a new day in New York
    assert (await r.complete(ask("three"))).status == "ok"


async def test_identical_requests_are_answered_from_the_cache_until_it_expires():
    clock = FakeClock(NOW)
    fake = Scripted("ok")
    r = router(fake, clock, brain_llm_cache_minutes=30)
    assert (await r.complete(ask())).status == "ok"
    again = await r.complete(ask())
    assert again.status == "cached" and again.response.text == "ok" and len(fake.calls) == 1
    assert r.status()["usage"]["tokens"] == 150 and r.status()["usage"]["cached"] == 1  # free
    assert (await r.complete(ask("different"))).status == "ok"
    clock.advance(31 * 60)
    assert (await r.complete(ask())).status == "ok" and len(fake.calls) == 3


async def test_output_is_capped_and_calls_are_limited_in_parallel():
    fake = Scripted("ok", delay=0.02)
    r = router(fake, brain_llm_max_output_tokens=200, brain_llm_max_concurrency=1)
    outs = await asyncio.gather(*(r.complete(ask(f"q{i}", max_tokens=4000)) for i in range(3)))
    assert all(o.status == "ok" for o in outs)
    assert {req.max_tokens for req, _ in fake.calls} == {200} and fake.max_in_flight == 1


async def test_usage_survives_a_restart():
    class Store:
        def __init__(self):
            self.state = {}

        async def get_state(self, key):
            return self.state.get(key)

        async def set_state(self, key, value, now):
            self.state[key] = json.loads(json.dumps(value))

    store, clock = Store(), FakeClock(NOW)
    await router(Scripted("ok", usage=(600, 300)), clock, store).complete(ask())
    restarted = router(Scripted("ok"), clock, store)
    await restarted.complete(ask("another"))
    assert restarted.status()["usage"]["tokens"] == 900 + 150 and restarted.status()["usage"]["calls"] == 2


# ---------------------------------------------------------------------------------------------- answers
async def test_structured_answers_must_parse_and_match_the_schema():
    good = json.dumps({"summary": "fine", "points": ["a"]})
    r = router(Scripted(good, "```json\n" + good + "\n```", "not json", json.dumps({"summary": 1})))
    first = await r.complete(ask("a", schema=SCHEMA))
    assert first.status == "ok" and first.parsed == {"summary": "fine", "points": ["a"]}
    assert (await r.complete(ask("b", schema=SCHEMA))).parsed["points"] == ["a"]  # a fenced answer
    bad = await r.complete(ask("c", schema=SCHEMA))
    assert bad.status == "failed" and "not valid JSON" in bad.reason
    wrong = await r.complete(ask("d", schema=SCHEMA))
    assert (
        wrong.status == "failed" and "expected string" in wrong.reason and "missing 'points'" in wrong.reason
    )
    assert (await r.complete(ask("c", schema=SCHEMA))).status == "failed"  # a failure is never cached

    assert schema_errors({"summary": "x", "points": ["a"], "extra": 1}, SCHEMA) == ["$: unexpected 'extra'"]
    assert schema_errors(True, {"type": "number"}) and not schema_errors(2, {"type": "number"})


async def test_refusals_truncation_timeouts_and_errors_are_reported_not_hidden():
    refused = await router(Scripted("I can't help with that.", stop="refusal")).complete(ask())
    assert refused.status == "refused" and not refused.ok
    cut = await router(Scripted('{"summary": "tru', stop="max_tokens")).complete(ask(schema=SCHEMA))
    assert cut.status == "failed" and "cut off" in cut.reason
    cut_text = await router(Scripted("partial", stop="max_tokens")).complete(ask())
    assert cut_text.status == "failed" and "cut off" in cut_text.reason

    slow = router(Scripted("ok", delay=0.5), brain_llm_timeout_seconds=0.05)
    timed_out = await slow.complete(ask())
    assert timed_out.status == "failed" and "no answer within" in timed_out.reason
    broken = router(Scripted(ConnectionError("upstream reset\nwith detail")))
    err = await broken.complete(ask())
    assert err.status == "failed" and err.reason == "ConnectionError: upstream reset"
    usage = broken.status()["usage"]
    assert usage["failed"] == 1 and usage["tokens"] == 0 and usage["estimated"] > 0  # charged conservatively
    assert broken.status()["recent"][0]["status"] == "failed"
    assert "prompt" not in json.dumps(broken.status())  # the record never holds the prompt or the answer


# ---------------------------------------------------------------------------------------------- the agent
def team_ctx():
    rng = np.random.default_rng(4)
    paths = {
        s: path(d, 0.012, i) for i, (s, d) in enumerate({"AAA": 0.002, "BBB": -0.001, "CCC": 0.0}.items())
    }
    paths["SPY"] = path(0.0004, 0.008, 9)
    ctx = make_ctx(paths)
    for s, score in {"AAA": 0.6, "BBB": -0.3, "CCC": 0.05}.items():
        ctx.working.add(
            Opinion(
                "momentum",
                "1",
                s,
                Stance.BULLISH if score > 0.15 else Stance.NEUTRAL,
                score,
                0.7,
                21,
                f"{s} momentum view",
                [Evidence("x", float(rng.random()), "d")],
            )
        )
    return ctx


async def test_the_briefing_agent_skips_without_a_model_and_never_votes_with_one():
    agent = BriefingAgent()
    ctx = team_ctx()
    assert agent.unavailable(ctx) == "no language model is attached to the brain"
    ctx.llm = ModelRouter(Settings(_env_file=None), FakeClock(NOW))
    assert "QP_BRAIN_LLM_PROVIDER=none" in agent.unavailable(ctx)

    answer = json.dumps(
        {"summary": "Momentum is positive.", "supporting": ["trend"], "opposing": [], "watch": "volume"}
    )
    fake = Scripted(answer)
    ctx.llm = router(fake, brain_llm_max_briefings=2)
    assert agent.unavailable(ctx) is None
    ops = await agent.analyze(ctx, agent.subjects(ctx))
    assert [o.subject for o in ops] == ["AAA", "BBB"]  # the strongest views, at most two
    for o in ops:
        assert o.stance is Stance.ABSTAIN and not o.directional and o.score == 0 and o.confidence == 0
        assert (
            o.meta["briefing"]["summary"] == "Momentum is positive." and o.meta["source"] == "language model"
        )
    request, model = fake.calls[0]
    assert model == "fast-model" and request.task is ModelTask.SUMMARIZE and request.schema is not None
    assert "AAA momentum view" in request.messages[0][1] and "do not calculate" in request.system
    assert "AAA momentum view" in facts(ctx, "AAA")

    ctx.llm = router(Scripted("not json"), brain_llm_max_briefings=1)
    (failed,) = await agent.analyze(ctx, agent.subjects(ctx))
    assert failed.stance is Stance.ABSTAIN and "not valid JSON" in failed.thesis
