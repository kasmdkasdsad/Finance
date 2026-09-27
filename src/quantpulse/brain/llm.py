"""Language models behind one interface, used only where words are the work.

The brain is deterministic. Indicators, statistics, correlations, betas, volatility forecasts, position
weights, spreads, quote ages, risk limits and everything about the account are calculated in Python, the
same way every time, and a language model is never asked for any of them (:data:`DETERMINISTIC_ONLY`
refuses such a request outright). A model may only do the tasks in :class:`ModelTask` — summarise,
extract, classify, research, argue, synthesise — and what it writes is context for people and for the
record: no model output sizes, approves or sends an order. Orders still go only through the deterministic
chain (risk engine → live-data and spread checks → trading controls → order manager → Alpaca paper).

**Providers.** :class:`ModelProvider` is the whole contract a vendor integration has to meet: say why it
cannot be used (a missing credential or package) and turn a :class:`ModelRequest` into a
:class:`ModelResponse`. The request is shaped like a chat-completions call — a system prompt, alternating
user/assistant messages, an output-token cap and an optional JSON schema for structured output — so it maps
directly onto current model APIs. No vendor is built in: the default provider is :class:`NoModelProvider`,
every model-backed agent then skips itself with the reason, and nothing pretends to be a model.
:func:`register_provider` plugs one in by name (``QP_BRAIN_LLM_PROVIDER``); it reads its own credential
from the environment and must never log or return it.

**Cost control** (:class:`ModelRouter`): each task goes to the *fast* or the *strong* tier (two model names
from settings; an unset tier is unavailable); a daily token budget is checked *before* a call with a
conservative estimate (input characters / 3 plus the output cap) and charged with the provider's reported
usage afterwards — it defaults to 0, so no call is ever made until someone sets it; answers are cached by
request for a TTL (a repeated question costs nothing); calls run under a concurrency limit and a timeout;
output is capped at ``QP_BRAIN_LLM_MAX_OUTPUT_TOKENS``. A structured answer must parse and match its
schema, a refusal is recorded as a refusal, and every failure is reported with its reason instead of being
papered over — the caller carries on without the model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal, Protocol

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK

from .types import ModelTier

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]
Status = Literal["ok", "cached", "skipped", "failed", "refused"]


class ModelTask(StrEnum):
    SUMMARIZE = "summarize"  # condense evidence the deterministic agents produced
    EXTRACT = "extract"  # structured fields from text (a filing, a headline)
    CLASSIFY = "classify"  # a label from a fixed set (the topic or tone of a news item)
    RESEARCH = "research"  # questions worth asking about an idea
    DEBATE = "debate"  # the strongest case for or against a view
    SYNTHESIS = "synthesis"  # a narrative across many findings


TASK_TIER: dict[ModelTask, ModelTier] = {
    ModelTask.SUMMARIZE: ModelTier.FAST,
    ModelTask.EXTRACT: ModelTier.FAST,
    ModelTask.CLASSIFY: ModelTier.FAST,
    ModelTask.RESEARCH: ModelTier.STRONG,
    ModelTask.DEBATE: ModelTier.STRONG,
    ModelTask.SYNTHESIS: ModelTier.STRONG,
}

# Calculated by code, never asked of a model.
DETERMINISTIC_ONLY = frozenset(
    {
        "rsi",
        "atr",
        "beta",
        "correlation",
        "volatility",
        "returns",
        "indicator",
        "position_weight",
        "position_size",
        "sizing",
        "spread",
        "quote_age",
        "risk_limit",
        "risk_check",
        "account",
        "pnl",
        "order",
    }
)


class DeterministicTaskError(ValueError):
    """A request asked a language model for something the brain calculates."""


def as_task(task: ModelTask | str) -> ModelTask:
    """The task, refusing anything the brain calculates deterministically."""
    name = (task.value if isinstance(task, ModelTask) else str(task)).strip().lower()
    if name in DETERMINISTIC_ONLY:
        raise DeterministicTaskError(f"{name!r} is calculated deterministically, never by a language model")
    try:
        return ModelTask(name)
    except ValueError:
        raise ValueError(f"unknown model task {name!r}; one of {[t.value for t in ModelTask]}") from None


@dataclass(frozen=True, slots=True)
class ModelRequest:
    task: ModelTask
    system: str
    messages: tuple[tuple[Role, str], ...]
    max_tokens: int = 600
    schema: dict[str, Any] | None = None  # JSON schema of the answer (structured output)
    purpose: str = ""  # who asks — an agent id — for the usage record

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", as_task(self.task))  # a plain string is checked the same way
        if not self.messages or self.messages[-1][0] != "user":
            raise ValueError("a request needs messages and must end with a user message")
        if any(role not in ("user", "assistant") for role, _ in self.messages):
            raise ValueError("message roles are 'user' or 'assistant' (the system prompt is separate)")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def ask(
        cls,
        task: ModelTask | str,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 600,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> ModelRequest:
        return cls(as_task(task), system, (("user", prompt),), max_tokens, schema, purpose)

    @property
    def tier(self) -> ModelTier:
        return TASK_TIER[self.task]

    def input_chars(self) -> int:
        return (
            len(self.system)
            + sum(len(text) for _, text in self.messages)
            + len(json.dumps(self.schema or {}))
        )

    def estimate_tokens(self) -> int:
        """A deliberately high estimate of the call's tokens (input at ~3 characters a token + the output
        cap), so the budget check errs on the side of not calling."""
        return self.input_chars() // 3 + 1 + self.max_tokens

    def key(self, model: str) -> str:
        blob = json.dumps(
            [model, self.task.value, self.system, self.messages, self.schema, self.max_tokens], sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str  # "end_turn", "max_tokens", "refusal", …
    cost_usd: float | None = None  # when the provider reports it
    parsed: Any = None  # the structured answer, if the provider parsed it already

    @property
    def tokens(self) -> int:
        return max(0, self.input_tokens) + max(0, self.output_tokens)


@dataclass(frozen=True, slots=True)
class ModelOutcome:
    """What happened to one request. Only ``ok`` and ``cached`` carry a usable answer."""

    status: Status
    task: str
    tier: str
    purpose: str = ""
    model: str | None = None
    response: ModelResponse | None = None
    parsed: Any = None  # the validated structured answer (requests with a schema)
    reason: str | None = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "cached") and self.response is not None

    def to_dict(self) -> dict[str, Any]:  # never the prompt or the answer, just what happened
        r = self.response
        return {
            "status": self.status,
            "task": self.task,
            "tier": self.tier,
            "purpose": self.purpose,
            "model": self.model,
            "reason": self.reason,
            "tokens": r.tokens if r is not None and self.status == "ok" else 0,
            "latency_ms": round(self.latency_ms, 1),
        }


class ModelProvider(Protocol):
    name: str

    def unavailable(self) -> str | None:
        """Why this provider cannot be used now (missing credential or package); ``None`` when it can."""

    async def complete(self, request: ModelRequest, model: str) -> ModelResponse:
        """One call to ``model``. Raise on a failed call; report a refusal through ``stop_reason``."""


class NoModelProvider:
    """The default: no language model. Model-backed agents skip themselves and say so."""

    name = "none"

    def unavailable(self) -> str | None:
        return "no language-model provider is configured (QP_BRAIN_LLM_PROVIDER=none)"

    async def complete(self, request: ModelRequest, model: str) -> ModelResponse:
        raise RuntimeError(self.unavailable())


class MissingProvider:
    """A provider named in settings that is not registered in this installation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def unavailable(self) -> str | None:
        return (
            f"language-model provider {self.name!r} is not installed (no integration is registered under it)"
        )

    async def complete(self, request: ModelRequest, model: str) -> ModelResponse:
        raise RuntimeError(self.unavailable())


ProviderFactory = Callable[[Settings], ModelProvider]
_PROVIDERS: dict[str, ProviderFactory] = {"none": lambda _s: NoModelProvider()}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Make a provider available under ``name`` (``QP_BRAIN_LLM_PROVIDER``)."""
    key = name.strip().lower()
    if key == "none":
        raise ValueError("'none' is reserved for running without a language model")
    _PROVIDERS[key] = factory


def unregister_provider(name: str) -> None:
    if name.strip().lower() != "none":
        _PROVIDERS.pop(name.strip().lower(), None)


def build_provider(settings: Settings) -> ModelProvider:
    name = settings.brain_llm_provider.strip().lower() or "none"
    factory = _PROVIDERS.get(name)
    return factory(settings) if factory is not None else MissingProvider(name)


# ------------------------------------------------------------------------------------------ schemas
_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Where ``value`` breaks ``schema`` (the JSON-schema subset used for answers: type, properties,
    required, additionalProperties false, items, enum, maxItems, maxLength)."""
    errors: list[str] = []
    kind = schema.get("type")
    kinds = [kind] if isinstance(kind, str) else list(kind or [])
    if kinds:
        ok = any(
            isinstance(value, _TYPES[k]) and not (k in ("number", "integer") and isinstance(value, bool))
            for k in kinds
            if k in _TYPES
        )
        if not ok:
            return [f"{path}: expected {'/'.join(kinds)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{path}: longer than {schema['maxLength']} characters")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [f"{path}: missing {k!r}" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected {k!r}" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errors += schema_errors(value[k], sub, f"{path}.{k}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errors


def parse_structured(text: str) -> Any:
    """The JSON value in a model's answer (a bare value, or one wrapped in a Markdown code fence)."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    return json.loads(body)


# ------------------------------------------------------------------------------------------ router
@dataclass
class _Usage:
    day: str = ""
    tokens: int = 0  # as reported by the provider
    estimated: int = 0  # charged for failed calls, whose usage the provider did not report
    cost_usd: float = 0.0
    calls: int = 0
    cached: int = 0
    failed: int = 0
    refused: int = 0
    skipped: int = 0
    by_purpose: dict[str, int] = field(default_factory=dict)  # tokens per asker

    @property
    def spent(self) -> int:
        return self.tokens + self.estimated

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "tokens": self.tokens,
            "estimated": self.estimated,
            "cost_usd": round(self.cost_usd, 4),
            "calls": self.calls,
            "cached": self.cached,
            "failed": self.failed,
            "refused": self.refused,
            "skipped": self.skipped,
            "by_purpose": dict(self.by_purpose),
        }


class UsageStore(Protocol):
    async def get_state(self, key: str) -> dict[str, Any] | None: ...

    async def set_state(self, key: str, value: dict[str, Any], now: Any) -> None: ...


USAGE_KEY = "llm_usage"
CACHE_LIMIT = 256


class ModelRouter:
    """Routes model requests by task, within the day's token budget, with a cache, a concurrency limit
    and a timeout. See the module docstring."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        provider: ModelProvider | None = None,
        store: UsageStore | None = None,
    ) -> None:
        self._clock = clock
        self._provider = provider or build_provider(settings)
        self._models = {
            ModelTier.FAST: settings.brain_llm_fast_model.strip(),
            ModelTier.STRONG: settings.brain_llm_strong_model.strip(),
        }
        self._budget = settings.brain_llm_daily_token_budget
        self._max_output = settings.brain_llm_max_output_tokens
        self._timeout = settings.brain_llm_timeout_seconds
        self._ttl = settings.brain_llm_cache_minutes * 60.0
        self._concurrency = settings.brain_llm_max_concurrency
        self.max_briefings = settings.brain_llm_max_briefings
        self._store = store
        self._cache: OrderedDict[str, tuple[float, ModelResponse, Any]] = OrderedDict()
        self._usage = _Usage()
        self._loaded = False
        self._reserved = 0  # estimated tokens of calls in flight
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self.recent: deque[dict[str, Any]] = deque(maxlen=25)

    # ------------------------------------------------------------------ state
    @property
    def provider(self) -> str:
        return self._provider.name

    def model_for(self, tier: ModelTier) -> str:
        return self._models.get(tier, "")

    def _today(self) -> str:
        return self._clock.now().astimezone(NEW_YORK).date().isoformat()

    def _roll(self) -> None:
        today = self._today()
        if self._usage.day != today:
            self._usage = _Usage(day=today)

    def remaining(self) -> int:
        self._roll()
        return max(0, self._budget - self._usage.spent - self._reserved)

    def unavailable(self, tier: ModelTier | None = None) -> str | None:
        """Why a call on ``tier`` (any tier when ``None``) cannot be made now; ``None`` when it can."""
        reason = self._provider.unavailable()
        if reason:
            return reason
        tiers = [tier] if tier is not None else [t for t in self._models if self._models[t]]
        if not tiers or not all(self._models.get(t) for t in tiers):
            which = tier.value if tier is not None else "fast or strong"
            return f"no {which} model is configured (QP_BRAIN_LLM_{(tier or ModelTier.FAST).value.upper()}_MODEL)"
        if self._budget <= 0:
            return "the daily token budget is 0 (QP_BRAIN_LLM_DAILY_TOKEN_BUDGET): no language-model calls"
        if self.remaining() <= 0:
            return f"today's token budget ({self._budget:,}) is spent"
        return None

    def status(self) -> dict[str, Any]:
        self._roll()
        return {
            "provider": self._provider.name,
            "available": self.unavailable() is None,
            "reason": self.unavailable(),
            "models": {t.value: (m or None) for t, m in self._models.items()},
            "daily_token_budget": self._budget,
            "remaining_tokens": self.remaining(),
            "max_output_tokens": self._max_output,
            "cache_minutes": round(self._ttl / 60),
            "cache_entries": len(self._cache),
            "usage": self._usage.to_dict(),
            "recent": list(self.recent),
            "deterministic_only": sorted(DETERMINISTIC_ONLY),
        }

    async def _load(self) -> None:
        """Today's usage survives restarts (so a restart never resets the budget)."""
        if self._loaded or self._store is None:
            self._loaded = True
            return
        try:
            saved = await self._store.get_state(USAGE_KEY)
        except Exception:  # the budget then starts from what this process has used
            logger.warning("could not read language-model usage; counting from zero", exc_info=True)
            saved = None
        self._loaded = True
        if saved and saved.get("day") == self._today():
            self._usage = _Usage(
                day=saved["day"],
                tokens=int(saved.get("tokens", 0)),
                estimated=int(saved.get("estimated", 0)),
                cost_usd=float(saved.get("cost_usd", 0.0)),
                calls=int(saved.get("calls", 0)),
                cached=int(saved.get("cached", 0)),
                failed=int(saved.get("failed", 0)),
                refused=int(saved.get("refused", 0)),
                skipped=int(saved.get("skipped", 0)),
                by_purpose=dict(saved.get("by_purpose", {})),
            )

    async def _save(self) -> None:
        if self._store is None:
            return
        try:
            await self._store.set_state(USAGE_KEY, self._usage.to_dict(), self._clock.now())
        except Exception:
            logger.warning("could not record language-model usage", exc_info=True)

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(self._concurrency), loop
        return self._sem

    def _note(self, outcome: ModelOutcome) -> ModelOutcome:
        self.recent.appendleft({**outcome.to_dict(), "at": self._clock.now().isoformat()})
        return outcome

    # ------------------------------------------------------------------ calls
    async def complete(self, request: ModelRequest) -> ModelOutcome:
        tier = request.tier
        base: dict[str, Any] = {"task": request.task.value, "tier": tier.value, "purpose": request.purpose}
        await self._load()
        self._roll()
        why = self.unavailable(tier)
        if why:
            self._usage.skipped += 1
            return self._note(ModelOutcome("skipped", reason=why, **base))
        model = self._models[tier]
        req = replace(request, max_tokens=min(request.max_tokens, self._max_output))
        key = req.key(model)
        hit = self._cache.get(key)
        if hit is not None and self._clock.monotonic() - hit[0] <= self._ttl:
            self._cache.move_to_end(key)
            self._usage.cached += 1
            return self._note(ModelOutcome("cached", model=model, response=hit[1], parsed=hit[2], **base))

        estimate = req.estimate_tokens()
        if estimate > self.remaining():  # checked and reserved with no await in between: no race
            self._usage.skipped += 1
            return self._note(
                ModelOutcome(
                    "skipped",
                    model=model,
                    reason=f"would exceed today's token budget (~{estimate:,} needed, "
                    f"{self.remaining():,} of {self._budget:,} left)",
                    **base,
                )
            )
        self._reserved += estimate
        started = time.perf_counter()
        try:
            async with self._semaphore():
                response = await asyncio.wait_for(self._provider.complete(req, model), timeout=self._timeout)
        except TimeoutError:
            return await self._failed(
                base, model, f"no answer within {self._timeout:.0f}s", started, estimate
            )
        except Exception as exc:  # the caller carries on without the model
            detail = str(exc).splitlines()[0][:200] if str(exc) else ""
            return await self._failed(base, model, f"{type(exc).__name__}: {detail}", started, estimate)
        finally:
            self._reserved -= estimate
        latency = (time.perf_counter() - started) * 1000
        self._usage.calls += 1
        self._usage.tokens += response.tokens
        self._usage.cost_usd += response.cost_usd or 0.0
        who = request.purpose or request.task.value
        self._usage.by_purpose[who] = self._usage.by_purpose.get(who, 0) + response.tokens

        outcome: ModelOutcome
        if response.stop_reason == "refusal":
            self._usage.refused += 1
            outcome = ModelOutcome("refused", model=model, response=response, reason="the model declined to "
                                   "answer", latency_ms=latency, **base)  # fmt: skip
        elif req.schema is not None:
            parsed, problem = self._structured(response, req.schema)
            if problem:
                self._usage.failed += 1
                outcome = ModelOutcome("failed", model=model, response=response, reason=problem,
                                       latency_ms=latency, **base)  # fmt: skip
            else:
                outcome = ModelOutcome("ok", model=model, response=response, parsed=parsed, latency_ms=latency,
                                       **base)  # fmt: skip
        elif response.stop_reason == "max_tokens":
            self._usage.failed += 1
            outcome = ModelOutcome("failed", model=model, response=response, reason="the answer was cut off at "
                                   f"{req.max_tokens} output tokens", latency_ms=latency, **base)  # fmt: skip
        else:
            outcome = ModelOutcome("ok", model=model, response=response, latency_ms=latency, **base)
        if outcome.status == "ok":
            self._cache[key] = (self._clock.monotonic(), response, outcome.parsed)
            while len(self._cache) > CACHE_LIMIT:
                self._cache.popitem(last=False)
        await self._save()
        return self._note(outcome)

    async def _failed(
        self, base: dict[str, Any], model: str, reason: str, started: float, estimate: int
    ) -> ModelOutcome:
        # a failed call may still have used tokens the provider did not report: charge the estimate
        self._usage.failed += 1
        self._usage.estimated += estimate
        await self._save()
        latency = (time.perf_counter() - started) * 1000
        return self._note(ModelOutcome("failed", model=model, reason=reason, latency_ms=latency, **base))

    @staticmethod
    def _structured(response: ModelResponse, schema: dict[str, Any]) -> tuple[Any, str | None]:
        if response.stop_reason == "max_tokens":
            return None, "the structured answer was cut off at the output-token cap"
        value = response.parsed
        if value is None:
            try:
                value = parse_structured(response.text)
            except (ValueError, IndexError):
                return None, "the answer was not valid JSON"
        errors = schema_errors(value, schema)
        if errors:
            return None, "the answer did not match its schema: " + "; ".join(errors[:3])
        return value, None
