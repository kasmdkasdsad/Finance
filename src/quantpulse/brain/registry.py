"""The agent registry: which agents exist, which are enabled, and which to run for a given context.

Agents are registered in code (see :mod:`quantpulse.brain.agents`) and mirrored into the ``brain_agents``
table, where they can be disabled at runtime. Selection is explicit and explained: each agent is either
*selected* (with the reason) or *skipped* (disabled, missing inputs, nothing to look at). Agents run in
dependency order, those at the same level concurrently, each with its own timeout — a failing agent is
recorded and the cycle carries on without it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .agents.base import Agent
from .context import BrainContext
from .types import Opinion

logger = logging.getLogger(__name__)


@dataclass
class Selection:
    agent: Agent
    subjects: list[str]
    reason: str


@dataclass
class Skip:
    agent_id: str
    version: str
    reason: str


@dataclass
class AgentRun:
    agent_id: str
    version: str
    status: str  # ok | failed | timeout
    started: float
    duration_ms: float
    subjects: int
    opinions: list[Opinion] = field(default_factory=list)
    error: str | None = None
    model_tier: str = "deterministic"
    cost: float = 0.0


class AgentRegistry:
    def __init__(self, agents: Iterable[Agent] = ()) -> None:
        self._agents: dict[str, Agent] = {}
        self._disabled: set[str] = set()
        for a in agents:
            self.register(a)

    def register(self, agent: Agent) -> None:
        spec = agent.spec
        if spec.id in self._agents:
            raise ValueError(f"agent {spec.id!r} is already registered")
        missing = [d for d in spec.dependencies if d not in self._agents]
        if missing:
            raise ValueError(f"agent {spec.id!r} depends on unregistered agents {missing}")
        self._agents[spec.id] = agent

    def get(self, agent_id: str) -> Agent:
        return self._agents[agent_id]

    def all(self) -> list[Agent]:
        return list(self._agents.values())

    def set_enabled(self, agent_id: str, enabled: bool) -> None:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        (self._disabled.discard if enabled else self._disabled.add)(agent_id)

    def enabled(self, agent_id: str) -> bool:
        return agent_id not in self._disabled

    def select(
        self, ctx: BrainContext, only: Sequence[str] | None = None
    ) -> tuple[list[Selection], list[Skip]]:
        """The agents worth running on ``ctx`` (optionally restricted to ``only``), each with a reason."""
        chosen: list[Selection] = []
        skipped: list[Skip] = []
        for agent in self._agents.values():
            spec = agent.spec
            if only is not None and spec.id not in only:
                continue
            if not self.enabled(spec.id):
                skipped.append(Skip(spec.id, spec.version, "disabled"))
                continue
            why_not = agent.unavailable(ctx)
            if why_not:
                skipped.append(Skip(spec.id, spec.version, why_not))
                continue
            subjects = agent.subjects(ctx)
            if not subjects:
                skipped.append(Skip(spec.id, spec.version, "nothing to analyse this cycle"))
                continue
            missing_dep = [d for d in spec.dependencies if d not in {s.agent.spec.id for s in chosen}]
            if missing_dep:
                skipped.append(Skip(spec.id, spec.version, f"its dependencies did not run: {missing_dep}"))
                continue
            chosen.append(
                Selection(agent, subjects, f"{len(subjects)} subject(s): {', '.join(subjects[:6])}")
            )
        return chosen, skipped

    @staticmethod
    def levels(selections: Sequence[Selection]) -> list[list[Selection]]:
        """Group selections so every agent runs after its stage's predecessors and the agents it depends
        on; agents in the same level run concurrently."""
        done: set[str] = set()
        out: list[list[Selection]] = []
        for stage in sorted({s.agent.spec.stage for s in selections}):
            pending = [s for s in selections if s.agent.spec.stage == stage]
            while pending:
                level = [s for s in pending if all(d in done for d in s.agent.spec.dependencies)]
                if not level:  # cannot happen: dependencies are checked at registration
                    raise RuntimeError("circular agent dependencies")
                out.append(level)
                done |= {s.agent.spec.id for s in level}
                pending = [s for s in pending if s not in level]
        return out

    async def run(self, ctx: BrainContext, selections: Sequence[Selection], timeout: float) -> list[AgentRun]:
        runs: list[AgentRun] = []
        for level in self.levels(selections):
            results = await asyncio.gather(*(self._run_one(ctx, s, timeout) for s in level))
            for run in results:
                for opinion in run.opinions:
                    ctx.working.add(opinion)  # later levels see earlier findings
                runs.append(run)
        return runs

    @staticmethod
    async def _run_one(ctx: BrainContext, sel: Selection, timeout: float) -> AgentRun:
        spec = sel.agent.spec
        started = time.perf_counter()
        wall = time.time()
        try:
            opinions = await asyncio.wait_for(sel.agent.analyze(ctx, sel.subjects), timeout=timeout)
            status, error = "ok", None
        except TimeoutError:
            opinions, status, error = [], "timeout", f"no answer within {timeout:.0f}s"
        except Exception as exc:  # one agent's failure never stops the cycle
            logger.exception("agent %s failed", spec.id)
            opinions, status, error = [], "failed", f"{type(exc).__name__}: {exc}"
        return AgentRun(
            agent_id=spec.id,
            version=spec.version,
            status=status,
            started=wall,
            duration_ms=(time.perf_counter() - started) * 1000,
            subjects=len(sel.subjects),
            opinions=[o for o in opinions if o.subject in sel.subjects],
            error=error,
            model_tier=spec.model_tier.value,
            cost=spec.cost if status == "ok" else 0.0,
        )
