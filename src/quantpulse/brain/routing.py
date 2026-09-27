"""Which agents a cycle needs, and how wide it looks — the orchestrator's cost control.

Not every agent runs every time. A ``full`` cycle runs the whole team on holdings, requested symbols,
detected opportunities and the pre-screen's best names. The cheaper kinds narrow both the agents and the
focus:

* ``portfolio`` — the holdings only, with the agents that manage positions (trend, momentum, volatility,
  earnings risk, statistics, research, posture);
* ``event`` — the symbols an event is about (plus holdings), with the agents that react to news in prices
  and positioning;
* ``deep`` — everything, with twice the pre-screen and opportunity budget (the weekend research pass).

An agent that cannot run on the data at hand still skips itself; routing only decides which ones are
worth asking.
"""

from __future__ import annotations

from dataclasses import dataclass

CORE = ("data_quality", "market_regime", "situational_awareness", "portfolio", "research")


@dataclass(frozen=True)
class Route:
    agents: tuple[str, ...] | None  # None: every registered agent
    pre_screen: float  # multiple of QP_BRAIN_FOCUS_CANDIDATES
    opportunities: float  # multiple of QP_BRAIN_MAX_OPPORTUNITIES


ROUTES: dict[str, Route] = {
    "full": Route(None, 1.0, 1.0),
    "deep": Route(None, 2.0, 2.0),
    "portfolio": Route(
        (*CORE, "technical", "momentum", "mean_reversion", "volatility", "statistical", "catalyst"), 0.0, 0.0
    ),
    "event": Route(
        (
            *CORE,
            "technical",
            "momentum",
            "mean_reversion",
            "volatility",
            "statistical",
            "catalyst",
            "options",
        ),
        0.0,
        0.0,
    ),
}
KINDS = tuple(ROUTES)


def route(kind: str) -> Route:
    return ROUTES.get(kind, ROUTES["full"])
