"""The versioned model registry: statistical, machine-learning and AI models as candidates — never trusted
because they look good in sample.

A model is registered under a *slot* (what it is for: ``stock_forecast``, ``iv_forecast``,
``fill_probability``, ``regime_classifier``, ``strategy_selector``, …) with a version, its kind, its parameters,
the data it was fitted on, and its metrics — in-sample metrics are stored for the record and **never read by a
gate**. Stages::

    CANDIDATE → OOS_VALIDATED → WALK_FORWARD_VALIDATED → STRESS_VALIDATED → PAPER_SHADOW → AUTHORITATIVE
                                                                            (any) → RETIRED

One stage at a time, each with its evidence. AUTHORITATIVE means the slot's champion: a challenger replaces
the champion only after shadowing it on live data and beating it out of sample by a margin; the previous
champion is kept (RETIRED, with its history), never deleted. Kinds ``ai`` (language or other opaque models)
additionally need a person's approval before AUTHORITATIVE, and none of them may ever send an order: models
inform the Brain; the trading service, the risk engine and the order manager decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ModelStage(StrEnum):
    CANDIDATE = "CANDIDATE"
    OOS_VALIDATED = "OOS_VALIDATED"
    WALK_FORWARD_VALIDATED = "WALK_FORWARD_VALIDATED"
    STRESS_VALIDATED = "STRESS_VALIDATED"
    PAPER_SHADOW = "PAPER_SHADOW"
    AUTHORITATIVE = "AUTHORITATIVE"
    RETIRED = "RETIRED"


ORDER = [ModelStage.CANDIDATE, ModelStage.OOS_VALIDATED, ModelStage.WALK_FORWARD_VALIDATED,
         ModelStage.STRESS_VALIDATED, ModelStage.PAPER_SHADOW, ModelStage.AUTHORITATIVE]  # fmt: skip
KINDS = ("statistical", "ml", "ai", "rule")
IN_SAMPLE_KEYS = frozenset({"in_sample", "train", "fit"})  # never consulted by a gate


@dataclass
class ModelEvidence:
    oos: dict[str, float | None] = field(
        default_factory=dict
    )  # held-out metrics: {"score": …, "baseline": …, "n": …}
    walk_forward: dict[str, Any] = field(
        default_factory=dict
    )  # {"passed": bool, "windows_positive": int, "windows": int}
    stress: dict[str, Any] = field(default_factory=dict)  # {"passed": bool}
    shadow: dict[str, float | None] = field(default_factory=dict)  # {"n": …, "score": …, "champion_score": …}
    human_approved: bool = False
    in_sample: dict[str, Any] = field(default_factory=dict)  # recorded, never used to promote


def gate(target: ModelStage, ev: ModelEvidence, *, kind: str, min_oos: int = 50, min_shadow: int = 30,
         margin: float = 0.0) -> list[str]:  # fmt: skip
    """Why ``target`` is not earned yet (empty: it is). Higher score is better; ``baseline`` is the naive
    benchmark (e.g. yesterday's value, the historical mean, the current champion)."""
    r: list[str] = []
    if target == ModelStage.OOS_VALIDATED:
        n, s, b = ev.oos.get("n") or 0, ev.oos.get("score"), ev.oos.get("baseline")
        if n < min_oos:
            r.append(f"{n} out-of-sample observations (minimum {min_oos})")
        if s is None or b is None or s <= b:
            r.append(f"out-of-sample score {s} does not beat the baseline {b}")
    elif target == ModelStage.WALK_FORWARD_VALIDATED:
        if not ev.walk_forward.get("passed"):
            r.append("walk-forward not passed")
    elif target == ModelStage.STRESS_VALIDATED:
        if not ev.stress.get("passed"):
            r.append("stress not passed")
    elif target == ModelStage.PAPER_SHADOW:
        pass  # entering shadow needs only the offline stages before it
    elif target == ModelStage.AUTHORITATIVE:
        n, s, champ = ev.shadow.get("n") or 0, ev.shadow.get("score"), ev.shadow.get("champion_score")
        if n < min_shadow:
            r.append(f"{n} shadow observations on live data (minimum {min_shadow})")
        if s is None or (champ is not None and s <= champ + margin):
            r.append(f"shadow score {s} does not beat the champion {champ} by {margin}")
        if kind == "ai" and not ev.human_approved:
            r.append("an AI model needs a person's approval before it becomes authoritative")
    return r


def advance(current: ModelStage, ev: ModelEvidence, *, kind: str, **kw: Any) -> tuple[ModelStage, list[str]]:
    if current in (ModelStage.AUTHORITATIVE, ModelStage.RETIRED):
        return current, ["no further stage"]
    nxt = ORDER[ORDER.index(current) + 1]
    missing = gate(nxt, ev, kind=kind, **kw)
    return (nxt if not missing else current), missing


def uses_in_sample(reasons: list[str]) -> bool:
    """For the audit: a gate reason may never mention in-sample evidence."""
    return any(k in r.lower() for r in reasons for k in ("in-sample", "in sample", "train"))
