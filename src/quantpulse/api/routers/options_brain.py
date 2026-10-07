"""Options intelligence: the Options Brain's status, live chains, candidates and theses, the research lab's
strategies and evidence, experiments, learning, the option book (paper and shadow apart), its Greeks,
performance, counterfactuals and missed opportunities.

Everything here reads; the two actions (a research run, a learning pass) never place an order. Option orders
are only ever sent by the Brain's cycle through the trading service — Alpaca PAPER only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from quantpulse.api.deps import ContainerDep, local_only_allowed
from quantpulse.api.params import normalise_symbol
from quantpulse.brain.options import agents as A
from quantpulse.core.errors import ProviderError
from quantpulse.services.container import Container

router = APIRouter(prefix="/options", tags=["options intelligence"])


async def control_allowed(request: Request) -> None:
    local_only_allowed(request, "options research controls")


ControlAuth = [Depends(control_allowed)]


async def options_summary(c: Container) -> dict[str, Any]:
    s = c.settings
    open_ = await c.options_brain.positions(status="open")
    last = await c.brain.store.get_state("options_last_cycle")
    return {
        "paper_only": True,
        "enabled": s.options_enabled,
        "execution": s.options_enabled and s.options_execution,
        "priority_weight": s.options_priority_weight,
        "open_positions": {"paper": sum(p["mode"] == "paper" for p in open_),
                           "shadow": sum(p["mode"] == "shadow" for p in open_)},
        "lab": await c.options_lab.counts(),
        "last_cycle": {k: (last or {}).get(k) for k in ("cycle_id", "at", "orders", "notes", "no_trade")} if last else None,
    }  # fmt: skip


@router.get(
    "/status", summary="The Options Brain: switches, limits, agents, the lab, the book, the last pass"
)
async def status(c: Container = ContainerDep) -> dict[str, Any]:
    s = c.settings
    data = c.options_data
    return {
        **await options_summary(c),
        "feed": getattr(data, "feed", None),
        "feed_note": "indicative: derived from OPRA, not firm — paper limit orders only; research uses "
        "model-priced chains (no historical option quotes), always labelled",
        "data_configured": bool(data is not None and data.configured()),
        "universe": s.options_universe,
        "scan": {"size": s.options_scan_size, "per_cycle": s.options_scan_per_cycle,
                 "last": (c.options_brain.last or {}).get("scan")},
        "allowed_structures": s.options_allowed_structures,
        "limits": {
            "max_loss_per_trade": s.options_max_loss_per_trade,
            "max_loss_pct_per_trade": s.options_max_loss_pct_per_trade,
            "max_total_risk_pct": s.options_max_total_risk_pct,
            "max_underlying_risk_pct": s.options_max_underlying_risk_pct,
            "max_positions": s.options_max_positions,
            "max_contracts": s.options_max_contracts,
            "dte": [s.options_min_dte, s.options_max_dte],
            "close_dte": s.options_close_dte,
            "max_spread_pct": s.options_max_spread_pct,
            "max_quote_age_seconds": s.options_max_quote_age_seconds,
            "min_open_interest": s.options_min_open_interest,
            "max_delta_pct": s.options_max_delta_pct,
            "max_vega_pct": s.options_max_vega_pct,
            "exploration": s.options_exploration,
            "exploration_max_loss": s.options_exploration_max_loss,
        },
        "never": ["live trading", "naked short options", "undefined-risk structures", "0DTE execution",
                  "exercising an option", "model or recorded quotes as execution quotes"],
        "agents": [{"agent": name, "question": q, "weight": w} for name, _, w, q in A.AGENTS]
        + [{"agent": name, "question": q, "weight": None, "where": "research lab"} for name, q in A.LAB_AGENTS],
        "last_pass": c.options_brain.last,
    }  # fmt: skip


@router.get("/chains", summary="A live option chain with its data quality and volatility structure")
async def chains(
    underlying: str = Query(..., min_length=1, max_length=10), c: Container = ContainerDep
) -> dict[str, Any]:
    from quantpulse.options.analytics import by_expiry, constant_maturity_iv, term_structure

    data = c.options_data
    if data is None or not data.configured():
        raise HTTPException(503, "options market data is not configured (Alpaca paper keys)")
    u = normalise_symbol(underlying)
    now = c.clock.now()
    try:
        chain = await data.chain(u)
    except ProviderError as exc:
        raise HTTPException(502, f"option chain for {u} unavailable: {exc}") from None
    exps = by_expiry(chain.quotes, chain.underlying_price, now)
    return {
        "underlying": u,
        "spot": chain.underlying_price,
        "feed": chain.feed,
        "fetched_at": chain.fetched_at.isoformat(),
        "quality": chain.quality(now),
        "term_structure": term_structure(exps),
        "atm_iv_30d": constant_maturity_iv(exps, 30),
        "expirations": [e.__dict__ for e in exps],
        "contracts": [{"symbol": q.symbol, "expiration": q.contract.expiration.isoformat(), "kind": q.contract.kind,
                       "strike": q.contract.strike, "bid": q.bid, "ask": q.ask, "iv": q.iv, "delta": q.greeks.delta,
                       "open_interest": q.open_interest, "age_seconds": q.age(now)} for q in chain.quotes[:600]],
        "notes": chain.notes,
    }  # fmt: skip


@router.get("/candidates", summary="Candidates considered, with their thesis, debate, agents and verdict")
async def candidates(
    limit: int = Query(50, ge=1, le=500), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    return await c.options_brain.candidates(limit)


@router.get("/strategies", summary="The strategy population: every version, its stage and evidence")
async def strategies(
    stage: str | None = Query(None), limit: int = Query(200, ge=1, le=1000), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    return await c.options_lab.strategies(stage=stage, limit=limit)


@router.get(
    "/strategies/{version_id}", summary="One strategy version: backtests, walk-forward, stress, lineage"
)
async def strategy(version_id: int = Path(..., ge=1), c: Container = ContainerDep) -> dict[str, Any]:
    out = await c.options_lab.strategy(version_id)
    if out is None:
        raise HTTPException(404, f"no strategy version {version_id}")
    return out


@router.get("/research", summary="The research library: sources, their claims (hypotheses) and tests")
async def research(c: Container = ContainerDep) -> dict[str, Any]:
    return {"sources": await c.options_lab.research_library(), "last_run": c.options_lab.last_run,
            "knowledge": await c.options_lab.knowledge(200)}  # fmt: skip


@router.post("/research/run", dependencies=ControlAuth, summary="Start a budgeted research run (background)")
async def run_research(c: Container = ContainerDep) -> dict[str, Any]:
    job = c.options_lab.start_research()
    return {"job_id": job.id, "status": job.status, "note": "research only: nothing is traded"}


@router.get("/experiments", summary="The experiment queue: hypotheses, changes, decisions")
async def experiments(
    limit: int = Query(100, ge=1, le=1000), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    return await c.options_lab.experiments(limit)


@router.get("/learning", summary="What the Options Brain has learned (weights, lessons, calibration)")
async def learning(c: Container = ContainerDep) -> dict[str, Any]:
    from sqlalchemy import select

    from quantpulse.db.options_models import (
        OptionsLearningEventRow,
        OptionsLessonRow,
        OptionsStrategyWeightRow,
    )
    from quantpulse.options.lab.learning import calibration

    async with c.db.session() as s:
        weights = (await s.scalars(select(OptionsStrategyWeightRow).order_by(OptionsStrategyWeightRow.updated_at.desc())
                                   .limit(200))).all()  # fmt: skip
        lessons = (
            await s.scalars(select(OptionsLessonRow).order_by(OptionsLessonRow.created_at.desc()).limit(100))
        ).all()
        events = (await s.scalars(select(OptionsLearningEventRow).where(OptionsLearningEventRow.kind == "trade_graded")
                                  .order_by(OptionsLearningEventRow.at.desc()).limit(2000))).all()  # fmt: skip
    cal = {}
    for mode in ("shadow", "paper"):
        pts = [
            (e.predicted, e.actual)
            for e in events
            if e.evidence == mode and e.predicted is not None and e.actual is not None
        ]
        cal[mode] = calibration([p for p, _ in pts], [bool(a) for _, a in pts]) if pts else None
    return {
        "weights": [{"strategy": w.strategy_key, "regime": w.regime, "structure": w.structure, "vol_state": w.vol_state,
                     "p_edge": w.weight, "mean": w.mean, "n": w.n} for w in weights],
        "lessons": [{"memory": x.memory, "observation": x.observation, "hypothesis": x.hypothesis,
                     "confidence": x.confidence, "n": x.sample_size, "status": x.status} for x in lessons],
        "calibration": cal,
        "graded": {m: sum(e.evidence == m for e in events) for m in ("shadow", "paper")},
        "note": "shadow and paper evidence are kept apart; weights are shrunk toward zero and recency-weighted",
    }  # fmt: skip


@router.post("/learn", dependencies=ControlAuth, summary="Run the learning pass now")
async def learn(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.options_brain.learn()


@router.get("/portfolio", summary="Open option positions (paper and shadow apart) and the book's risk")
async def portfolio(c: Container = ContainerDep) -> dict[str, Any]:
    positions = await c.options_brain.positions(status="open")
    pending = await c.options_brain.positions(status="pending") + await c.options_brain.positions(
        status="closing"
    )
    risk: dict[str, dict[str, float]] = {"paper": {}, "shadow": {}}
    for p in positions:
        risk[p["mode"]][p["underlying"]] = risk[p["mode"]].get(p["underlying"], 0.0) + float(
            p["max_loss"] or 0
        )
    return {"positions": positions, "pending": pending,
            "max_loss_by_underlying": risk, "total_max_loss": {m: round(sum(v.values()), 2) for m, v in risk.items()}}  # fmt: skip


@router.get("/positions", summary="Option positions (filter by mode and status)")
async def positions(
    mode: str | None = Query(None, pattern="^(paper|shadow)$"),
    status: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.options_brain.positions(mode=mode, status=status, limit=limit)


@router.get("/greeks", summary="The option book's net Greeks (from the latest marks)")
async def greeks(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.options_brain.greeks()


@router.get("/performance", summary="Shadow and paper results, never mixed (research is in /strategies)")
async def performance(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.options_brain.performance()


@router.get("/counterfactuals", summary="What the alternatives would have done")
async def counterfactuals(
    limit: int = Query(100, ge=1, le=1000), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    return await c.options_brain.counterfactuals(limit)


@router.get("/missed-opportunities", summary="Rejected candidates, graded once the market has spoken")
async def missed(limit: int = Query(100, ge=1, le=1000), c: Container = ContainerDep) -> dict[str, Any]:
    return await c.options_brain.missed(limit)
