"""The Phase 2 specialist agents on generated price panels (run through the real indicator code), a test
stock-model snapshot, a test option chain and test earnings events. Each test checks a behaviour the
agent is meant to have, not a particular number."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantpulse.brain.agents.catalyst import CatalystAgent
from quantpulse.brain.agents.factor import FactorAgent
from quantpulse.brain.agents.fundamental import FundamentalAgent, ValuationAgent
from quantpulse.brain.agents.mean_reversion import MeanReversionAgent
from quantpulse.brain.agents.options import OptionsAgent
from quantpulse.brain.agents.statistical import StatisticalAgent, market_model, variance_ratio
from quantpulse.brain.agents.volatility import VolatilityAgent, garch_forecast
from quantpulse.brain.consensus import build_consensus
from quantpulse.brain.context import BrainContext, PortfolioState
from quantpulse.brain.decisions import plan
from quantpulse.brain.indicators import compute_indicators
from quantpulse.brain.research_data import chain_metrics
from quantpulse.brain.types import MARKET, BrainMode, BrainSession, DataState, Stance
from quantpulse.schemas.common import DataStatus
from quantpulse.schemas.model import LiveScore
from quantpulse.schemas.options import OptionChain, OptionContract
from quantpulse.services.model import ModelSnapshot
from quantpulse.services.trading_risk import RiskLimits

AS_OF = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
N = 320


def path(drift: float, vol: float, seed: int, tail: list[float] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = drift + vol * rng.standard_normal(N)
    if tail:
        r[-len(tail) :] = tail
    return 50 * np.exp(np.cumsum(r))


def panel(paths: dict[str, np.ndarray]) -> tuple[pd.DataFrame, ...]:
    idx = pd.bdate_range(end="2026-09-24", periods=N)
    close = pd.DataFrame(paths, index=idx)
    high, low = close * 1.005, close * 0.995
    volume = pd.DataFrame(2_000_000.0, index=idx, columns=close.columns)
    return close, high, low, volume


def make_ctx(
    paths: dict[str, np.ndarray], *, focus=None, model=None, options=None, events=None
) -> BrainContext:
    close, high, low, volume = panel(paths)
    bench = close.pop("SPY")
    for f in (high, low, volume):
        f.pop("SPY")
    ind = compute_indicators(close, high, low, volume, bench)
    focus = list(focus or close.columns)
    return BrainContext(
        as_of=AS_OF,
        session=BrainSession.OPEN,
        market_open=True,
        clock_source="calendar",
        mode=BrainMode.DRY_RUN,
        universe=list(close.columns),
        close=close,
        high=high,
        low=low,
        volume=volume,
        benchmark=bench,
        benchmark_symbol="SPY",
        qqq=None,
        price_status=DataStatus.LIVE,
        quotes={},
        quality={},
        missing_quotes={},
        indicators=ind,
        market_stats={},
        regime=None,
        vix=None,
        implied_vol={},
        earnings={},
        model_z={},
        fundamentals=None,
        sectors={},
        portfolio=PortfolioState(available=True),
        account=PortfolioState(available=True),
        data_states=dict.fromkeys(close.columns, DataState.FRESH),
        limits=RiskLimits(),
        kill_switch=False,
        model=model,
        options=options or {},
        events=events or {},
        focus=focus,
    )


async def run(agent, ctx, subjects=None):
    return {o.subject: o for o in await agent.analyze(ctx, subjects or agent.subjects(ctx))}


# ---------------------------------------------------------------------------------------------- mean reversion
async def test_mean_reversion_buys_the_dip_in_an_uptrend_and_does_not_fade_a_strong_trend():
    dip = [0.004] * 12 + [-0.02, -0.025, -0.02, -0.02, -0.015]  # uptrend, then a sharp pullback
    rip = [0.03] * 8  # an already strong uptrend running hot
    ctx = make_ctx(
        {"SPY": path(0.0005, 0.008, 1), "DIP": path(0.003, 0.01, 2, dip), "HOT": path(0.004, 0.01, 3, rip)}
    )
    ops = await run(MeanReversionAgent(), ctx)
    assert ops["DIP"].stance is Stance.BULLISH and "oversold" in ops["DIP"].thesis
    assert ops["DIP"].invalidation and "20-day low" in ops["DIP"].invalidation
    assert ops["HOT"].score < 0 and "fading a strong trend" in ops["HOT"].thesis
    assert ops["HOT"].confidence < ops["DIP"].confidence  # fading strength is not trusted


# ---------------------------------------------------------------------------------------------- statistical
def test_variance_ratio_and_market_model_recover_known_structure():
    rng = np.random.default_rng(7)
    e = rng.standard_normal(2000) * 0.01
    reverting, trending = np.zeros_like(e), np.zeros_like(e)
    for i in range(1, e.size):
        reverting[i] = -0.4 * reverting[i - 1] + e[i]
        trending[i] = 0.4 * trending[i - 1] + e[i]
    vr_rev, z_rev = variance_ratio(reverting)
    vr_tr, z_tr = variance_ratio(trending)
    assert vr_rev < 0.8 and z_rev < -3 and vr_tr > 1.2 and z_tr > 3
    b = rng.standard_normal(500) * 0.01
    beta, r2, resid = market_model(1.5 * b + rng.standard_normal(500) * 0.002, b)
    assert beta == pytest.approx(1.5, abs=0.05) and r2 > 0.9 and abs(resid.mean()) < 1e-3


async def test_statistical_agent_reports_regime_beta_and_residual():
    ctx = make_ctx({"SPY": path(0.0003, 0.01, 4), "AAA": path(0.0003, 0.015, 5)})
    o = (await run(StatisticalAgent(), ctx))["AAA"]
    assert o.meta["regime"] in {"mean-reverting", "trending", "no significant regime"}
    assert {"vr", "vr_z", "beta", "resid_z"} <= set(o.meta) and 0 <= o.confidence <= 0.6


# ---------------------------------------------------------------------------------------------- volatility
def test_garch_forecast_tracks_the_true_volatility():
    daily = 0.02
    closes = path(0.0, daily, 8)
    forecast, method = garch_forecast(closes)
    assert method in {"garch", "ewma"} and forecast == pytest.approx(daily * np.sqrt(252), rel=0.35)
    assert garch_forecast(closes[:30]) == (None, None)  # too short: no forecast rather than a guess


async def test_volatility_agent_scales_size_down_for_volatile_names_and_posts_its_forecast():
    ctx = make_ctx(
        {"SPY": path(0.0003, 0.008, 9), "CALM": path(0.0005, 0.006, 10), "WILD": path(0.0, 0.045, 11)}
    )
    ops = await run(VolatilityAgent(), ctx)
    assert ops["WILD"].meta["size_scale"] < ops["CALM"].meta["size_scale"] == 1.0
    assert ops["WILD"].meta["regime"] in {"elevated", "extreme"} and ops["CALM"].meta["regime"] == "calm"
    assert set(ctx.working.facts["vol_forecast"]) == {"CALM", "WILD"}
    assert MARKET in ops and ops[MARKET].meta["benchmark_rv21"] > 0


# ---------------------------------------------------------------------------------------------- stock model agents
def snapshot(
    features: pd.DataFrame, probs: dict[str, float], as_of=date(2026, 9, 24), sectors=None
) -> ModelSnapshot:
    live = {
        s: LiveScore(
            symbol=s,
            rank=i + 1,
            score=p - 0.5,
            z=(p - 0.5) * 8,
            rating=max(1, min(10, round(p * 10))),
            prob_outperform=p,
            expected_excess_return=(p - 0.5) / 5,
            sector=(sectors or {}).get(s),
        )
        for i, (s, p) in enumerate(probs.items())
    }
    return ModelSnapshot(
        live=live, features=features, as_of=as_of, data_status=DataStatus.LIVE, label="21d ensemble"
    )


def universe_features(n=40, seed=12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["gross_profitability", "roe", "accruals", "asset_growth", "earnings_yield", "fcf_yield",
            "book_to_market", "mom_12_1", "mom_6_1", "vol_63", "beta_252"]  # fmt: skip
    f = pd.DataFrame(
        rng.normal(0, 1, (n, len(cols))) * 0.05 + 0.08, columns=cols, index=[f"S{i}" for i in range(n)]
    )
    f.loc["QUAL"] = [
        0.45,
        0.35,
        -0.10,
        0.01,
        0.05,
        0.06,
        0.2,
        0.2,
        0.1,
        0.2,
        1.0,
    ]  # profitable, clean earnings
    f.loc["JUNK"] = [
        0.01,
        -0.20,
        0.20,
        0.45,
        -0.05,
        -0.02,
        0.9,
        -0.3,
        -0.2,
        0.6,
        1.4,
    ]  # losses, accruals, growth
    f.loc["CHEAP"] = [0.10, 0.12, 0.00, 0.05, 0.16, 0.14, 0.9, 0.1, 0.05, 0.25, 1.0]
    f.loc["RICH"] = [0.10, 0.12, 0.00, 0.05, 0.01, 0.00, 0.02, 0.1, 0.05, 0.25, 1.0]
    return f


async def test_fundamental_agent_ranks_quality_against_the_universe():
    feats = universe_features()
    ctx = make_ctx({"SPY": path(0.0003, 0.01, 13), "QUAL": path(0.0, 0.01, 14), "JUNK": path(0.0, 0.01, 15)},
                   model=snapshot(feats, {"QUAL": 0.6, "JUNK": 0.4}))  # fmt: skip
    ops = await run(FundamentalAgent(), ctx)
    assert ops["QUAL"].stance is Stance.BULLISH and ops["JUNK"].stance is Stance.BEARISH
    assert any(e.name == "accruals" for e in ops["JUNK"].evidence) and ops["QUAL"].horizon_days == 63
    no_model = make_ctx({"SPY": path(0.0003, 0.01, 13), "QUAL": path(0.0, 0.01, 14)})
    assert "not available" in FundamentalAgent().unavailable(no_model)


async def test_valuation_agent_ignores_losses_and_flags_value_traps():
    feats = universe_features()
    ctx = make_ctx(
        {"SPY": path(0.0003, 0.01, 16), "CHEAP": path(0.0, 0.01, 17), "RICH": path(0.0, 0.01, 18), "JUNK": path(-0.002, 0.012, 19)},
        model=snapshot(feats, {"CHEAP": 0.55, "RICH": 0.5, "JUNK": 0.45}),
    )  # fmt: skip
    ops = await run(ValuationAgent(), ctx)
    assert ops["CHEAP"].stance is Stance.BULLISH and ops["RICH"].stance is Stance.BEARISH
    junk = ops["JUNK"]
    assert "loss-making: earnings yield ignored" in junk.meta["checks"]
    assert all(e.name != "earnings_yield" for e in junk.evidence)
    trap = [c for c in junk.meta["checks"] if "value trap" in c]
    assert trap and junk.score < 0.5  # cheap on book value but falling and unprofitable


async def test_factor_agent_uses_the_calibrated_model_and_distrusts_a_stale_run():
    feats = universe_features()
    probs = {"QUAL": 0.62, "JUNK": 0.38}
    fresh = make_ctx({"SPY": path(0.0003, 0.01, 20), "QUAL": path(0.0, 0.01, 21), "JUNK": path(0.0, 0.01, 22)},
                     model=snapshot(feats, probs))  # fmt: skip
    ops = await run(FactorAgent(), fresh)
    assert ops["QUAL"].stance is Stance.BULLISH and ops["JUNK"].stance is Stance.BEARISH
    assert ops["QUAL"].meta["exposures"]["quality"] > 0
    stale = make_ctx({"SPY": path(0.0003, 0.01, 20), "QUAL": path(0.0, 0.01, 21)},
                     model=snapshot(feats, probs, as_of=date(2026, 9, 1)))  # fmt: skip
    old = (await run(FactorAgent(), stale))["QUAL"]
    assert old.confidence < ops["QUAL"].confidence and old.meta["model_age_days"] > 5


# ---------------------------------------------------------------------------------------------- options
def chain(
    spot=100.0, put_iv=0.45, call_iv=0.28, near_iv=0.40, far_iv=0.32, put_vol=3000, call_vol=1000
) -> OptionChain:
    today = AS_OF.date()
    contracts = []
    for days, atm in ((30, near_iv), (95, far_iv)):
        exp = today + timedelta(days=days)
        for k in (90, 92, 95, 100, 105, 108, 110):
            for kind in ("call", "put"):
                iv = atm
                if days == 30 and kind == "put" and k <= 92:
                    iv = put_iv
                if days == 30 and kind == "call" and k >= 108:
                    iv = call_iv
                contracts.append(
                    OptionContract(
                        contract_symbol=f"X{exp:%y%m%d}{kind[0].upper()}{k}",
                        kind=kind,
                        strike=k,
                        expiration=exp,
                        implied_volatility=iv,
                        volume=(put_vol if kind == "put" else call_vol) / 14,
                        open_interest=500,
                    )
                )
    return OptionChain(
        underlying="X",
        underlying_price=spot,
        as_of=AS_OF,
        expirations=sorted({c.expiration for c in contracts}),
        contracts=contracts,
    )


def test_chain_metrics_measure_skew_term_structure_and_flow():
    m = chain_metrics(chain(), 100.0, AS_OF.date())
    assert m["atm_iv"] == pytest.approx(0.40) and m["atm_iv_far"] == pytest.approx(0.32)
    assert m["term_slope"] == pytest.approx(-0.08) and m["skew"] == pytest.approx(0.17)
    assert m["pc_volume"] == pytest.approx(3.0) and m["volume_oi"] == pytest.approx(4000 / 14000)


async def test_options_agent_reads_fear_and_greed_only_from_real_chains():
    bearish = chain_metrics(chain(), 100.0, AS_OF.date())
    calm = chain_metrics(
        chain(put_iv=0.30, call_iv=0.29, near_iv=0.3, far_iv=0.33, put_vol=500, call_vol=1500),
        100.0,
        AS_OF.date(),
    )
    ctx = make_ctx({"SPY": path(0.0003, 0.01, 23), "FEAR": path(0.0, 0.01, 24), "CALM": path(0.0, 0.01, 25), "NONE": path(0.0, 0.01, 26)},
                   options={"FEAR": bearish, "CALM": calm})  # fmt: skip
    agent = OptionsAgent()
    assert agent.subjects(ctx) == ["FEAR", "CALM"]  # symbols without a live chain are not analysed
    ops = await run(agent, ctx)
    assert ops["FEAR"].stance is Stance.BEARISH and ops["CALM"].score > ops["FEAR"].score
    assert ops["FEAR"].meta["implied_move"] == pytest.approx(0.40 * np.sqrt(21 / 252), rel=1e-3)
    assert "no live option chains" in agent.unavailable(
        make_ctx({"SPY": path(0, 0.01, 1), "A": path(0, 0.01, 2)})
    )


# ---------------------------------------------------------------------------------------------- catalyst
async def test_catalyst_agent_drift_after_a_surprise_and_event_risk_before_a_release():
    events = {
        "BEAT": {"next": "2026-12-20", "days_to_next": 86, "typical_move": 0.04, "days_since_last": 10,
                 "last_abnormal": 0.12, "reactions": [], "next_source": "estimated"},
        "SOON": {"next": "2026-09-27", "days_to_next": 2, "typical_move": 0.06, "days_since_last": 85,
                 "last_abnormal": 0.02, "reactions": [], "next_source": "scheduled"},
    }  # fmt: skip
    ctx = make_ctx(
        {"SPY": path(0.0003, 0.01, 27), "BEAT": path(0.001, 0.01, 28), "SOON": path(0.001, 0.01, 29)},
        events=events,
    )
    ops = await run(CatalystAgent(), ctx)
    assert ops["BEAT"].stance is Stance.BULLISH and ops["BEAT"].meta["surprise"] == pytest.approx(3.0)
    soon = ops["SOON"]
    assert soon.stance is Stance.ABSTAIN and soon.confidence == 0.0  # context, not a vote
    assert soon.meta["days_to_next"] == 2 and "earnings in 2 days" in soon.thesis
    assert ctx.working.facts["event_risk"]["SOON"]["days_to_earnings"] == 2

    # the planner will not open a position two days before the release, however bullish the view
    c = build_consensus("SOON", [ops["BEAT"].__class__(**{**_fields(ops["BEAT"]), "subject": "SOON"}),
                                 ops["BEAT"].__class__(**{**_fields(ops["BEAT"]), "subject": "SOON", "agent_id": "x"})])  # fmt: skip
    proposals = plan(
        ctx,
        {"SOON": c},
        min_confidence=0.0,
        max_new=2,
        vol_budget=0.02,
        vol_floor=0.15,
        earnings_caution_days=3,
    )
    assert proposals[0].action.value == "watch" and "earnings in 2 day(s)" in proposals[0].reasons[0]


def _fields(o):
    return {f: getattr(o, f) for f in o.__slots__}
