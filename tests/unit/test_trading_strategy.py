"""Trading strategy maths: signal scoring, the market regime, sizing, exits and portfolio construction."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantpulse.domain import trading_regime as tr
from quantpulse.domain import trading_signals as ts
from quantpulse.domain.trading_portfolio import (
    Candidate,
    Holding,
    PositionMemory,
    StrategyConfig,
    build_plan,
    size_positions,
    waterfill,
)

NOW = datetime(2026, 9, 25, 14, 30, tzinfo=UTC)
WEIGHTS = {
    "momentum": 0.25,
    "trend": 0.2,
    "volume": 0.1,
    "volatility": 0.1,
    "fundamental": 0.1,
    "model": 0.2,
    "regime": 0.05,
}


def panel(drifts: dict[str, float], n: int = 300, seed: int = 3, vol: float = 0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-06-02", periods=n)
    cols = {}
    for i, (s, d) in enumerate(drifts.items()):
        cols[s] = 50 * (1 + i / 10) * np.exp(np.cumsum(d + rng.normal(0, vol, n)))
    close = pd.DataFrame(cols, index=idx)
    return close, close * 1.005, close * 0.995, pd.DataFrame(3e6, index=idx, columns=close.columns)


# ----------------------------------------------------------------------------- signals
def test_zscore_is_winsorised_and_scale_free():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0, np.nan])
    z = ts.zscore(s)
    assert z.max() <= ts.WINSOR and np.isnan(z.iloc[-1])
    assert ts.zscore(s * 1e6).round(9).equals(z.round(9))  # the unit of a signal never matters
    assert (ts.zscore(pd.Series([5.0, 5.0, 5.0])) == 0).all()
    assert ts.zscore(pd.Series([1.0, 2.0, np.nan])).isna().all()  # two values are not a cross-section


def test_components_are_standardised_so_none_dominates():
    close, high, low, vol = panel({f"S{i}": d for i, d in enumerate(np.linspace(-0.003, 0.003, 12))})
    raw = ts.raw_signals(close, high, low, vol, close.mean(axis=1), live_row=False)
    comps = ts.component_scores(raw, model_z={"S0": 50.0, "S1": -50.0}, beta_tilt=0.5)
    for c in ("momentum", "trend", "volume", "volatility"):
        assert comps[c].abs().max() <= ts.WINSOR + 1e-9
        assert abs(comps[c].mean()) < 1e-9
    assert comps["model"].abs().max() <= ts.WINSOR  # a huge model score is still one z-score
    assert set(comps.columns) == set(ts.COMPONENTS)


def test_strong_uptrends_rank_first_and_downtrends_last():
    drifts = {"UP1": 0.004, "UP2": 0.003, "FLAT": 0.0, "DN1": -0.003, "DN2": -0.004}
    close, high, low, vol = panel(drifts, vol=0.004)
    raw = ts.raw_signals(close, high, low, vol, close.mean(axis=1), live_row=False)
    score = ts.opportunity_score(ts.component_scores(raw), WEIGHTS)
    order = list(score.sort_values(ascending=False).index)
    assert set(order[:2]) == {"UP1", "UP2"} and set(order[-2:]) == {"DN1", "DN2"}
    assert bool(raw.loc["UP1", "trend_ok"]) and not bool(raw.loc["DN1", "trend_ok"])
    assert bool(raw.loc["DN2", "trend_broken"]) and not bool(raw.loc["UP1", "trend_broken"])
    assert raw.loc["UP1", "persistence"] > 0.8 and raw.loc["DN1", "persistence"] < -0.8
    assert (raw["risk_vol"] > 0).all() and (raw["adv_dollar"] > 0).all()


def test_live_row_uses_todays_snapshot_and_intraday_signals():
    close, high, low, vol = panel({"A": 0.002, "B": 0.001, "C": 0.0, "D": -0.001})
    day = close.index[-1] + pd.offsets.BDay(1)
    live = {
        s: ts.LiveBar(price=float(close[s].iloc[-1]) * 1.01, volume=2e6, vwap=float(close[s].iloc[-1]))
        for s in "ABC"
    }
    c2, h2, l2, v2 = ts.append_live_row(close, high, low, vol, live, day)
    assert (
        c2.index[-1] == day
        and np.isnan(c2.loc[day, "D"])
        and c2.loc[day, "A"] == pytest.approx(live["A"].price)
    )
    raw = ts.raw_signals(
        c2,
        h2,
        l2,
        v2,
        c2.mean(axis=1),
        live_row=True,
        vwap={s: b.vwap for s, b in live.items()},
        session_fraction=0.5,
    )
    assert raw.loc["A", "px_vs_vwap"] == pytest.approx(0.01)
    # 2M traded by mid-session vs a 3M daily average: 2 / (3 × 0.5) − 1
    assert raw.loc["A", "rel_volume"] == pytest.approx(2 / 1.5 - 1)
    assert raw.loc["A", "volume_trend"] == pytest.approx(0.0)  # the partial day is left out


def test_implied_vol_raises_risk_and_lowers_the_volatility_score():
    close, high, low, vol = panel({f"S{i}": 0.001 for i in range(6)})
    calm = {f"S{i}": 0.16 for i in range(6)}
    base = ts.raw_signals(close, high, low, vol, close.mean(axis=1), live_row=False, implied_vol=calm)
    rich = ts.raw_signals(
        close, high, low, vol, close.mean(axis=1), live_row=False, implied_vol={**calm, "S0": 0.9}
    )
    assert rich.loc["S0", "risk_vol"] == pytest.approx(0.9)
    assert rich.loc["S0", "risk_vol"] > base.loc["S0", "risk_vol"]
    assert (
        ts.component_scores(rich).loc["S0", "volatility"] < ts.component_scores(base).loc["S0", "volatility"]
    )


def test_fundamentals_are_oriented():
    close, high, low, vol = panel({f"S{i}": 0.0 for i in range(5)})
    raw = ts.raw_signals(close, high, low, vol, close.mean(axis=1), live_row=False)
    f = pd.DataFrame(
        {"earnings_yield": [0.10, 0.05, 0.02, 0.01, -0.02], "accruals": [-0.05, 0.0, 0.0, 0.02, 0.1]},
        index=[f"S{i}" for i in range(5)],
    )
    comps = ts.component_scores(raw, fundamentals=f)
    assert comps["fundamental"].idxmax() == "S0" and comps["fundamental"].idxmin() == "S4"


# ----------------------------------------------------------------------------- regime
def _series(drift: float, n: int = 800, vol: float = 0.006, seed: int = 1, shock: float = 0.0) -> pd.Series:
    """A trending index whose last month is calm (so realised volatility is not stressed) unless shocked."""
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, vol, n)
    r[-30:] = drift + rng.normal(0, vol * 0.5, 30)
    if shock:
        r[-30:] += shock + rng.normal(0, vol * 4, 30)
    return pd.Series(400 * np.exp(np.cumsum(r)), index=pd.bdate_range("2023-06-01", periods=n))


def test_regime_labels():
    bull = tr.classify(_series(0.0015), _series(0.0015, seed=2), breadth_200=0.7)
    assert bull.label == "bullish" and bull.beta_tilt > 0 and bull.trend_score >= 1.5
    bear = tr.classify(_series(-0.0012), _series(-0.0012, seed=2), breadth_200=0.3)
    assert bear.label in ("bearish", "risk_off") and bear.beta_tilt < 0
    crash = tr.classify(_series(0.0015, shock=-0.012), _series(0.0015, seed=2, shock=-0.012))
    assert crash.label == "risk_off" and any("below its 52-week high" in r for r in crash.reasons)
    fearful = tr.classify(_series(0.0015), _series(0.0015, seed=2), breadth_200=0.7, vix=40.0)
    assert fearful.label == "risk_off"
    jittery = tr.classify(_series(0.0015), _series(0.0015, seed=2), breadth_200=0.7, vix=27.0)
    assert jittery.label == "high_volatility" and jittery.stressed
    short = tr.classify(_series(0.001, n=100))
    assert short.label == "neutral" and "assuming a neutral market" in short.reasons[0]


def test_breadth():
    close = pd.DataFrame({"A": np.linspace(10, 20, 250), "B": np.linspace(20, 10, 250)})
    assert tr.breadth(close) == (0.5, 0.5)


# ----------------------------------------------------------------------------- sizing
def test_waterfill_caps_and_redistributes():
    w = waterfill({"A": 10.0, "B": 1.0, "C": 1.0}, {"A": 0.3, "B": 0.3, "C": 0.3}, 0.9)
    assert (
        w["A"] == pytest.approx(0.3)
        and w["B"] == pytest.approx(0.3)
        and sum(w.values()) == pytest.approx(0.9)
    )
    capped = waterfill({"A": 1.0, "B": 1.0}, {"A": 0.3, "B": 0.3}, 0.95)
    assert sum(capped.values()) == pytest.approx(0.6)  # the rest stays in cash


def cand(symbol, score, vol=0.25, **kw):
    return Candidate(symbol=symbol, score=score, price=100.0, risk_vol=vol, adv_dollar=5e9, **kw)


def test_conviction_and_volatility_drive_weights():
    cfg = StrategyConfig()
    names = {"STRONG": cand("STRONG", 2.0), "WEAK": cand("WEAK", 0.8), "WILD": cand("WILD", 2.0, vol=0.9)}
    t = size_positions(names, set(), 0.6, 100_000, cfg)
    assert t["STRONG"].weight > t["WEAK"].weight  # higher conviction, same volatility → larger
    assert t["WEAK"].weight > t["WILD"].weight  # same conviction as STRONG, 3.6× the volatility → smaller
    full = size_positions(names, set(), 0.95, 100_000, cfg)
    assert full["WILD"].weight == pytest.approx(cfg.position_vol_budget / 0.9)  # the volatility cap binds
    assert full["WILD"].capped_by == "volatility cap"
    assert max(x.weight for x in full.values()) <= cfg.max_position_pct + 1e-9
    assert sum(x.weight for x in full.values()) <= 0.95 + 1e-9


def test_liquidity_cap_and_tiny_weights_are_dropped():
    cfg = StrategyConfig(min_position_pct=0.05)
    thin = Candidate("THIN", 2.0, 100.0, 0.25, adv_dollar=200_000)
    t = size_positions({"THIN": thin, "A": cand("A", 1.0)}, set(), 0.9, 100_000, cfg)
    assert "THIN" not in t  # 1% of $200k/day is $2k: a 2% weight, under the 5% minimum


# ----------------------------------------------------------------------------- plan
def plan_for(cands, holdings=None, **kw):
    return build_plan(
        {c.symbol: c for c in cands},
        holdings or {},
        100_000.0,
        kw.pop("config", StrategyConfig()),
        now=NOW,
        **kw,
    )


def test_entries_are_concentrated_capped_per_order_and_sorted():
    cands = [cand(f"S{i}", 2.5 - i * 0.2) for i in range(12)]
    p = plan_for(cands)
    assert len(p.targets) == 8 and set(p.targets) == {f"S{i}" for i in range(8)}
    assert all(t.side == "buy" and t.notional <= 15_000 + 1e-6 for t in p.trades)
    assert [t.symbol for t in p.trades] == sorted((t.symbol for t in p.trades), key=lambda s: int(s[1:]))
    assert sum(t.weight for t in p.targets.values()) == pytest.approx(0.95, abs=1e-6)
    assert any("scaling in" in t.reason for t in p.trades)


def test_entry_bar_trend_filter_blocks_and_regime_penalty():
    cands = [
        cand("OK", 1.0),
        cand("LOW", 0.5),
        cand("BROKEN", 2.0, trend_ok=False),
        cand("EARN", 2.0, entry_blocks=("earnings",)),
    ]
    p = plan_for(cands)
    assert set(p.targets) == {"OK"}
    assert "trend is not intact" in p.skipped["BROKEN"] and p.skipped["EARN"] == "earnings"
    assert plan_for([cand("OK", 1.0)], entry_penalty=0.5).targets == {}  # a bearish market asks for more
    assert plan_for([cand("OK", 1.0)], regime_exposure=0.4).gross_target == pytest.approx(0.95 * 0.4)


def hold(symbol, qty=100, avg=100.0, price=100.0):
    return Holding(symbol, qty, avg, price, qty * price, price / avg - 1)


def test_stop_loss_exits_even_the_best_name():
    p = plan_for([cand("A", 3.0, trend_ok=True)], {"A": hold("A", avg=110.0, price=100.0)})
    [t] = p.trades
    assert (t.side, t.kind, t.qty, t.closes_position) == ("sell", "stop_loss", 100, True)
    assert t.risk_reducing


def test_take_profit_trims_once_per_entry_price():
    h = {"A": hold("A", avg=70.0, price=100.0)}
    p = plan_for([cand("A", 2.0)], h)
    [t] = p.trades
    assert t.kind == "take_profit" and t.qty == 50 and t.side == "sell"
    again = plan_for([cand("A", 2.0)], h, memory={"A": PositionMemory(profit_taken_basis=70.0)})
    assert all(x.kind != "take_profit" for x in again.trades)


def test_strategy_exits():
    h = {s: hold(s) for s in ("REV", "WEAK", "FLIP", "KEEP")}
    cands = [
        cand("REV", 1.0, trend_broken=True),
        cand("WEAK", -0.5),
        cand("FLIP", 1.0, model_z=-1.0),
        cand("KEEP", 1.0, model_z=0.9),
    ]
    p = plan_for(cands, h, memory={"FLIP": PositionMemory(entry_model_z=1.2)})
    kinds = {t.symbol: t for t in p.trades if t.side == "sell"}
    assert "trend reversed" in kinds["REV"].reason and "deteriorated" in kinds["WEAK"].reason
    assert "model turned negative" in kinds["FLIP"].reason and "KEEP" not in kinds


def test_risk_off_closes_weak_holdings():
    p = plan_for([cand("A", 0.5)], {"A": hold("A")}, risk_off=True)
    assert [t.kind for t in p.trades] == ["risk_off_exit"]


def test_stronger_names_displace_holdings_but_incumbents_get_a_head_start():
    cfg = StrategyConfig(max_positions=2)
    holdings = {"OLD1": hold("OLD1"), "OLD2": hold("OLD2")}
    close_call = plan_for([cand("OLD1", 1.0), cand("OLD2", 1.0), cand("NEW", 1.2)], holdings, config=cfg)
    assert set(close_call.targets) == {"OLD1", "OLD2"}  # 1.2 < 1.0 + 0.35: no churn for a small edge
    clear = plan_for([cand("OLD1", 1.0), cand("OLD2", 0.9), cand("NEW", 2.5)], holdings, config=cfg)
    assert set(clear.targets) == {"OLD1", "NEW"} and "displaced" in clear.exits["OLD2"]


def test_rebalance_band_cooldown_and_working_orders():
    h = {"A": hold("A", qty=290, price=100.0)}  # 29% held
    p = plan_for([cand("A", 2.5), cand("B", 2.4)], h)
    assert "within the rebalance band" in p.skipped.get("A", "") or all(t.symbol != "A" for t in p.trades)
    sold_recently = {"B": (NOW - timedelta(minutes=30), "sell")}
    p2 = plan_for([cand("B", 2.0)], last_traded=sold_recently)
    assert p2.trades == [] and "cooldown" in p2.skipped["B"]
    bought_recently = {"B": (NOW - timedelta(minutes=30), "buy")}
    assert [t.symbol for t in plan_for([cand("B", 2.0)], last_traded=bought_recently).trades] == ["B"]
    p3 = plan_for([cand("B", 2.0)], working={"B"})
    assert p3.trades == [] and "still working" in p3.skipped["B"]


def test_turnover_budget_defers_discretionary_trades_but_never_exits():
    cfg = StrategyConfig(max_cycle_turnover_pct=0.2)
    p = plan_for(
        [cand(f"S{i}", 2.0 - 0.1 * i) for i in range(5)] + [cand("X", 3.0)],
        {"X": hold("X", avg=120.0)},
        config=cfg,
    )
    buys = [t for t in p.trades if t.side == "buy"]
    assert sum(t.notional for t in buys) <= 20_000 + 1e-6
    assert any("turnover budget" in r for r in p.skipped.values())
    assert any(t.kind == "stop_loss" for t in p.trades)


def test_unscored_holdings_are_left_alone():
    p = plan_for([cand("A", 2.0)], {"GHOST": hold("GHOST")})
    assert all(t.symbol != "GHOST" for t in p.trades) and "not scored" in p.skipped["GHOST"]
