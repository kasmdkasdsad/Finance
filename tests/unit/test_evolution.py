"""The Market Evolution Monitor on synthetic series: microstructure at several timescales, distribution shifts
and change points, relationships that weaken, vanish or invert, competing hypotheses (none assumed — AI
included), re-validation targets, and the model registry (no model is promoted on in-sample results)."""

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from quantpulse.evolution import hypotheses as hyp
from quantpulse.evolution import microstructure as ms
from quantpulse.evolution import monitor as mon
from quantpulse.evolution import registry as reg
from quantpulse.evolution import relationships as rel
from quantpulse.evolution import shifts


def session(seed: int, *, noise_bps: float = 0.0, vol: float = 0.2, reversal: float = 0.0, n: int = 390):
    """One session of 1-minute prices: an efficient price plus bid-ask bounce noise (and optional reversal)."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 9, 25, 13, 30, tzinfo=UTC)
    ts = [t0 + timedelta(minutes=i) for i in range(n)]
    eff = 100 * np.exp(np.cumsum(rng.normal(0, vol / math.sqrt(252 * 390), n)))
    bounce = noise_bps / 1e4 * 100 * np.where(rng.random(n) < 0.5, -1, 1)
    px = eff + bounce
    if reversal:
        r = np.diff(np.log(px))
        r[1:] -= reversal * r[:-1]
        px = 100 * np.exp(np.concatenate([[0], np.cumsum(r)]))
    vol_ = rng.integers(1000, 5000, n).astype(float)
    return ts, px, vol_


# --------------------------------------------------------------------------- microstructure
def test_micro_volatility_signature_and_noise():
    ts, px, v = session(1)
    clean = ms.session_metrics(ts, px, v)
    ts, px, v = session(1, noise_bps=5)
    noisy = ms.session_metrics(ts, px, v, bid=px - 0.02, ask=px + 0.02)
    assert set(clean) >= {"rv_1m", "rv_5m", "rv_15m", "rv_30m", "rv_60m", "variance_ratio_5_1", "jump_share",
                          "autocorr_1m", "amihud", "volume_open_share", "micro_vol"}  # fmt: skip
    assert 0.1 < clean["rv_5m"] < 0.35  # the efficient price's 20% volatility, roughly
    assert (
        noisy["rv_1m"] > noisy["rv_30m"] and noisy["noise_ratio"] > clean["noise_ratio"]
    )  # noise inflates fine scales
    assert (
        noisy["autocorr_1m"] < -0.2 and noisy["variance_ratio_5_1"] < 0.8
    )  # bounce = short-horizon reversal
    assert noisy["roll_spread_bps"] is not None and noisy["quoted_spread_bps"] == pytest.approx(4.0, rel=0.05)
    assert ms.session_metrics(ts[:10], px[:10])["note"].startswith("too few")
    panel = ms.daily_panel([clean, noisy])
    assert panel["rv_1m"] == [clean["rv_1m"], noisy["rv_1m"]]


# --------------------------------------------------------------------------- shifts and change points
def test_distribution_shifts_are_detected_and_classified():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 200).tolist()
    same = shifts.compare(ref, rng.normal(0, 1, 40).tolist())
    up = shifts.compare(ref, rng.normal(1.5, 1, 40).tolist())
    wide = shifts.compare(ref, rng.normal(0, 3, 40).tolist())
    assert same["kind"] == "none" and same["p_value"] > 0.05
    assert up["kind"] == "level_up" and up["effect_sd"] > 1 and up["p_value"] < 0.001 and up["psi"] > 0.25
    assert wide["kind"] == "more_dispersed" and wide["variance_ratio"] > 4
    assert "note" in shifts.compare([1, 2], [3])
    series = rng.normal(0, 1, 100).tolist() + rng.normal(3, 1, 100).tolist()
    cps = shifts.change_points(series)
    assert cps and abs(cps[0] - 100) <= 5
    assert shifts.cusum(series)["direction"] == "up" and shifts.cusum(series)["alarm"] >= 100
    assert shifts.change_points(rng.normal(0, 1, 200).tolist()) == []


# --------------------------------------------------------------------------- relationships
def test_relationships_weaken_disappear_invert_and_keep_their_history():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 300)

    def with_slope(b, n=300):
        return (b * x[:n] + rng.normal(0, 1, n)).tolist()

    established = rel.estimate(x.tolist(), with_slope(1.0))
    assert established.significant and established.slope == pytest.approx(1.0, abs=0.2)
    assert rel.status(established, rel.estimate(x.tolist(), with_slope(1.0)))["status"] == "STABLE"
    assert rel.status(established, rel.estimate(x.tolist(), with_slope(0.4)))["status"] == "WEAKENED"
    assert rel.status(established, rel.estimate(x.tolist(), with_slope(0.0)))["status"] == "DISAPPEARED"
    assert rel.status(established, rel.estimate(x.tolist(), with_slope(-0.8)))["status"] == "INVERTED"
    assert rel.status(established, rel.estimate(x.tolist(), with_slope(2.0)))["status"] == "STRENGTHENED"
    nothing = rel.estimate(x.tolist(), rng.normal(0, 1, 300).tolist())
    assert rel.status(nothing, rel.estimate(x.tolist(), with_slope(1.0)))["status"] == "EMERGED"
    assert rel.status(rel.estimate([1.0] * 3, [2.0] * 3), established)["status"] == "INSUFFICIENT"
    history = rel.rolling(x.tolist(), with_slope(1.0), 100)
    assert len(history) == 3 and all(h.n == 100 for h in history)  # every window kept, none overwritten


# --------------------------------------------------------------------------- competing hypotheses
def test_every_hypothesis_is_listed_and_none_is_assumed():
    names = {h.name for h in hyp.CATALOGUE}
    assert {"chance", "data_artifact", "macro_regime", "liquidity_change", "automated_liquidity"} <= names
    empty = hyp.evaluate({})
    assert len(empty) == len(hyp.CATALOGUE) and {r["verdict"] for r in empty} <= {"untestable"}
    ai = next(r for r in empty if r["name"] == "automated_liquidity")
    assert ai["identifiable_from_prices"] is False
    ev = {"q_value": 0.001, "persisted": True, "feed_changed": False, "data_gaps": 0.0, "benchmark_shift": True,
          "spread_change": -0.1, "illiquidity_change": -0.2, "autocorr_1m_change": -0.1, "variance_ratio_change": -0.1,
          "shift_without_event_days": True, "close_volume_share_change": 0.0, "universe_changed": False}  # fmt: skip
    results = {r["name"]: r for r in hyp.evaluate(ev)}
    assert (
        results["chance"]["verdict"] == "inconsistent"
        and results["data_artifact"]["verdict"] == "inconsistent"
    )
    assert results["macro_regime"]["verdict"] == "consistent"
    auto = results["automated_liquidity"]
    assert auto["verdict"] == "consistent" and "not identified by prices alone" in auto["detail"]
    text = hyp.summary(list(results.values()))
    assert "none is established" in text and "macro_regime" in text and "automated_liquidity" in text


# --------------------------------------------------------------------------- the monitor
def test_the_monitor_controls_false_discoveries_and_names_what_to_retest():
    rng = np.random.default_rng(4)
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(200)]
    series = []
    for i in range(30):  # thirty stable series: some will look odd by chance
        series.append(
            mon.Series(
                "volatility",
                f"U{i}",
                "rv20",
                "1d",
                list(zip(days, rng.normal(0.2, 0.02, 200).tolist(), strict=True)),
            )
        )
    shifted = rng.normal(0.2, 0.02, 200)
    shifted[-20:] += 0.08
    series.append(
        mon.Series("micro_volatility", "AAA", "rv_1m", "1m", list(zip(days, shifted.tolist(), strict=True)))
    )
    rows = mon.scan(series)
    sig = [r for r in rows if r["significant"]]
    assert [r["subject"] for r in sig] == ["AAA"]  # only the real change survives the FDR control
    assert (
        sig[0]["kind"] == "level_up"
        and sig[0]["q_value"] < 0.01
        and sig[0]["recent_window"][1] == days[-1].isoformat()
    )
    again = mon.scan(series, previous={r["key"]: r for r in rows})
    assert next(r for r in again if r["subject"] == "AAA")["persisted"] is True
    strategies = [
        {"key": "s1", "stage": "PAPER_ACTIVE", "underlyings": ["AAA"], "genome": {"max_spread_pct": 0.1}},
        {"key": "s2", "stage": "PAPER_ACTIVE", "underlyings": ["BBB"], "genome": {"max_spread_pct": 0.1}},
        {"key": "s3", "stage": "RESEARCH", "underlyings": ["AAA"], "genome": {}},
    ]
    assert mon.revalidation_targets(sig[0], strategies) == ["s1"]
    assert mon.revalidation_targets({**sig[0], "significant": False}, strategies) == []


def test_explaining_outcomes_by_micro_volatility_needs_both_halves():
    rng = np.random.default_rng(5)
    micro = rng.uniform(0.1, 0.5, 200)
    slippage = (3 * micro + rng.normal(0, 0.2, 200)).tolist()
    e = mon.explain(slippage, micro.tolist(), name="slippage_vs_micro_vol")
    assert e["explains"] and e["note"] == "association, not causation"
    half_only = np.concatenate([3 * micro[:100] + rng.normal(0, 0.2, 100), rng.normal(0, 1, 100)]).tolist()
    assert not mon.explain(half_only, micro.tolist(), name="x")["explains"]


# --------------------------------------------------------------------------- the model registry
def test_no_model_becomes_authoritative_on_in_sample_results():
    S = reg.ModelStage
    stellar_in_sample = reg.ModelEvidence(
        in_sample={"score": 0.99}, oos={"n": 10, "score": 0.1, "baseline": 0.2}
    )
    stage, why = reg.advance(S.CANDIDATE, stellar_in_sample, kind="ml")
    assert stage == S.CANDIDATE and why and not reg.uses_in_sample(why)
    good = reg.ModelEvidence(oos={"n": 200, "score": 0.6, "baseline": 0.5}, walk_forward={"passed": True},
                             stress={"passed": True}, shadow={"n": 40, "score": 0.62, "champion_score": 0.55})  # fmt: skip
    stage = S.CANDIDATE
    for expected in (
        S.OOS_VALIDATED,
        S.WALK_FORWARD_VALIDATED,
        S.STRESS_VALIDATED,
        S.PAPER_SHADOW,
        S.AUTHORITATIVE,
    ):
        stage, _ = reg.advance(stage, good, kind="ml")
        assert stage == expected  # one stage at a time
    ai_stage, why = reg.advance(S.PAPER_SHADOW, good, kind="ai")
    assert ai_stage == S.PAPER_SHADOW and any("person's approval" in w for w in why)
    weaker = reg.ModelEvidence(shadow={"n": 40, "score": 0.5, "champion_score": 0.55})
    assert reg.advance(S.PAPER_SHADOW, weaker, kind="ml")[0] == S.PAPER_SHADOW  # does not beat the champion
