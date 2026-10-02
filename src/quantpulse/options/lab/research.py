"""The research library: where strategy ideas come from, how good the source is, and what testing found.

A source is a hypothesis generator, never an oracle. Each one is recorded with its author, date, type and
quality category (PRIMARY, ACADEMIC, REGULATORY, EXCHANGE, PROFESSIONAL_RESEARCH, SECONDARY, UNVERIFIED) and an
evidence grade on six axes — clarity, reproducibility, sample size, methodology, transparency, independent
validation — graded from how the evidence is presented, not from how famous the source is. Its claims enter as
UNVERIFIED_RESEARCH and move only on QuantPulse's own tests: "independent testing supports this under
conditions Y", or "independent testing does not reproduce the claimed edge".

The seed library below cites real, public work. The claims are paraphrased qualitatively (no numbers are
quoted from the sources); what a source reports and what QuantPulse finds are always kept apart. Sources
that need capabilities QuantPulse does not execute (delta hedging, index options, naked short volatility) are
kept with that limitation stated and a defined-risk *related* hypothesis — never silently transformed.

:func:`claim_test` tests a claim of the form "strategy X works when feature F is above/below T": the trades
where the condition held against those where it did not, the effect size, a bootstrap interval, a p-value,
and the same comparison inside each regime (so a regime difference is not mistaken for the claim).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

QUALITY = (
    "PRIMARY",
    "ACADEMIC",
    "REGULATORY",
    "EXCHANGE",
    "PROFESSIONAL_RESEARCH",
    "SECONDARY",
    "UNVERIFIED",
)
STATUSES = ("UNVERIFIED_RESEARCH", "EXTRACTED", "TESTING", "SUPPORTED", "NOT_REPRODUCED", "INCONCLUSIVE")
AXES = ("clarity", "reproducibility", "sample_size", "methodology", "transparency", "independent_validation")


@dataclass(frozen=True, slots=True)
class Source:
    key: str
    title: str
    author: str
    published: date | None
    source_type: str
    quality: str
    reference: str
    market: str
    period: str
    claim: str
    rules_text: str  # the strategy as the source describes it (paraphrased), fed to the extractor
    assumptions: tuple[str, ...] = ()
    limitations: str = ""
    grade: dict[str, float] = field(default_factory=dict)

    def evidence_score(self) -> float:
        """The mean of the six axes (0–1): how well the evidence is presented — not whether it is true."""
        vals = [self.grade.get(a, 0.0) for a in AXES]
        return round(sum(vals) / len(vals), 3)


def _g(
    clarity: float, repro: float, sample: float, method: float, transp: float, indep: float
) -> dict[str, float]:
    return dict(zip(AXES, (clarity, repro, sample, method, transp, indep), strict=True))


SEED: tuple[Source, ...] = (
    Source("whaley2002_bxm", "Return and Risk of CBOE Buy Write Monthly Index", "Robert E. Whaley", date(2002, 1, 1),
           "paper", "ACADEMIC", "Journal of Derivatives 10(2), 2002", "S&P 500 index options", "1988–2001",
           "A systematic monthly at-the-money covered call on the index delivered returns comparable to the index "
           "with lower volatility over the study period.",
           "Covered call: hold the shares and sell a 30 day at the money call every month; hold to expiration.",
           ("index options applied to a liquid index ETF",), "Index level, one period, costs modelled simply.",
           _g(0.9, 0.8, 0.6, 0.7, 0.8, 0.8)),
    Source("cboe_put_index", "Cboe S&P 500 PutWrite Index (PUT) methodology", "Cboe Global Markets", None,
           "index_methodology", "EXCHANGE", "Cboe index methodology (public)", "S&P 500 index options", "ongoing",
           "Selling one-month at-the-money index puts, fully collateralized, is a rules-based strategy with a "
           "published track record.",
           "Cash-secured put: sell a 30 day at the money put every month, fully collateralized; hold to expiration.",
           ("index options applied to a liquid index ETF",), "A benchmark index, not a claim of future returns.",
           _g(1.0, 0.9, 0.8, 0.6, 0.9, 0.7)),
    Source("coval_shumway2001", "Expected Option Returns", "Joshua D. Coval, Tyler Shumway", date(2001, 6, 1), "paper",
           "ACADEMIC", "Journal of Finance 56(3), 2001", "S&P 500 index options", "1986–1995",
           "Bought options earn returns below what their market risk alone would justify; bought puts in "
           "particular do poorly on average — buyers pay for protection.",
           "Long put: buy a 30 day at the money put every month; hold to 5 days before expiration.",
           ("tested as: long at-the-money puts lose money on average",), "Index options; a pricing study, not a trading rule.",
           _g(0.8, 0.7, 0.6, 0.9, 0.8, 0.8)),
    Source("bakshi_kapadia2003", "Delta-Hedged Gains and the Negative Market Volatility Risk Premium",
           "Gurdip Bakshi, Nikunj Kapadia", date(2003, 1, 1), "paper", "ACADEMIC", "Review of Financial Studies 16(2), 2003",
           "S&P 500 index options", "1988–1995",
           "Delta-hedged long option positions lose money on average: a negative volatility risk premium.",
           "Sell iron condor when implied volatility is high: short 20 delta put and call, wings 5% wide, 30-45 DTE, "
           "take profit at 50%, stop at 2x.",
           ("the paper studies delta-hedged options; QuantPulse does not delta-hedge — a related defined-risk "
            "short-volatility structure is tested instead",), "Requires delta hedging for the paper's exact test.",
           _g(0.8, 0.6, 0.6, 0.9, 0.7, 0.8)),
    Source("carr_wu2009", "Variance Risk Premiums", "Peter Carr, Liuren Wu", date(2009, 1, 1), "paper", "ACADEMIC",
           "Review of Financial Studies 22(3), 2009", "index and single-stock options", "1996–2003",
           "Implied variance tends to exceed the variance that is subsequently realized, for indices and many stocks.",
           "Sell put credit spread when IV/RV above 1.2: short 25 delta put, 5% wide, 30-45 DTE, take profit at 50%, "
           "stop at 2x.",
           ("tested directly as the sign of IV − later RV, and through a defined-risk short-volatility structure",),
           "Variance swaps are not traded here.", _g(0.8, 0.7, 0.7, 0.9, 0.8, 0.8)),
    Source("goyal_saretto2009", "Cross-section of option returns and volatility", "Amit Goyal, Alessio Saretto",
           date(2009, 11, 1), "paper", "ACADEMIC", "Journal of Financial Economics 94(2), 2009", "single-stock options",
           "1996–2006",
           "Options on stocks whose implied volatility is low relative to historical volatility subsequently do better "
           "than those where it is high.",
           "Buy long straddle when IV/RV below 0.9 at 30-45 DTE; take profit at 50%, stop at 50%.",
           ("straddles held about a month",), "Portfolio sorts; single-name liquidity varies.",
           _g(0.8, 0.7, 0.8, 0.8, 0.7, 0.6)),
    Source("israelov_nielsen2015", "Covered Calls Uncovered", "Roni Israelov, Lars N. Nielsen", date(2015, 11, 1),
           "paper", "PROFESSIONAL_RESEARCH", "Financial Analysts Journal 71(6), 2015", "S&P 500 index options",
           "1996–2014",
           "Covered call returns decompose into equity exposure, a short-volatility premium and an equity-timing "
           "component; the volatility premium is the part that is paid for.",
           "Covered call: sell a 30 delta call at 30-45 DTE against shares held; take profit at 50%.",
           ("tested through P&L attribution (delta vs vega/theta)",), "Decomposition study.",
           _g(0.8, 0.7, 0.7, 0.8, 0.8, 0.6)),
    Source("cboe_cndr", "Cboe S&P 500 Iron Condor Index (CNDR) methodology", "Cboe Global Markets", None,
           "index_methodology", "EXCHANGE", "Cboe index methodology (public)", "S&P 500 index options", "ongoing",
           "A rules-based monthly short iron condor with defined risk.",
           "Iron condor: short 20 delta put and call, wings 5% wide, 30-45 DTE, hold to 7 days before expiration.",
           ("index options applied to a liquid index ETF",), "A benchmark index, not a claim of future returns.",
           _g(1.0, 0.9, 0.8, 0.6, 0.9, 0.6)),
    Source("jegadeesh_titman1993", "Returns to Buying Winners and Selling Losers", "Narasimhan Jegadeesh, Sheridan Titman",
           date(1993, 3, 1), "paper", "ACADEMIC", "Journal of Finance 48(1), 1993", "US stocks", "1965–1989",
           "Stocks with high returns over the past months tend to keep outperforming over the following months.",
           "Buy bull call spread on momentum: long 50 delta call, 5% wide, 30-60 DTE, take profit at 80%, stop at 50%.",
           ("the paper holds stocks; the option structure is QuantPulse's own choice",), "A stock anomaly, not an option result.",
           _g(0.9, 0.9, 0.9, 0.9, 0.8, 0.9)),
    Source("dubinsky2019_earnings", "Option Pricing of Earnings Announcement Risks",
           "Andrew Dubinsky, Michael Johannes, Andreas Kaeck, Norman J. Seeger", date(2019, 2, 1), "paper", "ACADEMIC",
           "Review of Financial Studies 32(2), 2019", "single-stock options", "1996–2015",
           "Option prices embed the risk of scheduled earnings announcements: implied volatility rises into them and "
           "falls afterwards.",
           "Sell iron condor before earnings: short 20 delta put and call, wings 5% wide, 7-21 DTE, close after 3 days.",
           ("the paper is about pricing; selling premium through earnings is a separate hypothesis",),
           "The event jump can exceed what was priced.", _g(0.8, 0.6, 0.8, 0.9, 0.7, 0.5)),
    Source("tasty_45dte_management", "45-DTE premium selling with 50% profit management (educational material)",
           "tastytrade (educational content)", None, "educational", "SECONDARY", "public educational videos and articles",
           "US equity and ETF options", "unspecified",
           "Selling premium around 45 days to expiration and closing winners at half the maximum profit improves "
           "risk-adjusted results.",
           "Sell iron condor: short 16 delta put and call, wings 5% wide, 40-50 DTE, take profit at 50%, close at 21 DTE, "
           "stop at 2x.",
           ("short strangles are undefined-risk and never executed: tested as an iron condor with explicit wings",),
           "Promotional context; methodology not published in full.", _g(0.7, 0.4, 0.3, 0.3, 0.3, 0.2)),
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class ClaimTest:
    n_with: int
    n_without: int
    mean_with: float | None
    mean_without: float | None
    effect: float | None
    ci: tuple[float, float] | None
    p_value: float | None
    within_regime: dict[str, float | None]
    verdict: str  # SUPPORTED | NOT_REPRODUCED | INCONCLUSIVE

    def as_dict(self) -> dict[str, Any]:
        return {"n_with": self.n_with, "n_without": self.n_without, "mean_with": self.mean_with,
                "mean_without": self.mean_without, "effect": self.effect,
                "ci": list(self.ci) if self.ci else None, "p_value": self.p_value,
                "within_regime": self.within_regime, "verdict": self.verdict}  # fmt: skip


def claim_test(trades: Sequence[dict[str, Any]], feature: str, threshold: float, *, above: bool = True,
               min_n: int = 10, seed: int = 1) -> ClaimTest:  # fmt: skip
    """Does the strategy do better when ``feature`` (a trade's entry feature) is above ``threshold``?"""
    from scipy.stats import ttest_ind

    def r(t: dict[str, Any]) -> float:
        return float(t["pnl"]) / float(t.get("max_loss") or 1.0)

    w: list[float] = []
    wo: list[float] = []
    by_regime: dict[str, tuple[list[float], list[float]]] = {}
    for t in trades:
        v = (t.get("features") or {}).get(feature)
        if v is None:
            continue
        cond = v > threshold if above else v < threshold
        (w if cond else wo).append(r(t))
        reg = by_regime.setdefault(str(t.get("regime")), ([], []))
        (reg[0] if cond else reg[1]).append(r(t))
    within = {k: (round(float(np.mean(a) - np.mean(b)), 4) if len(a) >= 3 and len(b) >= 3 else None)
              for k, (a, b) in by_regime.items()}  # fmt: skip
    if len(w) < min_n or len(wo) < min_n:
        return ClaimTest(len(w), len(wo), _m(w), _m(wo), None, None, None, within, "INCONCLUSIVE")
    effect = float(np.mean(w) - np.mean(wo))
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(w, len(w)).mean() - rng.choice(wo, len(wo)).mean()) for _ in range(1000)]
    ci = (round(float(np.percentile(boots, 5)), 4), round(float(np.percentile(boots, 95)), 4))
    p = float(ttest_ind(w, wo, equal_var=False, alternative="greater").pvalue)
    agree = [v for v in within.values() if v is not None]
    consistent = not agree or sum(1 for v in agree if v > 0) * 2 >= len(agree)
    if ci[0] > 0 and p < 0.05 and consistent:
        verdict = "SUPPORTED"
    elif ci[1] < 0 or (p > 0.5 and effect <= 0):
        verdict = "NOT_REPRODUCED"
    else:
        verdict = "INCONCLUSIVE"
    return ClaimTest(len(w), len(wo), _m(w), _m(wo), round(effect, 4), ci, round(p, 4), within, verdict)


def _m(x: list[float]) -> float | None:
    return round(float(np.mean(x)), 4) if x else None


def sign_test(values: Sequence[float], *, expected_positive: bool = True) -> dict[str, Any]:
    """Is the mean of ``values`` of the expected sign (e.g. IV minus later realized volatility > 0)?"""
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if len(x) < 10:
        return {"n": len(x), "verdict": "INCONCLUSIVE"}
    from scipy.stats import ttest_1samp

    alt = "greater" if expected_positive else "less"
    p = float(ttest_1samp(x, 0.0, alternative=alt).pvalue)
    verdict = "SUPPORTED" if p < 0.05 else "NOT_REPRODUCED" if p > 0.5 else "INCONCLUSIVE"
    return {"n": len(x), "mean": round(float(x.mean()), 5), "p_value": round(p, 5), "verdict": verdict}
