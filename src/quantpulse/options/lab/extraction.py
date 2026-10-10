"""StrategyExtraction: human-language strategy rules → an explicit, testable genome (or a refusal).

Deterministic pattern rules, no language model: the structure ("put credit spread", "iron condor", "covered
call", …), the entry ("momentum", "trend", "when implied volatility is high", "before earnings"), the filters
("IV rank above 50", "IV/RV above 1.2"), the days to expiration ("30-45 DTE", "45 days"), the strikes ("16
delta", "at the money", "5% wide"), and the exits ("take profit at 50%", "close at 21 DTE", "stop at 2x",
"close after 3 days").

Every parameter the text does not state is filled from a stated default *and recorded as an assumption* —
the difference between what the source said and what QuantPulse chose is never lost. Undefined-risk
structures (naked short options, short strangles) are refused as written; when a defined-risk form exists
it is proposed separately, with the substitution recorded. A text from which no structure or no entry can be
read is NEEDS_SPECIFICATION: no vague strategy enters the lab.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from quantpulse.options.lab.genome import Genome
from quantpulse.options.structures import FAMILIES

STRUCTURES: tuple[tuple[str, str, str], ...] = (
    # (pattern, family, direction)
    (r"iron condor", "iron_condor", "neutral"),
    (r"(put credit spread|bull put spread|short put spread)", "bull_put_spread", "bullish"),
    (r"(call credit spread|bear call spread|short call spread)", "bear_call_spread", "bearish"),
    (r"(call debit spread|bull call spread)", "bull_call_spread", "bullish"),
    (r"(put debit spread|bear put spread)", "bear_put_spread", "bearish"),
    (r"(cash[- ]secured put|put ?write)", "cash_secured_put", "income"),
    (r"covered call", "covered_call", "income"),
    (r"long straddle|buy (a |the )?straddle", "long_straddle", "volatility"),
    (r"long strangle|buy (a |the )?strangle", "long_strangle", "volatility"),
    (r"(butterfly)", "call_butterfly", "neutral"),
    (r"(long call|buy (a |an )?(\S+ )?call)", "long_call", "bullish"),
    (r"(long put|buy (a |an )?(\S+ )?put)", "long_put", "bearish"),
)
UNDEFINED = r"short strangle|sell (a |the )?strangle|naked (call|put)|short straddle|sell (a |the )?straddle"
SIGNALS: tuple[tuple[str, str], ...] = (
    (r"momentum|winners|past (months'? )?returns", "momentum_up"),
    (r"down ?trend|bearish trend", "trend_down"),
    (r"\btrend", "trend_up"),
    (r"breakout", "breakout_up"),
    (r"oversold|mean[- ]revert", "reversion_up"),
    (r"before earnings|into earnings|earnings", "pre_event"),
    (
        r"implied volatility is (unusually )?(high|rich|elevated)|when (iv|implied volatility) is high",
        "iv_high",
    ),
    (r"every month|each month|monthly|systematic|every week", "always"),
)


@dataclass
class Extraction:
    status: str  # EXTRACTED | NEEDS_SPECIFICATION | REFUSED_UNDEFINED_RISK
    genome: Genome | None
    stated: dict[str, Any] = field(default_factory=dict)  # values read from the text
    assumed: dict[str, Any] = field(default_factory=dict)  # values QuantPulse chose (with why)
    problems: list[str] = field(default_factory=list)
    substitution: str | None = None
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "genome": self.genome.canonical() if self.genome else None,
                "stated": self.stated, "assumed": self.assumed, "problems": self.problems,
                "substitution": self.substitution, "confidence": self.confidence}  # fmt: skip


def _num(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def extract(text: str, *, allow_substitution: bool = True) -> Extraction:
    t = " " + text.lower().replace("–", "-").replace("—", "-") + " "
    stated: dict[str, Any] = {}
    assumed: dict[str, Any] = {}
    substitution = None
    if re.search(UNDEFINED, t):
        if not allow_substitution:
            return Extraction("REFUSED_UNDEFINED_RISK", None, problems=["undefined-risk structure"])
        substitution = (
            "undefined-risk short strangle/straddle → iron condor with explicit wings (defined risk)"
        )
        t = re.sub(UNDEFINED, "iron condor", t)
    family = direction = None
    for pat, fam, direc in STRUCTURES:
        if re.search(pat, t):
            family, direction = fam, direc
            break
    if family is None:
        return Extraction(
            "NEEDS_SPECIFICATION", None, problems=["no option structure can be read from the text"]
        )
    stated["family"] = family
    signal = None
    for pat, sig in SIGNALS:
        if re.search(pat, t):
            signal = sig
            break
    iv_condition = re.search(r"iv rank|iv percentile|iv/rv", t) is not None
    if signal is None and iv_condition:  # the volatility filter is the entry condition itself
        signal, stated["entry_signal"] = "always", "the stated volatility filter"
    if signal is None:
        if direction in ("income", "neutral", "volatility"):
            signal, assumed["entry_signal"] = (
                "always",
                "no entry condition stated: entered whenever the filters allow",
            )
        else:
            return Extraction("NEEDS_SPECIFICATION", None, stated=stated,
                              problems=["a directional structure with no stated entry condition"])  # fmt: skip
    elif "entry_signal" not in stated:
        stated["entry_signal"] = signal
    g: dict[str, Any] = {"family": family, "direction": direction, "entry_signal": signal}
    rank = _num(r"iv rank (?:above|over|>=?|greater than) (\d+)", t)
    if rank is not None:
        g["iv_rank_min"] = stated["iv_rank_min"] = rank
    elif signal == "iv_high":
        g["iv_rank_min"], assumed["iv_rank_min"] = (
            50.0,
            "'high implied volatility' made explicit as IV rank ≥ 50",
        )
    pct = _num(r"iv percentile (?:above|over|>=?) (\d+)", t)
    if pct is not None:
        g["iv_percentile_min"] = stated["iv_percentile_min"] = pct
    ivrv_hi = _num(r"iv/rv (?:above|over|>) ?(\d+(?:\.\d+)?)", t)
    ivrv_lo = _num(r"iv/rv (?:below|under|<) ?(\d+(?:\.\d+)?)", t)
    if ivrv_hi is not None:
        g["iv_rv_min"] = stated["iv_rv_min"] = ivrv_hi
    if ivrv_lo is not None:
        g["iv_rv_max"] = stated["iv_rv_max"] = ivrv_lo
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(?:dte|days)", t)
    if m:
        g["dte_min"], g["dte_max"] = int(m.group(1)), int(m.group(2))
        stated["dte"] = [g["dte_min"], g["dte_max"]]
    else:
        one = _num(r"(\d+)[- ]?(?:day|dte)", t)
        if one is not None:
            g["dte_min"], g["dte_max"] = max(1, int(one) - 5), int(one) + 5
            stated["dte"], assumed["dte_window"] = (
                int(one),
                "a single DTE widened to ±5 days to find a listed expiration",
            )
        else:
            g["dte_min"], g["dte_max"] = 30, 45
            assumed["dte"] = "no expiration stated: 30-45 DTE"
    delta = _num(r"(\d+)[- ]?delta", t)
    if delta is not None:
        g["delta_target"] = stated["delta_target"] = delta / 100
    elif re.search(r"at[- ]the[- ]money|\batm\b", t):
        g["delta_target"] = stated["delta_target"] = 0.5
    else:
        default = 0.5 if family in ("long_call", "long_put", "bull_call_spread", "bear_put_spread") else 0.3
        g["delta_target"], assumed["delta_target"] = default, f"no strike stated: |delta| {default}"
    width = _num(r"(\d+(?:\.\d+)?)\s*% (?:wide|width|wings)", t)
    needs_width = family in ("bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread", "iron_condor",
                             "call_butterfly")  # fmt: skip
    if width is not None:
        g["width_pct"] = stated["width_pct"] = width / 100
    elif needs_width:
        g["width_pct"], assumed["width_pct"] = 0.05, "no spread width stated: 5% of the spot"
    if family in ("long_strangle",):
        g["wing_pct"], assumed["wing_pct"] = 0.05, "strangle legs 5% from the spot"
    if family == "iron_condor":
        g["wing_pct"] = g.get("wing_pct") or 0.05
    tp = _num(r"(?:take profit|manage (?:winners )?|close winners) at (\d+)%", t)
    if tp is not None:
        g["take_profit"] = stated["take_profit"] = tp / 100
    else:
        g["take_profit"], assumed["take_profit"] = 0.5, "no profit target stated: 50%"
    sl_x = _num(r"stop (?:loss )?at (\d+(?:\.\d+)?)x", t)
    sl_pct = _num(r"stop (?:loss )?at (\d+)%", t)
    if sl_x is not None:
        g["stop_loss"] = stated["stop_loss"] = sl_x
    elif sl_pct is not None:
        g["stop_loss"] = stated["stop_loss"] = sl_pct / 100
    else:
        credit = FAMILIES[family].vol == "short_vol"
        g["stop_loss"] = 2.0 if credit else 0.5
        assumed["stop_loss"] = (
            f"no stop stated: {g['stop_loss']} ({'multiple of the credit' if credit else 'share of the debit'})"
        )
    ex = _num(r"close at (\d+) dte", t)
    if ex is None:
        ex = _num(r"(\d+) days before expiration", t)
    if ex is not None:
        g["exit_dte"] = stated["exit_dte"] = int(ex)
    elif "hold to expiration" in t or "to expiration" in t:
        g["exit_dte"], assumed["exit_dte"] = (
            1,
            "'hold to expiration': closed 1 day before (never carried through)",
        )
    else:
        g["exit_dte"], assumed["exit_dte"] = (
            min(7, g["dte_min"] - 1),
            "no exit timing stated: 7 days before expiration",
        )
    if g["exit_dte"] >= g["dte_min"]:
        g["exit_dte"] = max(0, g["dte_min"] - 1)
    hold = _num(r"close after (\d+) days?", t)
    if hold is not None:
        g["max_hold_days"] = stated["max_hold_days"] = int(hold)
    else:
        g["max_hold_days"] = g["dte_max"]
    ev = "require" if signal == "pre_event" else "avoid"
    g["event_filter"] = ev
    if ev == "avoid":
        assumed["event_filter"] = (
            "earnings inside the option's life avoided unless the strategy is about them"
        )
    genome = Genome(**g)
    problems = genome.problems()
    n_stated = len(stated)
    confidence = round(n_stated / (n_stated + len(assumed)), 3) if (n_stated + len(assumed)) else 0.0
    status = "EXTRACTED" if not problems else "NEEDS_SPECIFICATION"
    return Extraction(
        status, genome if not problems else None, stated, assumed, problems, substitution, confidence
    )
