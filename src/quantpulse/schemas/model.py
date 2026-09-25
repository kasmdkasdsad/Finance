"""The cross-sectional stock model, signal research and market regime."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import DataStatus, StrictModel

MODEL_DISCLAIMER = (
    "The model ranks stocks against each other; every statistic shown is out-of-sample (the model never saw "
    "the returns it is scored on). Small universes and a few years of history give noisy estimates, and past "
    "skill can disappear. Not investment advice."
)


class ICOut(StrictModel):
    mean_ic: float
    ic_std: float
    t_stat: float | None
    positive_share: float
    hit_rate: float | None
    n_dates: int


class PerfOut(StrictModel):
    total_return: float
    annual_return: float | None
    annual_volatility: float | None
    sharpe: float | None
    max_drawdown: float | None


class BacktestOut(StrictModel):
    dates: list[date]
    strategy: list[float]
    universe: list[float]
    benchmark: list[float]
    strategy_metrics: PerfOut
    universe_metrics: PerfOut
    benchmark_metrics: PerfOut
    beat_universe_share: float | None
    turnover: float = Field(description="Average share of the book replaced at each rebalance.")
    periods: int
    cost_bps: float


class CalibrationBinOut(StrictModel):
    z_mid: float
    n: int
    observed: float
    probability: float
    mean_excess: float


class ImportanceOut(StrictModel):
    feature: str
    description: str
    coefficient: float
    sign_consistency: float


class ICPoint(StrictModel):
    date: date
    ic: float
    rolling: float | None
    baseline_rolling: float | None


class LiveScore(StrictModel):
    symbol: str
    rank: int
    score: float
    z: float
    rating: int = Field(ge=1, le=10)
    prob_outperform: float = Field(
        description="Calibrated probability of beating the benchmark over the horizon."
    )
    expected_excess_return: float


class ModelReport(StrictModel):
    as_of: date
    computed_at: AwareDatetime
    horizon: int
    benchmark: str
    symbols: list[str]
    skipped: dict[str, str] = Field(default_factory=dict)
    oos_start: date
    oos_end: date
    oos: ICOut
    baseline: ICOut
    verdict: str
    has_skill: bool
    ic_timeline: list[ICPoint]
    buckets: list[float | None]
    backtest: BacktestOut
    base_rate: float
    calibration: list[CalibrationBinOut]
    importance: list[ImportanceOut]
    live: list[LiveScore]
    retrains: int
    lambdas: list[float]
    data_status: DataStatus
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = MODEL_DISCLAIMER


class HorizonICOut(StrictModel):
    horizon: int
    mean_ic: float | None
    t_stat: float | None
    positive_share: float | None
    n_dates: int


class SignalOut(StrictModel):
    feature: str
    description: str
    by_horizon: list[HorizonICOut]
    quintile_returns: list[float | None]
    spread: float | None


class ResearchReport(StrictModel):
    computed_at: AwareDatetime
    start: date
    end: date
    horizons: list[int]
    main_horizon: int
    n_symbols: int
    signals: list[SignalOut]
    correlation: dict[str, dict[str, float | None]]
    data_status: DataStatus
    note: str = (
        "IC = rank correlation between a signal and later returns across stocks. |t| > 2 is conventionally "
        "significant, but testing many signals at once means some will pass by luck."
    )


class ConditionalOut(StrictModel):
    state: str
    n: int
    mean: float | None
    median: float | None
    positive_share: float | None


class RegimeOut(StrictModel):
    label: str
    benchmark: str
    as_of: date
    price: float
    above_sma200: bool
    sma50_above_sma200: bool
    distance_sma200: float
    return_3m: float
    drawdown_52w: float
    volatility_21d: float
    volatility_percentile: float
    breadth_above_sma200: float | None
    breadth_above_sma50: float | None
    curve_slope_10y_3m: float | None
    curve_inverted: bool | None
    history: list[ConditionalOut]
    history_horizon_days: int = 21
    notes: list[str]
    data_status: DataStatus
