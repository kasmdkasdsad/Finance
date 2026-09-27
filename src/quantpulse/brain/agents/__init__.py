"""The brain's agents. :func:`default_agents` lists the ones registered at start-up, in dependency order."""

from __future__ import annotations

from .base import Agent
from .catalyst import CatalystAgent
from .data_quality import DataQualityAgent
from .factor import FactorAgent
from .fundamental import FundamentalAgent, ValuationAgent
from .mean_reversion import MeanReversionAgent
from .momentum import MomentumAgent
from .options import OptionsAgent
from .portfolio import PortfolioAgent
from .regime import MarketRegimeAgent
from .statistical import StatisticalAgent
from .technical import TechnicalAgent
from .volatility import VolatilityAgent


def default_agents() -> list[Agent]:
    return [
        DataQualityAgent(),
        MarketRegimeAgent(),
        TechnicalAgent(),
        MomentumAgent(),
        MeanReversionAgent(),
        VolatilityAgent(),
        StatisticalAgent(),
        FundamentalAgent(),
        ValuationAgent(),
        FactorAgent(),
        OptionsAgent(),
        CatalystAgent(),
        PortfolioAgent(),
    ]


__all__ = [
    "Agent",
    "CatalystAgent",
    "DataQualityAgent",
    "FactorAgent",
    "FundamentalAgent",
    "MarketRegimeAgent",
    "MeanReversionAgent",
    "MomentumAgent",
    "OptionsAgent",
    "PortfolioAgent",
    "StatisticalAgent",
    "TechnicalAgent",
    "ValuationAgent",
    "VolatilityAgent",
    "default_agents",
]
