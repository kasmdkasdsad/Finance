"""The brain's agents. :func:`default_agents` lists the ones registered at start-up, in dependency order."""

from __future__ import annotations

from .base import Agent
from .data_quality import DataQualityAgent
from .momentum import MomentumAgent
from .portfolio import PortfolioAgent
from .regime import MarketRegimeAgent
from .technical import TechnicalAgent


def default_agents() -> list[Agent]:
    return [DataQualityAgent(), MarketRegimeAgent(), TechnicalAgent(), MomentumAgent(), PortfolioAgent()]


__all__ = [
    "Agent",
    "DataQualityAgent",
    "MarketRegimeAgent",
    "MomentumAgent",
    "PortfolioAgent",
    "TechnicalAgent",
    "default_agents",
]
