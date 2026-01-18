"""
BondTrader - Comprehensive Bond Trading & Arbitrage Detection System
"""

__version__ = "1.0.0"

from bondtrader.config import Config, get_config, set_config
from bondtrader.core import ArbitrageDetector, Bond, BondType, BondValuator

__all__ = [
    "Bond",
    "BondType",
    "BondValuator",
    "ArbitrageDetector",
    "Config",
    "get_config",
    "set_config",
]
