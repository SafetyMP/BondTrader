"""
Base Class for Analytics Modules
Provides common initialization pattern to eliminate duplication
"""

from typing import Optional

from bondtrader.core.bond_valuation import BondValuator


class AnalyticsBase:
    """Base class for analytics modules with shared initialization"""

    def __init__(self, valuator: Optional[BondValuator] = None):
        """
        Initialize analytics module with shared valuator pattern

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator
