"""
Enhanced Liquidity Risk Module
Comprehensive liquidity analysis: bid-ask spreads, market depth, LVaR
Industry-standard liquidity risk management
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class LiquidityRiskEnhanced:
    """
    Enhanced liquidity risk analysis
    Includes bid-ask spreads, market depth, and Liquidity-adjusted VaR (LVaR)
    """

    def __init__(self, valuator: BondValuator = None):
        """
        Initialize enhanced liquidity risk analyzer

        Args:
            valuator: Optional BondValuator instance. If None, gets from container.
        """
        if valuator is None:
            from bondtrader.core.container import get_container

            self.valuator = get_container().get_valuator()
        else:
            self.valuator = valuator

    def calculate_bid_ask_spread(self, bond: Bond, base_spread: Optional[float] = None) -> Dict:
        """
        Calculate bid-ask spread for a bond

        Spread depends on liquidity, credit quality, maturity, and market conditions

        Args:
            bond: Bond object
            base_spread: Base spread in basis points (if None, calculated from characteristics)

        Returns:
            Bid-ask spread analysis
        """
        if base_spread is None:
            # Calculate spread based on bond characteristics
            base_spread = self._estimate_spread_from_characteristics(bond)

        spread_bps = base_spread  # In basis points
        spread_decimal = spread_bps / 10000  # As decimal

        mid_price = bond.current_price
        bid_price = mid_price * (1 - spread_decimal / 2)
        ask_price = mid_price * (1 + spread_decimal / 2)

        # Spread as percentage of price
        spread_pct = (spread_decimal / mid_price) * 100 if mid_price > 0 else 0

        return {
            "bid_price": bid_price,
            "ask_price": ask_price,
            "mid_price": mid_price,
            "spread_bps": spread_bps,
            "spread_decimal": spread_decimal,
            "spread_pct": spread_pct,
            "spread_per_10000": spread_bps,  # Traditional units
        }

    def _estimate_spread_from_characteristics(self, bond: Bond) -> float:
        """
        Estimate bid-ask spread based on bond characteristics

        Factors:
        - Credit rating (lower rating = wider spread)
        - Maturity (longer maturity = wider spread)
        - Trading volume / liquidity proxy
        - Bond type
        """
        # Base spread by rating (in basis points)
        rating_spreads = {
            "AAA": 2,
            "AA+": 3,
            "AA": 4,
            "AA-": 5,
            "A+": 6,
            "A": 8,
            "A-": 10,
            "BBB+": 15,
            "BBB": 20,
            "BBB-": 25,
            "BB+": 35,
            "BB": 50,
            "BB-": 75,
            "B+": 100,
            "B": 150,
            "B-": 200,
            "CCC+": 300,
            "CCC": 400,
            "CCC-": 500,
            "D": 1000,
            "NR": 25,
        }

        base_spread = rating_spreads.get(bond.credit_rating.upper(), 25)

        # Adjust for maturity (longer = wider spread)
        time_to_maturity = bond.time_to_maturity
        if time_to_maturity > 10:
            maturity_factor = 1.0 + (time_to_maturity - 10) * 0.02  # 2% per year over 10
        elif time_to_maturity < 1:
            maturity_factor = 0.8  # Tighter spreads for short maturity
        else:
            maturity_factor = 1.0

        # Adjust for bond type
        if bond.bond_type.value == "Treasury":
            type_factor = 0.5  # Much tighter for Treasury
        elif bond.bond_type.value == "High Yield":
            type_factor = 1.5  # Wider for high yield
        else:
            type_factor = 1.0

        estimated_spread = base_spread * maturity_factor * type_factor
        return estimated_spread

    def estimate_market_depth(self, bond: Bond, trade_size: float = 100000) -> Dict:  # Trade size in face value
        """
        Estimate market depth and liquidity

        Market depth measures how much can be traded without significant price impact

        Args:
            bond: Bond object
            trade_size: Size of trade (in face value)

        Returns:
            Market depth analysis
        """
        # Simplified market depth model
        # In production, would use actual order book data

        # Depth factors
        base_depth = bond.face_value * 10  # Assume base depth is 10x face value

        # Adjust for characteristics
        rating = bond.credit_rating.upper()
        if rating in ["AAA", "AA", "A"]:
            depth_multiplier = 5.0  # High liquidity
        elif rating in ["BBB"]:
            depth_multiplier = 2.0  # Medium liquidity
        else:
            depth_multiplier = 0.5  # Low liquidity

        estimated_depth = base_depth * depth_multiplier

        # Price impact
        if trade_size <= estimated_depth:
            # Linear impact for small trades
            price_impact_pct = (trade_size / estimated_depth) * 0.001  # 0.1% for full depth
        else:
            # Exponential impact for large trades
            excess = trade_size - estimated_depth
            price_impact_pct = 0.001 + (excess / estimated_depth) * 0.005  # Additional impact

        # Time to liquidation
        if estimated_depth > trade_size:
            liquidation_time_days = 1  # Can trade in 1 day
        else:
            liquidation_time_days = 1 + (trade_size / estimated_depth)  # Proportional

        return {
            "estimated_depth": estimated_depth,
            "trade_size": trade_size,
            "depth_coverage": estimated_depth / trade_size if trade_size > 0 else 0,
            "price_impact_pct": price_impact_pct * 100,
            "liquidation_time_days": liquidation_time_days,
            "liquidity_rating": self._get_liquidity_rating(estimated_depth / trade_size if trade_size > 0 else 0),
        }

    def _get_liquidity_rating(self, depth_ratio: float) -> str:
        """Get liquidity rating based on depth ratio"""
        if depth_ratio >= 10:
            return "Excellent"
        elif depth_ratio >= 5:
            return "Good"
        elif depth_ratio >= 2:
            return "Fair"
        elif depth_ratio >= 1:
            return "Poor"
        else:
            return "Very Poor"

    def calculate_lvar(
        self,
        bonds: List[Bond],
        weights: Optional[List[float]] = None,
        confidence_level: float = 0.95,
        liquidation_horizon: float = 1.0,  # Days to liquidate
        method: str = "additive",
    ) -> Dict:
        """
        Calculate Liquidity-adjusted Value at Risk (LVaR)

        LVaR = VaR + Liquidity Cost
        Accounts for both market risk and liquidity risk

        Args:
            bonds: List of bonds in portfolio
            weights: Portfolio weights
            confidence_level: Confidence level
            liquidation_horizon: Time to liquidate position (days)
            method: 'additive' or 'multiplicative'

        Returns:
            LVaR metrics
        """
        from bondtrader.risk.risk_management import RiskManager

        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        risk_manager = RiskManager(self.valuator)

        # Calculate standard VaR
        var_result = risk_manager.calculate_var(
            bonds,
            weights,
            confidence_level,
            time_horizon=int(liquidation_horizon),
            method="monte_carlo",
        )

        var_value = var_result["var_value"]
        portfolio_value = sum(b.current_price * w * b.face_value for b, w in zip(bonds, weights))

        # Calculate liquidity cost
        liquidity_cost = 0
        liquidity_costs_by_bond = []

        for bond, weight in zip(bonds, weights):
            position_size = bond.face_value * weight

            # Get bid-ask spread
            spread_result = self.calculate_bid_ask_spread(bond)
            spread_cost = (spread_result["spread_decimal"] / 2) * position_size * bond.current_price

            # Get market depth impact
            depth_result = self.estimate_market_depth(bond, position_size)
            impact_cost = (depth_result["price_impact_pct"] / 100) * position_size * bond.current_price

            total_liquidity_cost = spread_cost + impact_cost
            liquidity_cost += total_liquidity_cost

            liquidity_costs_by_bond.append(
                {
                    "bond_id": bond.bond_id,
                    "position_size": position_size,
                    "spread_cost": spread_cost,
                    "impact_cost": impact_cost,
                    "total_liquidity_cost": total_liquidity_cost,
                }
            )

        if method == "additive":
            # LVaR = VaR + Liquidity Cost
            lvar_value = var_value + liquidity_cost
        else:  # multiplicative
            # LVaR = VaR * (1 + liquidity_cost / portfolio_value)
            liquidity_factor = 1 + (liquidity_cost / portfolio_value) if portfolio_value > 0 else 1
            lvar_value = var_value * liquidity_factor

        lvar_pct = (lvar_value / portfolio_value) * 100 if portfolio_value > 0 else 0
        liquidity_cost_pct = (liquidity_cost / portfolio_value) * 100 if portfolio_value > 0 else 0

        return {
            "lvar_value": lvar_value,
            "lvar_pct": lvar_pct,
            "var_value": var_value,
            "var_pct": (var_value / portfolio_value) * 100 if portfolio_value > 0 else 0,
            "liquidity_cost": liquidity_cost,
            "liquidity_cost_pct": liquidity_cost_pct,
            "liquidity_adjustment": liquidity_cost,
            "confidence_level": confidence_level,
            "liquidation_horizon": liquidation_horizon,
            "method": method,
            "portfolio_value": portfolio_value,
            "liquidity_costs_by_bond": liquidity_costs_by_bond,
        }

    def analyze_liquidity_risk(self, bonds: List[Bond], weights: Optional[List[float]] = None) -> Dict:
        """
        Comprehensive liquidity risk analysis

        Args:
            bonds: List of bonds
            weights: Portfolio weights

        Returns:
            Comprehensive liquidity risk metrics
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        portfolio_liquidity_metrics = []
        total_spread_cost = 0
        total_impact_cost = 0

        for bond, weight in zip(bonds, weights):
            position_size = bond.face_value * weight

            spread_result = self.calculate_bid_ask_spread(bond)
            depth_result = self.estimate_market_depth(bond, position_size)

            spread_cost = (spread_result["spread_decimal"] / 2) * position_size * bond.current_price
            impact_cost = (depth_result["price_impact_pct"] / 100) * position_size * bond.current_price

            total_spread_cost += spread_cost
            total_impact_cost += impact_cost

            portfolio_liquidity_metrics.append(
                {
                    "bond_id": bond.bond_id,
                    "rating": bond.credit_rating,
                    "spread_bps": spread_result["spread_bps"],
                    "depth_ratio": depth_result["depth_coverage"],
                    "liquidity_rating": depth_result["liquidity_rating"],
                    "spread_cost": spread_cost,
                    "impact_cost": impact_cost,
                    "total_cost": spread_cost + impact_cost,
                }
            )

        portfolio_value = sum(b.current_price * w * b.face_value for b, w in zip(bonds, weights))

        avg_spread = np.mean([m["spread_bps"] for m in portfolio_liquidity_metrics])
        avg_depth_ratio = np.mean([m["depth_ratio"] for m in portfolio_liquidity_metrics])

        return {
            "portfolio_value": portfolio_value,
            "total_spread_cost": total_spread_cost,
            "total_impact_cost": total_impact_cost,
            "total_liquidity_cost": total_spread_cost + total_impact_cost,
            "liquidity_cost_pct": (
                ((total_spread_cost + total_impact_cost) / portfolio_value) * 100 if portfolio_value > 0 else 0
            ),
            "avg_spread_bps": avg_spread,
            "avg_depth_ratio": avg_depth_ratio,
            "bond_metrics": portfolio_liquidity_metrics,
        }
