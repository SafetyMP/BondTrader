"""
Transaction Costs Module
Calculates trading costs, bid-ask spreads, and net profit for arbitrage
"""

from typing import Dict, List, Optional

import numpy as np

from bondtrader.core.bond_models import Bond


class TransactionCostCalculator:
    """Calculate transaction costs for bond trading"""

    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% commission
        bid_ask_spread: float = 0.002,  # 0.2% bid-ask spread
        slippage_rate: float = 0.0005,  # 0.05% slippage
        minimum_commission: float = 10.0,
    ):
        """
        Initialize transaction cost calculator

        Args:
            commission_rate: Commission rate as decimal (e.g., 0.001 for 0.1%)
            bid_ask_spread: Bid-ask spread as decimal
            slippage_rate: Slippage rate as decimal
            minimum_commission: Minimum commission in dollars
        """
        self.commission_rate = commission_rate
        self.bid_ask_spread = bid_ask_spread
        self.slippage_rate = slippage_rate
        self.minimum_commission = minimum_commission

    def calculate_trading_cost(
        self, bond: Bond, quantity: float = 1.0, is_buy: bool = True
    ) -> Dict:
        """
        Calculate total trading cost for a bond transaction

        Args:
            bond: Bond object
            quantity: Number of bonds (in face value units)
            is_buy: True for buy, False for sell

        Returns:
            Dictionary with cost breakdown
        """
        # Notional value
        notional_value = bond.current_price * quantity

        # Commission
        commission = max(notional_value * self.commission_rate, self.minimum_commission)

        # Bid-ask spread cost (paid on buy, received on sell but with penalty)
        spread_cost = notional_value * (self.bid_ask_spread / 2)
        if not is_buy:
            spread_cost = -spread_cost  # Receive on sell

        # Slippage
        slippage = notional_value * self.slippage_rate
        if not is_buy:
            slippage = -slippage  # Positive slippage on sell

        # Total cost
        total_cost = commission + abs(spread_cost) + abs(slippage)

        # Effective price (after costs)
        if is_buy:
            effective_price = bond.current_price + (total_cost / quantity)
        else:
            effective_price = bond.current_price - (total_cost / quantity)

        return {
            "notional_value": notional_value,
            "commission": commission,
            "bid_ask_spread_cost": abs(spread_cost),
            "slippage": abs(slippage),
            "total_cost": total_cost,
            "cost_percentage": (total_cost / notional_value) * 100,
            "effective_price": effective_price,
            "is_buy": is_buy,
        }

    def calculate_round_trip_cost(self, bond: Bond, quantity: float = 1.0) -> Dict:
        """
        Calculate round-trip trading cost (buy + sell)

        Args:
            bond: Bond object
            quantity: Number of bonds

        Returns:
            Dictionary with round-trip costs
        """
        buy_costs = self.calculate_trading_cost(bond, quantity, is_buy=True)
        sell_costs = self.calculate_trading_cost(bond, quantity, is_buy=False)

        total_cost = buy_costs["total_cost"] + sell_costs["total_cost"]
        notional_value = buy_costs["notional_value"]

        return {
            "buy_cost": buy_costs["total_cost"],
            "sell_cost": sell_costs["total_cost"],
            "total_round_trip_cost": total_cost,
            "round_trip_cost_pct": (total_cost / notional_value) * 100,
            "notional_value": notional_value,
        }

    def net_profit_after_costs(self, bond: Bond, fair_value: float, quantity: float = 1.0) -> Dict:
        """
        Calculate net profit after transaction costs

        Args:
            bond: Bond object
            fair_value: Theoretical fair value
            quantity: Number of bonds

        Returns:
            Dictionary with net profit metrics
        """
        # Gross profit
        gross_profit = (fair_value - bond.current_price) * quantity

        # Determine if buy or sell opportunity
        is_buy_opportunity = fair_value > bond.current_price

        # Calculate costs
        if is_buy_opportunity:
            costs = self.calculate_trading_cost(bond, quantity, is_buy=True)
        else:
            costs = self.calculate_trading_cost(bond, quantity, is_buy=False)

        # Round-trip cost if we need to exit later
        round_trip = self.calculate_round_trip_cost(bond, quantity)

        # Net profit (assuming we exit after realization)
        net_profit = gross_profit - round_trip["total_round_trip_cost"]

        # Net profit percentage
        notional = bond.current_price * quantity
        net_profit_pct = (net_profit / notional) * 100 if notional > 0 else 0

        return {
            "gross_profit": gross_profit,
            "gross_profit_pct": (gross_profit / notional) * 100 if notional > 0 else 0,
            "round_trip_cost": round_trip["total_round_trip_cost"],
            "round_trip_cost_pct": round_trip["round_trip_cost_pct"],
            "net_profit": net_profit,
            "net_profit_pct": net_profit_pct,
            "is_profitable": net_profit > 0,
            "breakeven_threshold": round_trip["round_trip_cost_pct"],
            "notional_value": notional,
        }

    def minimum_profit_threshold(self, bond: Bond, quantity: float = 1.0) -> float:
        """
        Calculate minimum profit threshold to cover transaction costs

        Args:
            bond: Bond object
            quantity: Number of bonds

        Returns:
            Minimum profit percentage required
        """
        round_trip = self.calculate_round_trip_cost(bond, quantity)
        return round_trip["round_trip_cost_pct"]
