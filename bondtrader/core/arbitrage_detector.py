"""
Arbitrage Opportunity Detection
Identifies pricing discrepancies and arbitrage opportunities
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from bondtrader.analytics.transaction_costs import TransactionCostCalculator
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.ml.ml_adjuster import MLBondAdjuster


class ArbitrageDetector:
    """Detects arbitrage opportunities in bond market"""

    def __init__(
        self,
        valuator: BondValuator = None,
        ml_adjuster: MLBondAdjuster = None,
        transaction_costs: TransactionCostCalculator = None,
        min_profit_threshold: float = 0.01,  # 1% minimum profit
        include_transaction_costs: bool = True,
    ):
        """
        Initialize arbitrage detector

        Args:
            valuator: Bond valuator instance
            ml_adjuster: ML adjuster instance (optional)
            transaction_costs: Transaction cost calculator (optional)
            min_profit_threshold: Minimum profit percentage to consider (as decimal)
            include_transaction_costs: Whether to account for transaction costs
        """
        self.valuator = valuator if valuator else BondValuator()
        self.ml_adjuster = ml_adjuster
        self.transaction_costs = transaction_costs if transaction_costs else TransactionCostCalculator()
        self.min_profit_threshold = min_profit_threshold
        self.include_transaction_costs = include_transaction_costs

    def find_arbitrage_opportunities(self, bonds: List[Bond], use_ml: bool = True) -> List[Dict]:
        """
        Find arbitrage opportunities in bond list

        Args:
            bonds: List of bonds to analyze
            use_ml: Whether to use ML-adjusted valuations

        Returns:
            List of arbitrage opportunity dictionaries
        """
        opportunities = []

        # OPTIMIZED: Pre-calculate YTM for all bonds in batch (leverages caching)
        # This allows us to reuse YTM for duration calculation and avoid redundant calculations
        # Only calculate YTM once per bond upfront
        bond_ytms = {bond.bond_id: self.valuator.calculate_yield_to_maturity(bond) for bond in bonds}

        for bond in bonds:
            # Get fair value
            if use_ml and self.ml_adjuster and self.ml_adjuster.is_trained:
                ml_result = self.ml_adjuster.predict_adjusted_value(bond)
                fair_value = ml_result["ml_adjusted_fair_value"]
                theoretical_fv = ml_result["theoretical_fair_value"]
            else:
                fair_value = self.valuator.calculate_fair_value(bond)
                theoretical_fv = fair_value

            # Calculate mismatch
            market_price = bond.current_price
            profit = fair_value - market_price
            profit_pct = (profit / market_price) * 100 if market_price > 0 else 0

            # Calculate net profit after transaction costs
            if self.include_transaction_costs:
                net_profit_data = self.transaction_costs.net_profit_after_costs(bond, fair_value, quantity=1.0)
                net_profit = net_profit_data["net_profit"]
                net_profit_pct = net_profit_data["net_profit_pct"]
                is_profitable_after_costs = net_profit_data["is_profitable"]
            else:
                net_profit = profit
                net_profit_pct = profit_pct
                is_profitable_after_costs = profit > 0

            # Only include if exceeds threshold and profitable after costs
            if abs(profit_pct) >= (self.min_profit_threshold * 100) and is_profitable_after_costs:
                # OPTIMIZED: Reuse pre-calculated YTM instead of recalculating
                ytm = bond_ytms[bond.bond_id]
                duration = self.valuator.calculate_duration(bond, ytm)

                opportunity = {
                    "bond_id": bond.bond_id,
                    "bond_type": bond.bond_type.value,
                    "issuer": bond.issuer,
                    "market_price": market_price,
                    "theoretical_fair_value": theoretical_fv,
                    "adjusted_fair_value": fair_value,
                    "gross_profit": profit,
                    "gross_profit_percentage": profit_pct,
                    "net_profit": net_profit if self.include_transaction_costs else profit,
                    "net_profit_percentage": net_profit_pct if self.include_transaction_costs else profit_pct,
                    "profit_opportunity": net_profit if self.include_transaction_costs else profit,  # For sorting
                    "profit_percentage": net_profit_pct if self.include_transaction_costs else profit_pct,  # For sorting
                    "transaction_costs": net_profit_data["round_trip_cost"] if self.include_transaction_costs else 0,
                    "transaction_costs_pct": net_profit_data["round_trip_cost_pct"] if self.include_transaction_costs else 0,
                    "ytm": ytm * 100,  # Convert to percentage
                    "duration": duration,
                    "maturity_date": bond.maturity_date.strftime("%Y-%m-%d"),
                    "credit_rating": bond.credit_rating,
                    "recommendation": "BUY" if profit > 0 else "SELL",
                    "arbitrage_type": self._classify_arbitrage_type(
                        bond, net_profit_pct if self.include_transaction_costs else profit_pct
                    ),
                }
                opportunities.append(opportunity)

        # Sort by absolute profit percentage (descending)
        opportunities.sort(key=lambda x: abs(x["profit_percentage"]), reverse=True)

        return opportunities

    def _classify_arbitrage_type(self, bond: Bond, profit_pct: float) -> str:
        """Classify type of arbitrage opportunity"""
        if abs(profit_pct) < 1:
            return "Minor Mispricing"
        elif abs(profit_pct) < 3:
            return "Moderate Arbitrage"
        elif abs(profit_pct) < 5:
            return "Significant Arbitrage"
        else:
            return "High-Arbitrage Opportunity"

    def compare_equivalent_bonds(self, bonds: List[Bond], grouping_key: str = "bond_type") -> List[Dict]:
        """
        Compare bonds with similar characteristics to find relative mispricing

        Args:
            bonds: List of bonds
            grouping_key: 'bond_type', 'credit_rating', or 'maturity_bucket'

        Returns:
            List of comparison results
        """
        # Group bonds
        groups = {}
        for bond in bonds:
            if grouping_key == "bond_type":
                key = bond.bond_type.value
            elif grouping_key == "credit_rating":
                key = bond.credit_rating
            elif grouping_key == "maturity_bucket":
                ttm = bond.time_to_maturity
                if ttm < 1:
                    key = "< 1 year"
                elif ttm < 5:
                    key = "1-5 years"
                elif ttm < 10:
                    key = "5-10 years"
                else:
                    key = "> 10 years"
            else:
                key = "all"

            if key not in groups:
                groups[key] = []
            groups[key].append(bond)

        comparisons = []
        for group_name, group_bonds in groups.items():
            if len(group_bonds) < 2:
                continue

            # OPTIMIZED: Calculate fair values once and reuse
            # Calculate fair value for each bond once
            fair_values = [self.valuator.calculate_fair_value(b) for b in group_bonds]
            avg_fair_value = np.mean(fair_values)

            # Create map for O(1) lookup instead of recalculating
            fair_value_map = {bond.bond_id: fv for bond, fv in zip(group_bonds, fair_values)}

            # Find most undervalued and overvalued (reuse calculated fair values)
            for bond in group_bonds:
                fair_value = fair_value_map[bond.bond_id]  # Reuse instead of recalculating
                market_price = bond.current_price

                rel_to_avg = ((market_price - fair_value) / fair_value) * 100
                rel_to_group = ((fair_value - avg_fair_value) / avg_fair_value) * 100

                comparison = {
                    "bond_id": bond.bond_id,
                    "group": group_name,
                    "market_price": market_price,
                    "fair_value": fair_value,
                    "group_avg_fair_value": avg_fair_value,
                    "relative_mispricing_pct": rel_to_avg,
                    "vs_group_avg_pct": rel_to_group,
                    "bond_type": bond.bond_type.value,
                }
                comparisons.append(comparison)

        return comparisons

    def calculate_portfolio_arbitrage(self, bonds: List[Bond], weights: List[float] = None) -> Dict:
        """
        Analyze arbitrage opportunities in a portfolio

        Args:
            bonds: List of bonds in portfolio
            weights: Portfolio weights (if None, equal weights assumed)

        Returns:
            Portfolio arbitrage analysis
        """
        if weights is None:
            weights = [1.0 / len(bonds)] * len(bonds)

        if len(weights) != len(bonds):
            raise ValueError("Weights must match bonds length")

        opportunities = self.find_arbitrage_opportunities(bonds, use_ml=False)

        total_market_value = sum(b.current_price * w for b, w in zip(bonds, weights))

        # OPTIMIZED: Reuse fair values from opportunities instead of recalculating
        # Create map from opportunities for O(1) lookup
        fair_value_map = {opp["bond_id"]: opp["adjusted_fair_value"] for opp in opportunities}

        # Calculate total fair value, reusing from opportunities where available
        # For bonds not in opportunities (threshold filtered), calculate once
        total_fair_value = 0
        for bond, weight in zip(bonds, weights):
            if bond.bond_id in fair_value_map:
                # Reuse fair value from opportunities (already calculated)
                total_fair_value += fair_value_map[bond.bond_id] * weight
            else:
                # Only calculate if bond was filtered out from opportunities
                # This should be rare (only bonds below threshold)
                total_fair_value += self.valuator.calculate_fair_value(bond) * weight

        portfolio_profit = total_fair_value - total_market_value
        portfolio_profit_pct = (portfolio_profit / total_market_value) * 100 if total_market_value > 0 else 0

        return {
            "total_market_value": total_market_value,
            "total_fair_value": total_fair_value,
            "portfolio_profit": portfolio_profit,
            "portfolio_profit_pct": portfolio_profit_pct,
            "num_opportunities": len(opportunities),
            "avg_opportunity_pct": np.mean([abs(o["profit_percentage"]) for o in opportunities]) if opportunities else 0,
        }
