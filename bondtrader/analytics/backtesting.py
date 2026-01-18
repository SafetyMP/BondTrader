"""
Backtesting Framework
Historical performance validation and strategy testing
Industry-standard backtesting capabilities
"""

from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class BacktestEngine:
    """
    Backtesting engine for bond trading strategies
    Validates strategies on historical data
    """

    def __init__(self, valuator: BondValuator = None):
        """Initialize backtesting engine"""
        self.valuator = valuator if valuator else BondValuator()

    def backtest_arbitrage_strategy(
        self, historical_bonds: List[List[Bond]], initial_capital: float = 1000000, transaction_costs: bool = True
    ) -> Dict:
        """
        Backtest arbitrage detection strategy

        Args:
            historical_bonds: List of bond lists for each time period
            initial_capital: Starting capital
            transaction_costs: Include transaction costs

        Returns:
            Backtest results with performance metrics
        """
        from bondtrader.analytics.transaction_costs import TransactionCostCalculator

        capital = initial_capital
        positions = {}  # {bond_id: quantity}
        trades = []
        portfolio_values = [initial_capital]

        detector = ArbitrageDetector(self.valuator)
        cost_calc = TransactionCostCalculator() if transaction_costs else None

        for period, bonds in enumerate(historical_bonds):
            # Find arbitrage opportunities
            opportunities = detector.find_arbitrage_opportunities(bonds, use_ml=False)

            # Close existing positions (simplified: sell all)
            for bond_id, quantity in list(positions.items()):
                bond = next((b for b in bonds if b.bond_id == bond_id), None)
                if bond:
                    if cost_calc:
                        sell_cost = cost_calc.calculate_trading_cost(bond, quantity, is_buy=False)
                        capital += bond.current_price * quantity - sell_cost["total_cost"]
                    else:
                        capital += bond.current_price * quantity

                    trades.append(
                        {
                            "period": period,
                            "bond_id": bond_id,
                            "action": "SELL",
                            "quantity": quantity,
                            "price": bond.current_price,
                            "capital": capital,
                        }
                    )

            positions = {}

            # Open new positions based on opportunities
            for opp in opportunities[:10]:  # Limit to top 10
                bond = next((b for b in bonds if b.bond_id == opp["bond_id"]), None)
                if not bond:
                    continue

                if opp["recommendation"] == "BUY" and capital > 0:
                    # Allocate 10% of capital to each opportunity
                    allocation = capital * 0.1
                    quantity = allocation / bond.current_price

                    if cost_calc:
                        buy_cost = cost_calc.calculate_trading_cost(bond, quantity, is_buy=True)
                        total_cost = bond.current_price * quantity + buy_cost["total_cost"]
                    else:
                        total_cost = bond.current_price * quantity

                    if total_cost <= capital:
                        capital -= total_cost
                        positions[bond.bond_id] = quantity

                        trades.append(
                            {
                                "period": period,
                                "bond_id": bond.bond_id,
                                "action": "BUY",
                                "quantity": quantity,
                                "price": bond.current_price,
                                "capital": capital,
                            }
                        )

            # Calculate portfolio value
            portfolio_value = capital
            for bond_id, quantity in positions.items():
                bond = next((b for b in bonds if b.bond_id == bond_id), None)
                if bond:
                    portfolio_value += bond.current_price * quantity

            portfolio_values.append(portfolio_value)

        # Calculate performance metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital

        # Sharpe ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualize
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = sum(
            1
            for t in trades
            if t["action"] == "SELL"
            and any(t2["action"] == "BUY" and t2["bond_id"] == t["bond_id"] for t2 in trades if t2["period"] < t["period"])
        )
        total_trades = len([t for t in trades if t["action"] == "SELL"])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "initial_capital": initial_capital,
            "final_capital": portfolio_values[-1],
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": (1 + total_return) ** (252 / len(historical_bonds)) - 1 if len(historical_bonds) > 0 else 0,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "num_trades": len(trades),
            "portfolio_values": portfolio_values,
            "returns": returns.tolist(),
            "trades": trades,
        }

    def calculate_performance_metrics(self, returns: np.ndarray) -> Dict:
        """
        Calculate comprehensive performance metrics

        Args:
            returns: Array of returns

        Returns:
            Performance metrics
        """
        if len(returns) == 0:
            return {}

        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Calmar ratio
        annual_return = mean_return * 252
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annual_return,
            "volatility": std_return * np.sqrt(252),
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "num_periods": len(returns),
        }
