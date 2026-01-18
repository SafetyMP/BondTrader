"""
Portfolio Optimization Module
Markowitz optimization, Black-Litterman, risk parity, and other strategies
Industry-standard portfolio construction methods
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize
from scipy.linalg import inv, pinv
from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.utils.utils import logger


class PortfolioOptimizer:
    """
    Portfolio optimization engine
    Implements Markowitz, Black-Litterman, risk parity, and other strategies
    """
    
    def __init__(self, valuator: BondValuator = None):
        """Initialize portfolio optimizer"""
        self.valuator = valuator if valuator else BondValuator()
    
    def calculate_returns_and_covariance(
        self,
        bonds: List[Bond],
        lookback_periods: int = 252,
        method: str = 'historical'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate expected returns and covariance matrix
        
        Args:
            bonds: List of bonds
            lookback_periods: Number of periods for historical data
            method: 'historical' or 'implied'
            
        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        n = len(bonds)
        
        if method == 'historical':
            # Simulate historical returns (in production, use actual data)
            returns = []
            for bond in bonds:
                # Simulate returns based on YTM and duration
                ytm = self.valuator.calculate_yield_to_maturity(bond)
                duration = self.valuator.calculate_duration(bond, ytm)
                
                # Generate synthetic returns
                bond_returns = np.random.normal(
                    ytm / 252,  # Daily return
                    abs(duration * 0.001),  # Volatility based on duration
                    lookback_periods
                )
                returns.append(bond_returns)
            
            returns = np.array(returns)
            expected_returns = np.mean(returns, axis=1) * 252  # Annualize
            covariance = np.cov(returns) * 252  # Annualize
        
        else:  # implied
            # Use bond characteristics to estimate returns and covariances
            expected_returns = np.array([
                self.valuator.calculate_yield_to_maturity(bond)
                for bond in bonds
            ])
            
            # Build covariance matrix based on correlations
            covariance = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Variance: based on duration
                        duration_i = self.valuator.calculate_duration(bonds[i])
                        covariance[i, j] = (duration_i * 0.01) ** 2
                    else:
                        # Covariance: correlation * sqrt(var_i * var_j)
                        duration_i = self.valuator.calculate_duration(bonds[i])
                        duration_j = self.valuator.calculate_duration(bonds[j])
                        
                        # Correlation based on credit rating similarity
                        rating_i = bonds[i].credit_rating
                        rating_j = bonds[j].credit_rating
                        correlation = 0.3 if rating_i == rating_j else 0.1
                        
                        std_i = duration_i * 0.01
                        std_j = duration_j * 0.01
                        covariance[i, j] = correlation * std_i * std_j
        
        return expected_returns, covariance
    
    def markowitz_optimization(
        self,
        bonds: List[Bond],
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Markowitz mean-variance optimization
        
        Maximizes: μ'w - λ * w'Σw
        Subject to: constraints
        
        Args:
            bonds: List of bonds
            target_return: Target portfolio return (if None, maximizes Sharpe)
            risk_aversion: Risk aversion parameter (λ)
            constraints: Additional constraints
            
        Returns:
            Optimal portfolio weights and metrics
        """
        n = len(bonds)
        expected_returns, covariance = self.calculate_returns_and_covariance(bonds)
        
        # Objective function: minimize negative utility
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })
        
        # Bounds: long-only (0 <= w <= 1)
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess: equal weights
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning("Optimization did not converge")
            weights = x0  # Fallback to equal weights
        else:
            weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'weights': weights.tolist(),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'optimization_success': result.success,
            'method': 'Markowitz'
        }
    
    def black_litterman_optimization(
        self,
        bonds: List[Bond],
        market_weights: Optional[np.ndarray] = None,
        views: Optional[List[Dict]] = None,
        tau: float = 0.05,
        risk_aversion: float = 3.0
    ) -> Dict:
        """
        Black-Litterman optimization
        
        Combines market equilibrium with investor views
        
        Args:
            bonds: List of bonds
            market_weights: Market capitalization weights (if None, equal weights)
            views: List of views [{'assets': [indices], 'return': value, 'confidence': value}]
            tau: Scaling factor
            risk_aversion: Risk aversion parameter
            
        Returns:
            Optimal portfolio weights
        """
        n = len(bonds)
        expected_returns, covariance = self.calculate_returns_and_covariance(bonds)
        
        # Market weights (if not provided, use equal weights)
        if market_weights is None:
            market_weights = np.ones(n) / n
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(covariance, market_weights)
        
        # Process views
        if views is None or len(views) == 0:
            # No views: use equilibrium returns
            bl_returns = pi
        else:
            # Build view matrices
            P = np.zeros((len(views), n))
            Q = np.zeros(len(views))
            Omega = np.zeros((len(views), len(views)))
            
            for i, view in enumerate(views):
                asset_indices = view.get('assets', [])
                view_return = view.get('return', 0)
                confidence = view.get('confidence', 1.0)
                
                # Normalize weights
                view_weights = np.array(view.get('weights', [1.0/len(asset_indices)] * len(asset_indices)))
                view_weights = view_weights / np.sum(view_weights)
                
                P[i, asset_indices] = view_weights
                Q[i] = view_return
                Omega[i, i] = 1.0 / confidence if confidence > 0 else 1.0
            
            # Black-Litterman formula
            tau_sigma = tau * covariance
            M1 = inv(tau_sigma)
            M2 = np.dot(P.T, np.dot(inv(Omega), P))
            M3 = np.dot(P.T, np.dot(inv(Omega), Q))
            
            bl_returns = np.dot(
                inv(M1 + M2),
                np.dot(M1, pi) + M3
            )
        
        # Optimize with BL returns
        def objective(weights):
            portfolio_return = np.dot(weights, bl_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x if result.success else x0
        
        portfolio_return = np.dot(weights, bl_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        return {
            'weights': weights.tolist(),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_std,
            'bl_returns': bl_returns.tolist(),
            'equilibrium_returns': pi.tolist(),
            'method': 'Black-Litterman'
        }
    
    def risk_parity_optimization(
        self,
        bonds: List[Bond],
        target_risk: Optional[float] = None
    ) -> Dict:
        """
        Risk parity optimization
        
        Equalizes risk contribution from each asset
        
        Args:
            bonds: List of bonds
            target_risk: Target portfolio volatility (if None, minimizes)
            
        Returns:
            Risk parity portfolio weights
        """
        n = len(bonds)
        _, covariance = self.calculate_returns_and_covariance(bonds)
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
            
            # Risk contributions
            marginal_contrib = np.dot(covariance, weights) / portfolio_vol if portfolio_vol > 0 else np.zeros(n)
            risk_contrib = weights * marginal_contrib
            
            # Minimize variance of risk contributions
            return np.var(risk_contrib)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x if result.success else x0
        
        # Calculate metrics
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        marginal_contrib = np.dot(covariance, weights) / portfolio_vol if portfolio_vol > 0 else np.zeros(n)
        risk_contrib = weights * marginal_contrib
        
        return {
            'weights': weights.tolist(),
            'portfolio_volatility': portfolio_vol,
            'risk_contributions': risk_contrib.tolist(),
            'risk_contribution_std': np.std(risk_contrib),
            'method': 'Risk Parity'
        }
    
    def efficient_frontier(
        self,
        bonds: List[Bond],
        num_points: int = 50
    ) -> Dict:
        """
        Calculate efficient frontier
        
        Args:
            bonds: List of bonds
            num_points: Number of points on frontier
            
        Returns:
            Efficient frontier data
        """
        expected_returns, covariance = self.calculate_returns_and_covariance(bonds)
        
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []
        
        for target_ret in target_returns:
            result = self.markowitz_optimization(bonds, target_return=target_ret)
            frontier_returns.append(result['portfolio_return'])
            frontier_volatilities.append(result['portfolio_volatility'])
            frontier_weights.append(result['weights'])
        
        # Find maximum Sharpe ratio portfolio
        sharpe_ratios = [r / v if v > 0 else 0 for r, v in zip(frontier_returns, frontier_volatilities)]
        max_sharpe_idx = np.argmax(sharpe_ratios)
        
        return {
            'returns': frontier_returns,
            'volatilities': frontier_volatilities,
            'weights': frontier_weights,
            'sharpe_ratios': sharpe_ratios,
            'max_sharpe_portfolio': {
                'return': frontier_returns[max_sharpe_idx],
                'volatility': frontier_volatilities[max_sharpe_idx],
                'sharpe_ratio': sharpe_ratios[max_sharpe_idx],
                'weights': frontier_weights[max_sharpe_idx]
            }
        }
