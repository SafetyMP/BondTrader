"""
Advanced Analytics Module
Term structure modeling, credit spreads, scenario analysis, and relative value
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# Optional statsmodels for enhanced yield curve fitting
try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from bondtrader.core.bond_models import Bond
from bondtrader.core.bond_valuation import BondValuator


class AdvancedAnalytics:
    """Advanced bond analytics including term structure and scenario analysis"""

    def __init__(self, valuator: BondValuator = None):
        """Initialize advanced analytics"""
        self.valuator = valuator if valuator else BondValuator()

    def fit_yield_curve(self, bonds: List[Bond], method: str = "nelson_siegel", use_statsmodels: bool = False) -> Dict:
        """
        Fit yield curve to bond data

        Args:
            bonds: List of bonds to fit curve to
            method: 'nelson_siegel' or 'svensson'
            use_statsmodels: Use statsmodels for enhanced fitting (if available)

        Returns:
            Dictionary with curve parameters and fitted yields
        """
        # Extract maturities and yields
        maturities = []
        yields = []

        for bond in bonds:
            ttm = bond.time_to_maturity
            if ttm > 0:
                ytm = self.valuator.calculate_yield_to_maturity(bond)
                maturities.append(ttm)
                yields.append(ytm)

        if len(maturities) < 3:
            raise ValueError("Need at least 3 bonds to fit yield curve")

        maturities = np.array(maturities)
        yields = np.array(yields)

        if method == "nelson_siegel":
            return self._fit_nelson_siegel(maturities, yields)
        elif method == "svensson":
            return self._fit_svensson(maturities, yields)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fit_nelson_siegel(self, maturities: np.ndarray, yields: np.ndarray) -> Dict:
        """Fit Nelson-Siegel model to yield data"""

        def nelson_siegel(params, t):
            beta0, beta1, beta2, tau = params
            t_over_tau = t / tau if tau > 0 else 1e-10
            return (
                beta0
                + beta1 * ((1 - np.exp(-t_over_tau)) / t_over_tau)
                + beta2 * (((1 - np.exp(-t_over_tau)) / t_over_tau) - np.exp(-t_over_tau))
            )

        def objective(params):
            fitted = np.array([nelson_siegel(params, t) for t in maturities])
            return np.sum((yields - fitted) ** 2)

        # Initial guess
        initial = [np.mean(yields), -0.01, 0.01, 2.0]

        # Bounds
        bounds = [(-0.1, 0.2), (-0.1, 0.1), (-0.1, 0.1), (0.1, 10.0)]

        # Optimize
        result = minimize(objective, initial, bounds=bounds, method="L-BFGS-B")

        beta0, beta1, beta2, tau = result.x
        fitted_yields = np.array([nelson_siegel(result.x, t) for t in maturities])

        return {
            "method": "nelson_siegel",
            "parameters": {"beta0": beta0, "beta1": beta1, "beta2": beta2, "tau": tau},
            "fitted_yields": fitted_yields,
            "actual_yields": yields,
            "maturities": maturities,
            "rmse": np.sqrt(np.mean((yields - fitted_yields) ** 2)),
        }

    def _fit_svensson(self, maturities: np.ndarray, yields: np.ndarray) -> Dict:
        """
        Fit full Svensson model to yield data

        Svensson model: y(t) = β₀ + β₁((1-exp(-t/τ₁))/(t/τ₁)) +
                        β₂((1-exp(-t/τ₁))/(t/τ₁) - exp(-t/τ₁)) +
                        β₃((1-exp(-t/τ₂))/(t/τ₂) - exp(-t/τ₂))

        Adds second hump term for better fit
        """

        def svensson(params, t):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            t_over_tau1 = t / tau1 if tau1 > 0 else 1e-10
            t_over_tau2 = t / tau2 if tau2 > 0 else 1e-10

            term1 = (1 - np.exp(-t_over_tau1)) / t_over_tau1
            term2 = term1 - np.exp(-t_over_tau1)
            term3 = (1 - np.exp(-t_over_tau2)) / t_over_tau2 - np.exp(-t_over_tau2)

            return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

        def objective(params):
            fitted = np.array([svensson(params, t) for t in maturities])
            return np.sum((yields - fitted) ** 2)

        # Initial guess (extend Nelson-Siegel)
        initial = [np.mean(yields), -0.01, 0.01, 0.005, 2.0, 5.0]

        # Bounds
        bounds = [
            (-0.1, 0.2),  # beta0
            (-0.1, 0.1),  # beta1
            (-0.1, 0.1),  # beta2
            (-0.1, 0.1),  # beta3
            (0.1, 10.0),  # tau1
            (0.1, 20.0),  # tau2
        ]

        # Optimize
        result = minimize(objective, initial, bounds=bounds, method="L-BFGS-B")

        beta0, beta1, beta2, beta3, tau1, tau2 = result.x
        fitted_yields = np.array([svensson(result.x, t) for t in maturities])

        return {
            "method": "svensson",
            "parameters": {"beta0": beta0, "beta1": beta1, "beta2": beta2, "beta3": beta3, "tau1": tau1, "tau2": tau2},
            "fitted_yields": fitted_yields,
            "actual_yields": yields,
            "maturities": maturities,
            "rmse": np.sqrt(np.mean((yields - fitted_yields) ** 2)),
        }

    def calculate_z_spread(self, bond: Bond, market_price: Optional[float] = None) -> Dict:
        """
        Calculate Z-spread (zero volatility spread) for a bond

        Args:
            bond: Bond object
            market_price: Market price (uses bond.current_price if None)

        Returns:
            Dictionary with Z-spread and details
        """
        price = market_price if market_price is not None else bond.current_price
        time_to_maturity = bond.time_to_maturity

        if time_to_maturity <= 0:
            return {"z_spread": 0, "error": "Bond has matured"}

        # Get risk-free rates for different maturities (simplified)
        # In practice, would use treasury yield curve
        rf_rate = self.valuator.risk_free_rate

        # Calculate spread that makes PV = market price
        periods = int(time_to_maturity * bond.frequency)
        coupon_payment = (bond.coupon_rate / 100) * bond.face_value / bond.frequency

        def price_with_spread(spread):
            pv = 0
            for i in range(1, periods + 1):
                discount_rate = rf_rate + spread
                pv += coupon_payment / ((1 + discount_rate / bond.frequency) ** i)
            pv += bond.face_value / ((1 + discount_rate / bond.frequency) ** periods)
            return pv

        # Find spread using binary search
        spread_low = -0.1
        spread_high = 0.2

        for _ in range(50):  # Binary search
            spread_mid = (spread_low + spread_high) / 2
            pv_mid = price_with_spread(spread_mid)

            if abs(pv_mid - price) < 0.01:
                break

            if pv_mid > price:
                spread_low = spread_mid
            else:
                spread_high = spread_mid

        z_spread = spread_mid

        return {
            "z_spread": z_spread,
            "z_spread_bps": z_spread * 10000,  # Basis points
            "market_price": price,
            "risk_free_rate": rf_rate,
            "total_yield": rf_rate + z_spread,
        }

    def monte_carlo_scenario(
        self, bonds: List[Bond], num_scenarios: int = 1000, time_horizon: int = 252  # Trading days
    ) -> Dict:
        """
        Monte Carlo simulation of bond prices under various scenarios

        Args:
            bonds: List of bonds
            num_scenarios: Number of scenarios to simulate
            time_horizon: Time horizon in days

        Returns:
            Scenario analysis results
        """
        scenarios = []

        for _ in range(num_scenarios):
            scenario_values = {}
            total_value = 0

            for bond in bonds:
                # Simulate yield change
                yield_change = np.random.normal(0, 0.001) * np.sqrt(time_horizon)
                new_ytm = self.valuator.calculate_yield_to_maturity(bond) + yield_change

                # Calculate new price
                duration = self.valuator.calculate_duration(bond, new_ytm)
                convexity = self.valuator.calculate_convexity(bond, new_ytm)

                price_change = -duration * yield_change + 0.5 * convexity * (yield_change**2)
                new_price = bond.current_price * (1 + price_change)

                scenario_values[bond.bond_id] = new_price
                total_value += new_price

            scenarios.append({"bond_prices": scenario_values, "portfolio_value": total_value})

        # Aggregate statistics
        portfolio_values = [s["portfolio_value"] for s in scenarios]
        current_portfolio_value = sum(b.current_price for b in bonds)

        return {
            "num_scenarios": num_scenarios,
            "time_horizon": time_horizon,
            "current_portfolio_value": current_portfolio_value,
            "mean_portfolio_value": np.mean(portfolio_values),
            "std_portfolio_value": np.std(portfolio_values),
            "min_portfolio_value": np.min(portfolio_values),
            "max_portfolio_value": np.max(portfolio_values),
            "percentile_5": np.percentile(portfolio_values, 5),
            "percentile_95": np.percentile(portfolio_values, 95),
            "scenarios": scenarios[:10],  # Return first 10 for detail
        }

    def relative_value_analysis(self, bond: Bond, benchmark_bonds: List[Bond]) -> Dict:
        """
        Relative value analysis comparing bond to benchmarks

        Args:
            bond: Bond to analyze
            benchmark_bonds: List of similar bonds for comparison

        Returns:
            Relative value metrics
        """
        bond_ytm = self.valuator.calculate_yield_to_maturity(bond)
        bond_duration = self.valuator.calculate_duration(bond, bond_ytm)

        benchmark_ytms = [self.valuator.calculate_yield_to_maturity(b) for b in benchmark_bonds]
        benchmark_durations = [self.valuator.calculate_duration(b, ytm) for b, ytm in zip(benchmark_bonds, benchmark_ytms)]

        avg_benchmark_ytm = np.mean(benchmark_ytms)
        avg_benchmark_duration = np.mean(benchmark_durations)

        # Spread to benchmark
        spread_to_benchmark = bond_ytm - avg_benchmark_ytm

        # Duration-adjusted yield
        duration_adjusted_yield = (
            bond_ytm * (bond_duration / avg_benchmark_duration) if avg_benchmark_duration > 0 else bond_ytm
        )

        # Relative value score (higher is better)
        relative_value_score = spread_to_benchmark * 100  # In basis points

        return {
            "bond_ytm": bond_ytm * 100,
            "benchmark_avg_ytm": avg_benchmark_ytm * 100,
            "spread_to_benchmark_bps": spread_to_benchmark * 10000,
            "bond_duration": bond_duration,
            "benchmark_avg_duration": avg_benchmark_duration,
            "duration_adjusted_yield": duration_adjusted_yield * 100,
            "relative_value_score": relative_value_score,
            "num_benchmarks": len(benchmark_bonds),
        }
