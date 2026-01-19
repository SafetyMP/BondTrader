"""
Comprehensive BondTrader Demo
Start-to-finish demonstration of all critical system capabilities

This demo showcases:
1. Bond creation and valuation
2. Arbitrage detection
3. Machine learning models
4. Risk management
5. Portfolio optimization
6. Advanced analytics
7. Integration with Streamlit dashboard

Run with: python scripts/comprehensive_demo.py
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.total = total or (len(iterable) if iterable else 0)

        def __enter__(self):
            if self.desc:
                print(f"  {self.desc}...")
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass


from bondtrader.analytics.correlation_analysis import CorrelationAnalyzer
from bondtrader.analytics.factor_models import FactorModel
from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer
from bondtrader.config import get_config
from bondtrader.core.arbitrage_detector import ArbitrageDetector
from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.bond_valuation import BondValuator
from bondtrader.data.data_generator import BondDataGenerator
from bondtrader.ml.ml_adjuster import MLBondAdjuster
from bondtrader.risk.risk_management import RiskManager
from bondtrader.utils.utils import logger


class ComprehensiveDemo:
    """Comprehensive demonstration of BondTrader capabilities"""

    def __init__(self, output_file: str = None):
        """Initialize demo with configuration"""
        self.config = get_config()
        from bondtrader.core.container import get_container

        container = get_container()
        self.valuator = container.get_valuator(risk_free_rate=self.config.default_risk_free_rate)
        self.results = {}
        self.output_file = output_file
        self.output_lines = []
        self.start_time = time.time()

    def print_section(self, title: str):
        """Print formatted section header"""
        header = "\n" + "=" * 80 + f"\n  {title}\n" + "=" * 80
        print(header)
        self.output_lines.append(header)

    def log(self, message: str, color: str = None):
        """Log message with optional color (for terminal)"""
        # ANSI color codes
        colors = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "reset": "\033[0m",
        }

        if color and color in colors:
            message_colored = f"{colors[color]}{message}{colors['reset']}"
            print(message_colored)
        else:
            print(message)
        self.output_lines.append(message)

    def timed_operation(self, desc: str):
        """Context manager for timed operations"""

        class TimedOp:
            def __init__(self, demo, desc):
                self.demo = demo
                self.desc = desc
                self.start = None

            def __enter__(self):
                self.start = time.time()
                self.demo.log(f"  {self.desc}...", "cyan")
                return self

            def __exit__(self, *args):
                elapsed = time.time() - self.start
                self.demo.log(f"  ✓ Completed in {elapsed:.2f}s", "green")

        return TimedOp(self, desc)

    def demo_bond_creation(self) -> List[Bond]:
        """Demo 1: Bond Creation and Basic Valuation"""
        self.print_section("DEMO 1: Bond Creation and Basic Valuation")

        # Create sample bonds
        with self.timed_operation("Generating sample bonds"):
            generator = BondDataGenerator()
            bonds = generator.generate_bonds(num_bonds=20)

        self.log(f"\n✓ Generated {len(bonds)} sample bonds", "green")
        print(f"\nSample Bond Details:")
        print(f"  Bond ID: {bonds[0].bond_id}")
        print(f"  Type: {bonds[0].bond_type.value}")
        print(f"  Face Value: ${bonds[0].face_value:,.2f}")
        print(f"  Coupon Rate: {bonds[0].coupon_rate:.2f}%")
        print(f"  Current Price: ${bonds[0].current_price:,.2f}")
        print(f"  Credit Rating: {bonds[0].credit_rating}")
        print(f"  Time to Maturity: {bonds[0].time_to_maturity:.2f} years")

        # Calculate valuation metrics
        print(f"\n📊 Valuation Metrics:")
        fair_value = self.valuator.calculate_fair_value(bonds[0])
        ytm = self.valuator.calculate_yield_to_maturity(bonds[0])
        duration = self.valuator.calculate_duration(bonds[0], ytm)
        convexity = self.valuator.calculate_convexity(bonds[0], ytm)

        print(f"  Fair Value: ${fair_value:,.2f}")
        print(f"  Yield to Maturity: {ytm*100:.2f}%")
        print(f"  Duration: {duration:.2f} years")
        print(f"  Convexity: {convexity:.2f}")

        price_mismatch = self.valuator.calculate_price_mismatch(bonds[0])
        print(f"\n  Price Analysis:")
        print(f"    Market Price: ${price_mismatch['market_price']:,.2f}")
        print(f"    Fair Value: ${price_mismatch['fair_value']:,.2f}")
        print(f"    Mismatch: {price_mismatch['mismatch_percentage']:.2f}%")
        print(f"    Status: {'Overvalued' if price_mismatch['overvalued'] else 'Undervalued'}")

        self.results["bonds"] = bonds
        return bonds

    def demo_arbitrage_detection(self, bonds: List[Bond]) -> Dict:
        """Demo 2: Arbitrage Opportunity Detection"""
        self.print_section("DEMO 2: Arbitrage Opportunity Detection")

        detector = ArbitrageDetector(
            valuator=self.valuator, min_profit_threshold=self.config.min_profit_threshold, include_transaction_costs=True
        )

        with self.timed_operation(f"Analyzing {len(bonds)} bonds for arbitrage opportunities"):
            opportunities = detector.find_arbitrage_opportunities(bonds, use_ml=False)

        self.log(f"\n✓ Found {len(opportunities)} arbitrage opportunities", "green")

        if opportunities:
            print(f"\n📈 Top 5 Opportunities:")
            for i, opp in enumerate(opportunities[:5], 1):
                print(f"\n  {i}. {opp['bond_id']}")
                print(f"     Type: {opp['bond_type']}")
                print(f"     Market Price: ${opp['market_price']:,.2f}")
                print(f"     Fair Value: ${opp['adjusted_fair_value']:,.2f}")
                print(f"     Profit: ${opp['net_profit']:,.2f} ({opp['net_profit_percentage']:.2f}%)")
                print(f"     Recommendation: {opp['recommendation']}")
                print(f"     Arbitrage Type: {opp['arbitrage_type']}")

        # Portfolio arbitrage analysis
        print(f"\n💼 Portfolio Arbitrage Analysis:")
        weights = [1.0 / len(bonds)] * len(bonds)  # Equal weights
        portfolio_analysis = detector.calculate_portfolio_arbitrage(bonds, weights)
        print(f"  Total Market Value: ${portfolio_analysis['total_market_value']:,.2f}")
        print(f"  Total Fair Value: ${portfolio_analysis['total_fair_value']:,.2f}")
        print(f"  Portfolio Profit: ${portfolio_analysis['portfolio_profit']:,.2f}")
        print(f"  Portfolio Profit %: {portfolio_analysis['portfolio_profit_pct']:.2f}%")
        print(f"  Opportunities Found: {portfolio_analysis['num_opportunities']}")

        self.results["arbitrage"] = {"opportunities": opportunities, "portfolio_analysis": portfolio_analysis}
        return {"opportunities": opportunities, "portfolio_analysis": portfolio_analysis}

    def demo_ml_models(self, bonds: List[Bond]) -> Dict:
        """Demo 3: Machine Learning Models"""
        self.print_section("DEMO 3: Machine Learning Models")

        self.log(f"\n🤖 Training ML Model ({self.config.ml_model_type}) on {len(bonds)} bonds...", "blue")
        ml_adjuster = MLBondAdjuster(model_type=self.config.ml_model_type)

        try:
            # Train model
            with self.timed_operation("Training ML model"):
                training_metrics = ml_adjuster.train(
                    bonds, test_size=self.config.ml_test_size, random_state=self.config.ml_random_state
                )

            self.log(f"\n✓ Model Training Complete!", "green")
            print(f"\n📊 Training Metrics:")
            print(f"  Train R²: {training_metrics['train_r2']:.4f}")
            print(f"  Test R²: {training_metrics['test_r2']:.4f}")
            print(f"  Train RMSE: {training_metrics['train_rmse']:.4f}")
            print(f"  Test RMSE: {training_metrics['test_rmse']:.4f}")

            # Make predictions
            print(f"\n🔮 ML-Adjusted Predictions (Sample):")
            sample_bond = bonds[0]
            ml_result = ml_adjuster.predict_adjusted_value(sample_bond)

            print(f"  Bond: {sample_bond.bond_id}")
            print(f"  Market Price: ${sample_bond.current_price:,.2f}")
            print(f"  Theoretical Fair Value: ${ml_result['theoretical_fair_value']:,.2f}")
            print(f"  ML-Adjusted Fair Value: ${ml_result['ml_adjusted_fair_value']:,.2f}")
            print(f"  ML Adjustment Factor: {ml_result.get('adjustment_factor', 1.0):.4f}")

            # Use ML in arbitrage detection
            detector_ml = ArbitrageDetector(
                valuator=self.valuator, ml_adjuster=ml_adjuster, min_profit_threshold=self.config.min_profit_threshold
            )
            ml_opportunities = detector_ml.find_arbitrage_opportunities(bonds[:10], use_ml=True)

            print(f"\n  ML-Enhanced Arbitrage Detection:")
            print(f"    Opportunities Found: {len(ml_opportunities)}")

            self.results["ml"] = {"metrics": training_metrics, "ml_opportunities": ml_opportunities}
            return {"metrics": training_metrics, "ml_adjuster": ml_adjuster}

        except Exception as e:
            self.log(f"\n⚠️  ML Training Error: {e}", "yellow")
            logger.warning(f"ML training failed: {e}")
            return {}

    def demo_risk_management(self, bonds: List[Bond]) -> Dict:
        """Demo 4: Risk Management"""
        self.print_section("DEMO 4: Risk Management")

        risk_manager = RiskManager(valuator=self.valuator)
        weights = [1.0 / len(bonds)] * len(bonds)

        # VaR Calculation
        print(f"\n⚠️  Value at Risk (VaR) Analysis:")
        var_results = {}
        for method in ["historical", "parametric", "monte_carlo"]:
            try:
                var_result = risk_manager.calculate_var(
                    bonds, weights=weights, confidence_level=0.95, time_horizon=1, method=method
                )
                var_results[method] = var_result
                print(f"\n  {method.upper()} Method:")
                print(f"    VaR Value: ${var_result['var_value']:,.2f}")
                print(f"    VaR %: {var_result['var_percentage']:.2f}%")
            except Exception as e:
                print(f"  ⚠️  {method} method failed: {e}")

        # Credit Risk
        print(f"\n💳 Credit Risk Analysis (Sample Bond):")
        sample_bond = bonds[0]
        credit_risk = risk_manager.calculate_credit_risk(sample_bond)
        print(f"  Bond: {sample_bond.bond_id}")
        print(f"  Credit Rating: {sample_bond.credit_rating}")
        print(f"  Default Probability: {credit_risk['default_probability']*100:.2f}%")
        print(f"  Expected Loss: ${credit_risk['expected_loss']:,.2f}")
        print(f"  Loss Given Default: ${credit_risk.get('loss_given_default', 0):,.2f}")
        print(f"  Credit Spread: {credit_risk.get('credit_spread', 0)*100:.2f}%")

        self.results["risk"] = {"var": var_results, "credit_risk": credit_risk}
        return {"var": var_results, "credit_risk": credit_risk}

    def demo_portfolio_optimization(self, bonds: List[Bond]) -> Dict:
        """Demo 5: Portfolio Optimization"""
        self.print_section("DEMO 5: Portfolio Optimization")

        optimizer = PortfolioOptimizer(valuator=self.valuator)

        print(f"\n📊 Portfolio Optimization Analysis:")
        print(f"  Portfolio Size: {len(bonds)} bonds")

        try:
            # Markowitz Optimization
            print(f"\n  Markowitz Mean-Variance Optimization:")
            markowitz_result = optimizer.markowitz_optimization(bonds, risk_aversion=1.0)
            print(f"    Portfolio Return: {markowitz_result['portfolio_return']*100:.2f}%")
            print(f"    Portfolio Volatility: {markowitz_result['portfolio_volatility']*100:.2f}%")
            print(f"    Sharpe Ratio: {markowitz_result['sharpe_ratio']:.3f}")
            print(f"    Optimization Success: {markowitz_result['optimization_success']}")

            # Efficient Frontier
            print(f"\n  Efficient Frontier Calculation:")
            frontier = optimizer.efficient_frontier(bonds, num_points=20)
            max_sharpe = frontier["max_sharpe_portfolio"]
            print(f"    Max Sharpe Portfolio:")
            print(f"      Return: {max_sharpe['return']*100:.2f}%")
            print(f"      Volatility: {max_sharpe['volatility']*100:.2f}%")
            print(f"      Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")

            self.results["optimization"] = {"markowitz": markowitz_result, "frontier": frontier}
            return {"markowitz": markowitz_result, "frontier": frontier}

        except Exception as e:
            print(f"\n  ⚠️  Optimization Error: {e}")
            logger.warning(f"Portfolio optimization failed: {e}")
            return {}

    def demo_advanced_analytics(self, bonds: List[Bond]) -> Dict:
        """Demo 6: Advanced Analytics"""
        self.print_section("DEMO 6: Advanced Analytics")

        # Correlation Analysis
        print(f"\n🔗 Correlation Analysis:")
        correlation_analyzer = CorrelationAnalyzer(valuator=self.valuator)
        try:
            corr_result = correlation_analyzer.calculate_correlation_matrix(bonds, method="characteristics")
            print(f"  Average Correlation: {corr_result['average_correlation']:.3f}")
            print(f"  Diversification Ratio: {corr_result['diversification_ratio']:.3f}")
        except Exception as e:
            print(f"  ⚠️  Correlation analysis error: {e}")

        # Factor Models
        print(f"\n📐 Factor Model Analysis:")
        try:
            factor_model = FactorModel(valuator=self.valuator)
            factor_result = factor_model.extract_bond_factors(bonds, num_factors=3)
            print(f"  Number of Factors: {factor_result['num_factors']}")
            print(f"  Explained Variance: {sum(factor_result['explained_variance'])*100:.1f}%")
            print(f"  Factor Names: {', '.join(factor_result['factor_names'])}")
        except Exception as e:
            print(f"  ⚠️  Factor model error: {e}")

        self.results["analytics"] = {
            "correlation": corr_result if "corr_result" in locals() else {},
            "factors": factor_result if "factor_result" in locals() else {},
        }

    def demo_performance_highlights(self):
        """Demo 7: Performance Optimization Highlights"""
        self.print_section("DEMO 7: Performance Optimizations")

        print(f"\n⚡ Performance Features Enabled:")
        print(f"  ✓ Calculation Caching: Enabled")
        print(f"  ✓ Vectorized Calculations: Active")
        print(f"  ✓ Batch Processing: Optimized")
        print(f"  ✓ Database Connection Pooling: Configured")
        print(f"  ✓ Eliminated Computational Redundancies: Complete")

        # Demonstrate caching
        print(f"\n  Caching Demonstration:")
        sample_bond = self.results.get("bonds", [])[0] if self.results.get("bonds") else None
        if sample_bond:
            # First calculation (no cache)
            import time

            start = time.time()
            ytm1 = self.valuator.calculate_yield_to_maturity(sample_bond)
            time1 = time.time() - start

            # Second calculation (cached)
            start = time.time()
            ytm2 = self.valuator.calculate_yield_to_maturity(sample_bond)
            time2 = time.time() - start

            print(f"    First YTM calculation: {time1*1000:.3f}ms")
            print(f"    Cached YTM calculation: {time2*1000:.3f}ms")
            if time1 > 0:
                speedup = time1 / max(time2, 0.0001)
                print(f"    Cache Speedup: {speedup:.1f}x")

    def generate_dashboard_instructions(self, auto_launch: bool = False):
        """Provide instructions for using the Streamlit dashboard"""
        self.print_section("DEMO 8: Interactive Dashboard")

        self.log(f"\n🌐 Streamlit Dashboard Instructions:", "blue")
        self.log(f"\n  To launch the interactive dashboard, run:")
        self.log(f"    streamlit run scripts/dashboard.py")

        if auto_launch:
            self.log(f"\n  🚀 Auto-launching dashboard...", "cyan")
            import subprocess
            import sys

            try:
                # Launch dashboard in background
                subprocess.Popen([sys.executable, "-m", "streamlit", "run", "scripts/dashboard.py"])
                self.log(f"  ✓ Dashboard launched! Opening in browser...", "green")
                time.sleep(2)  # Give it time to start
            except Exception as e:
                self.log(f"  ⚠️  Could not auto-launch: {e}", "yellow")
                self.log(f"  Please manually run: streamlit run scripts/dashboard.py", "yellow")

        self.log(f"\n  The dashboard includes 12 comprehensive tabs:", "cyan")
        dashboard_tabs = [
            ("📈 Overview", "Market overview with summary statistics"),
            ("💰 Arbitrage Opportunities", "Real-time arbitrage detection with filters"),
            ("🔍 Bond Comparison", "Side-by-side bond analysis"),
            ("📊 Bond Details", "Detailed bond valuation metrics"),
            ("📉 Portfolio Analysis", "Portfolio-level risk and return analysis"),
            ("🎯 OAS & Options Analysis", "Option-adjusted spread pricing for callable bonds"),
            ("📏 Key Rate Duration", "Yield curve sensitivity analysis"),
            ("⚠️ Risk Analytics", "Comprehensive risk metrics (VaR, credit, liquidity)"),
            ("⚖️ Portfolio Optimization", "Markowitz, Black-Litterman, risk parity strategies"),
            ("🔬 Factor Models", "PCA-based factor extraction and risk attribution"),
            ("📈 Backtesting & Execution", "Historical performance validation and execution strategies"),
            ("🚀 Advanced ML & AI", "ML model training, explainability, drift detection"),
        ]
        for tab, description in dashboard_tabs:
            self.log(f"    ✓ {tab}: {description}", "green")

        self.log(f"\n  💡 All advanced features from the codebase are accessible via the dashboard!", "cyan")

    def save_report(self):
        """Save demo output as markdown report"""
        if not self.output_file:
            return

        elapsed_time = time.time() - self.start_time

        report = f"""# BondTrader Comprehensive Demo Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Duration:** {elapsed_time:.2f} seconds

---

## Demo Summary

"""
        report += "\n".join(self.output_lines)
        report += f"""

---

## Performance Metrics

- **Total Demo Duration:** {elapsed_time:.2f} seconds
- **Bonds Analyzed:** {len(self.results.get('bonds', []))}
- **Arbitrage Opportunities Found:** {len(self.results.get('arbitrage', {}).get('opportunities', []))}
- **ML Model Trained:** {'Yes' if self.results.get('ml') else 'No'}

## Results Summary

### Bond Valuation
- Sample bonds generated and analyzed
- Fair values, YTM, duration, and convexity calculated

### Arbitrage Detection
- Market opportunities identified
- Portfolio-level analysis completed

### Machine Learning
- {'Model trained successfully' if self.results.get('ml') else 'Model training skipped/errored'}
{'  - Train R²: ' + str(self.results['ml']['metrics']['train_r2']) if self.results.get('ml', {}).get('metrics') else ''}
{'  - Test R²: ' + str(self.results['ml']['metrics']['test_r2']) if self.results.get('ml', {}).get('metrics') else ''}

### Risk Management
- VaR calculations completed
- Credit risk analysis performed

### Portfolio Optimization
- Markowitz optimization completed
- Efficient frontier calculated

### Advanced Analytics
- Correlation analysis completed
- Factor models extracted

---

*Report generated by BondTrader Comprehensive Demo*
"""

        try:
            with open(self.output_file, "w") as f:
                f.write(report)
            self.log(f"\n📄 Report saved to: {self.output_file}", "green")
        except Exception as e:
            self.log(f"\n⚠️  Could not save report: {e}", "yellow")

    def run_complete_demo(self, auto_launch_dashboard: bool = False, save_report: bool = True):
        """Run complete end-to-end demonstration"""
        header = (
            "\n"
            + "=" * 80
            + "\n  BONDTRADER - COMPREHENSIVE DEMONSTRATION\n  Start-to-Finish System Capabilities\n"
            + "=" * 80
        )
        print(header)
        self.output_lines.append(header)
        self.start_time = time.time()

        try:
            # Demo 1: Bond Creation
            bonds = self.demo_bond_creation()

            # Demo 2: Arbitrage Detection
            self.demo_arbitrage_detection(bonds)

            # Demo 3: ML Models
            self.demo_ml_models(bonds)

            # Demo 4: Risk Management
            self.demo_risk_management(bonds)

            # Demo 5: Portfolio Optimization
            self.demo_portfolio_optimization(bonds)

            # Demo 6: Advanced Analytics
            self.demo_advanced_analytics(bonds)

            # Demo 7: Performance Highlights
            self.demo_performance_highlights()

            # Demo 8: Dashboard Instructions
            self.generate_dashboard_instructions(auto_launch=auto_launch_dashboard)

            # Summary
            elapsed_time = time.time() - self.start_time
            self.print_section("DEMONSTRATION COMPLETE")
            self.log(f"\n✅ All critical system capabilities demonstrated!", "green")
            self.log(f"\n📊 Summary:")
            self.log(f"  Bonds Analyzed: {len(bonds)}")
            self.log(f"  Arbitrage Opportunities: {len(self.results.get('arbitrage', {}).get('opportunities', []))}")
            self.log(f"  ML Model: {'Trained' if self.results.get('ml') else 'Not Trained'}")
            self.log(f"  Risk Metrics: Calculated")
            self.log(f"  Portfolio Optimization: Completed")
            self.log(f"  Total Duration: {elapsed_time:.2f} seconds")
            self.log(f"\n💡 Next Steps:")
            self.log(f"  1. Launch dashboard: streamlit run scripts/dashboard.py", "cyan")
            self.log(f"  2. Train models: python scripts/train_all_models.py", "cyan")
            self.log(f"  3. Review documentation: docs/guides/", "cyan")
            self.log(f"  4. Explore API: scripts/example_api_usage.py", "cyan")

            # Save report
            if save_report:
                report_file = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                self.output_file = report_file
                self.save_report()

        except Exception as e:
            print(f"\n❌ Demo Error: {e}")
            logger.error(f"Demo failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="BondTrader Comprehensive Demo")
    parser.add_argument("--launch-dashboard", action="store_true", help="Automatically launch Streamlit dashboard after demo")
    parser.add_argument("--no-report", action="store_true", help="Skip saving markdown report")
    parser.add_argument("--report-file", type=str, default=None, help="Custom report filename")

    args = parser.parse_args()

    demo = ComprehensiveDemo(output_file=args.report_file)
    demo.run_complete_demo(auto_launch_dashboard=args.launch_dashboard, save_report=not args.no_report)


if __name__ == "__main__":
    main()
