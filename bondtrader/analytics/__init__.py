"""Analytics and advanced financial modeling modules"""

# Lazy imports to avoid circular dependencies
__all__ = [
    'BacktestEngine',
    'PortfolioOptimizer',
    'FactorModel',
    'CorrelationAnalyzer',
    'OASPricer',
    'KeyRateDuration',
    'MultiCurveFramework',
    'FloatingRateBondPricer',
    'ExecutionStrategy',
    'AlternativeDataAnalyzer',
    'TransactionCostCalculator',
    'AdvancedAnalytics',
]

def __getattr__(name):
    """Lazy import for modules to avoid circular dependencies"""
    if name == 'BacktestEngine':
        from bondtrader.analytics.backtesting import BacktestEngine
        return BacktestEngine
    elif name == 'PortfolioOptimizer':
        from bondtrader.analytics.portfolio_optimization import PortfolioOptimizer
        return PortfolioOptimizer
    elif name == 'FactorModel':
        from bondtrader.analytics.factor_models import FactorModel
        return FactorModel
    elif name == 'CorrelationAnalyzer':
        from bondtrader.analytics.correlation_analysis import CorrelationAnalyzer
        return CorrelationAnalyzer
    elif name == 'OASPricer':
        from bondtrader.analytics.oas_pricing import OASPricer
        return OASPricer
    elif name == 'KeyRateDuration':
        from bondtrader.analytics.key_rate_duration import KeyRateDuration
        return KeyRateDuration
    elif name == 'MultiCurveFramework':
        from bondtrader.analytics.multi_curve import MultiCurveFramework
        return MultiCurveFramework
    elif name == 'FloatingRateBondPricer':
        from bondtrader.analytics.floating_rate_bonds import FloatingRateBondPricer
        return FloatingRateBondPricer
    elif name == 'ExecutionStrategy':
        from bondtrader.analytics.execution_strategies import ExecutionStrategy
        return ExecutionStrategy
    elif name == 'AlternativeDataAnalyzer':
        from bondtrader.analytics.alternative_data import AlternativeDataAnalyzer
        return AlternativeDataAnalyzer
    elif name == 'TransactionCostCalculator':
        from bondtrader.analytics.transaction_costs import TransactionCostCalculator
        return TransactionCostCalculator
    elif name == 'AdvancedAnalytics':
        from bondtrader.analytics.advanced_analytics import AdvancedAnalytics
        return AdvancedAnalytics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
