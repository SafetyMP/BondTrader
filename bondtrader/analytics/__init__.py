"""Analytics and advanced financial modeling modules"""

# Lazy imports to avoid circular dependencies
__all__ = [
    "BacktestEngine",
    "PortfolioOptimizer",
    "FactorModel",
    "CorrelationAnalyzer",
    "OASPricer",
    "KeyRateDuration",
    "MultiCurveFramework",
    "FloatingRateBondPricer",
    "ExecutionStrategy",
    "AlternativeDataAnalyzer",
    "TransactionCostCalculator",
    "AdvancedAnalytics",
]


# Mapping of class names to their module paths for lazy imports
_IMPORT_MAP = {
    "BacktestEngine": ("bondtrader.analytics.backtesting", "BacktestEngine"),
    "PortfolioOptimizer": ("bondtrader.analytics.portfolio_optimization", "PortfolioOptimizer"),
    "FactorModel": ("bondtrader.analytics.factor_models", "FactorModel"),
    "CorrelationAnalyzer": ("bondtrader.analytics.correlation_analysis", "CorrelationAnalyzer"),
    "OASPricer": ("bondtrader.analytics.oas_pricing", "OASPricer"),
    "KeyRateDuration": ("bondtrader.analytics.key_rate_duration", "KeyRateDuration"),
    "MultiCurveFramework": ("bondtrader.analytics.multi_curve", "MultiCurveFramework"),
    "FloatingRateBondPricer": ("bondtrader.analytics.floating_rate_bonds", "FloatingRateBondPricer"),
    "ExecutionStrategy": ("bondtrader.analytics.execution_strategies", "ExecutionStrategy"),
    "AlternativeDataAnalyzer": ("bondtrader.analytics.alternative_data", "AlternativeDataAnalyzer"),
    "TransactionCostCalculator": ("bondtrader.analytics.transaction_costs", "TransactionCostCalculator"),
    "AdvancedAnalytics": ("bondtrader.analytics.advanced_analytics", "AdvancedAnalytics"),
}


def __getattr__(name):
    """Lazy import for modules to avoid circular dependencies"""
    if name in _IMPORT_MAP:
        module_path, class_name = _IMPORT_MAP[name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
