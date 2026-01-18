"""
Financial constants and lookup tables
Shared constants used across multiple modules
"""

# Credit rating default probabilities (annual)
DEFAULT_PROBABILITIES = {
    "AAA": 0.0002,
    "AA+": 0.0005,
    "AA": 0.001,
    "AA-": 0.002,
    "A+": 0.003,
    "A": 0.005,
    "A-": 0.008,
    "BBB+": 0.012,
    "BBB": 0.020,
    "BBB-": 0.035,
    "BB+": 0.050,
    "BB": 0.080,
    "BB-": 0.120,
    "B+": 0.180,
    "B": 0.250,
    "B-": 0.350,
    "CCC+": 0.450,
    "CCC": 0.550,
    "CCC-": 0.650,
    "D": 1.000,
    "NR": 0.020,
}

# Credit rating recovery rates (standard)
# Used by basic risk management
RECOVERY_RATES_STANDARD = {
    "AAA": 0.60,
    "AA+": 0.58,
    "AA": 0.56,
    "AA-": 0.54,
    "A+": 0.52,
    "A": 0.50,
    "A-": 0.48,
    "BBB+": 0.46,
    "BBB": 0.44,
    "BBB-": 0.42,
    "BB+": 0.40,
    "BB": 0.38,
    "BB-": 0.36,
    "B+": 0.34,
    "B": 0.32,
    "B-": 0.30,
    "CCC+": 0.28,
    "CCC": 0.26,
    "CCC-": 0.24,
    "D": 0.20,
    "NR": 0.40,
}

# Credit rating recovery rates (enhanced)
# Used by enhanced credit risk models
RECOVERY_RATES_ENHANCED = {
    "AAA": 0.60,
    "AA+": 0.58,
    "AA": 0.58,
    "AA-": 0.56,
    "A+": 0.56,
    "A": 0.54,
    "A-": 0.52,
    "BBB+": 0.50,
    "BBB": 0.48,
    "BBB-": 0.46,
    "BB+": 0.44,
    "BB": 0.42,
    "BB-": 0.40,
    "B+": 0.38,
    "B": 0.36,
    "B-": 0.34,
    "CCC+": 0.32,
    "CCC": 0.30,
    "CCC-": 0.28,
    "D": 0.20,
    "NR": 0.40,
}


def get_default_probability(rating: str) -> float:
    """Get default probability based on credit rating (annual)"""
    return DEFAULT_PROBABILITIES.get(rating.upper(), 0.020)


def get_recovery_rate_standard(rating: str) -> float:
    """Get recovery rate based on credit rating (standard model)"""
    return RECOVERY_RATES_STANDARD.get(rating.upper(), 0.40)


def get_recovery_rate_enhanced(rating: str) -> float:
    """Get recovery rate based on credit rating (enhanced model)"""
    return RECOVERY_RATES_ENHANCED.get(rating.upper(), 0.40)
