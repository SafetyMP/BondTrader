"""Utility functions and helpers"""

from bondtrader.utils.utils import (
    ValidationError,
    cache_key,
    format_currency,
    format_date,
    format_percentage,
    handle_exceptions,
    logger,
    memoize,
    validate_bond_data,
)
from bondtrader.utils.validation import (
    validate_bond_input,
    validate_credit_rating,
    validate_file_path,
    validate_list_not_empty,
    validate_numeric_range,
)
from bondtrader.utils.validation import validate_percentage as validate_percentage_value
from bondtrader.utils.validation import (
    validate_positive,
    validate_probability,
    validate_weights_sum,
)

# Import new utilities (optional - only if available)
try:
    from bondtrader.utils.auth import (
        AuthenticationError,
        AuthorizationError,
        UserManager,
        get_user_manager,
        logout,
        require_auth,
        require_role,
    )

    _auth_available = True
except ImportError:
    _auth_available = False

try:
    from bondtrader.utils.rate_limiter import (
        RateLimiter,
        get_api_rate_limiter,
        get_dashboard_rate_limiter,
        rate_limit,
    )

    _rate_limiter_available = True
except ImportError:
    _rate_limiter_available = False

try:
    from bondtrader.utils.secrets import (
        SecretsManager,
        get_api_key,
        get_secrets_manager,
    )

    _secrets_available = True
except ImportError:
    _secrets_available = False

try:
    from bondtrader.utils.monitoring import (
        get_metrics,
        start_metrics_server,
        track_api_request,
        track_ml_prediction,
        track_risk_calculation,
        track_valuation,
    )

    _monitoring_available = True
except ImportError:
    _monitoring_available = False

__all__ = [
    "logger",
    "ValidationError",
    "validate_bond_data",
    "cache_key",
    "memoize",
    "handle_exceptions",
    "format_currency",
    "format_percentage",
    "format_date",
    # Validation functions
    "validate_bond_input",
    "validate_credit_rating",
    "validate_file_path",
    "validate_list_not_empty",
    "validate_numeric_range",
    "validate_percentage_value",
    "validate_positive",
    "validate_probability",
    "validate_weights_sum",
]

# Add new utilities to exports if available
if _auth_available:
    __all__.extend(
        [
            "AuthenticationError",
            "AuthorizationError",
            "UserManager",
            "get_user_manager",
            "require_auth",
            "require_role",
            "logout",
        ]
    )

if _rate_limiter_available:
    __all__.extend(
        [
            "RateLimiter",
            "get_api_rate_limiter",
            "get_dashboard_rate_limiter",
            "rate_limit",
        ]
    )

if _secrets_available:
    __all__.extend(
        [
            "SecretsManager",
            "get_secrets_manager",
            "get_api_key",
        ]
    )

if _monitoring_available:
    __all__.extend(
        [
            "start_metrics_server",
            "track_api_request",
            "track_valuation",
            "track_ml_prediction",
            "track_risk_calculation",
            "get_metrics",
        ]
    )
