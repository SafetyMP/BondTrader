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
    safe_divide,
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

# Import new Phase 2 utilities
try:
    from bondtrader.utils.mfa import MFAManager, get_mfa_manager

    _mfa_available = True
except ImportError:
    _mfa_available = False

try:
    from bondtrader.utils.rbac import (
        Permission,
        RBACManager,
        Role,
        get_rbac_manager,
    )

    _rbac_available = True
except ImportError:
    _rbac_available = False

try:
    from bondtrader.utils.api_keys import (
        APIKey,
        APIKeyManager,
        get_api_key_manager,
    )

    _api_keys_available = True
except ImportError:
    _api_keys_available = False

try:
    from bondtrader.utils.data_retention import (
        DataRetentionManager,
        RetentionPolicy,
        get_retention_manager,
    )

    _retention_available = True
except ImportError:
    _retention_available = False

try:
    from bondtrader.utils.redis_cache import (
        RedisCache,
        cache_result,
        get_redis_cache,
    )

    _redis_cache_available = True
except ImportError:
    _redis_cache_available = False

try:
    from bondtrader.utils.retry import (
        circuit_breaker,
        retry_with_backoff,
    )

    _retry_available = True
except ImportError:
    _retry_available = False

try:
    from bondtrader.utils.cache import (
        ModelCache,
        cache_model,
        clear_bond_cache,
        get_cached_bond,
    )

    _cache_available = True
except ImportError:
    _cache_available = False

# Import Phase 3 utilities
try:
    from bondtrader.utils.error_handling import (
        SanitizedException,
        handle_errors,
        sanitize_error_message,
        sanitize_exception,
    )
    from bondtrader.utils.graceful_degradation import (
        DegradationMode,
        GracefulDegradation,
        fallback_to_cached_data,
        fallback_to_simple_model,
        get_degradation_manager,
    )
    from bondtrader.utils.health import (
        ComponentHealth,
        HealthChecker,
        HealthStatus,
        get_health_checker,
    )
    from bondtrader.utils.performance_monitoring import (
        Alert,
        PerformanceMonitor,
        PerformanceThreshold,
        get_performance_monitor,
    )
    from bondtrader.utils.pool_monitoring import (
        PoolMonitor,
        get_pool_monitor,
    )

    _phase3_available = True
except ImportError:
    _phase3_available = False

__all__ = [
    "logger",
    "ValidationError",
    "cache_key",
    "memoize",
    "handle_exceptions",
    "format_currency",
    "format_percentage",
    "format_date",
    "safe_divide",
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

# Export Phase 2 utilities if available
if _mfa_available:
    __all__.extend(["MFAManager", "get_mfa_manager"])

if _rbac_available:
    __all__.extend(["Role", "Permission", "RBACManager", "get_rbac_manager"])

if _api_keys_available:
    __all__.extend(["APIKey", "APIKeyManager", "get_api_key_manager"])

if _retention_available:
    __all__.extend(["RetentionPolicy", "DataRetentionManager", "get_retention_manager"])

if _redis_cache_available:
    __all__.extend(["RedisCache", "get_redis_cache", "cache_result"])

if _retry_available:
    __all__.extend(["retry_with_backoff", "circuit_breaker"])

if _cache_available:
    __all__.extend(["ModelCache", "cache_model", "get_cached_bond", "clear_bond_cache"])

# Export Phase 3 utilities if available
if _phase3_available:
    __all__.extend(
        [
            "HealthStatus",
            "HealthChecker",
            "ComponentHealth",
            "get_health_checker",
            "DegradationMode",
            "GracefulDegradation",
            "get_degradation_manager",
            "fallback_to_simple_model",
            "fallback_to_cached_data",
            "sanitize_error_message",
            "sanitize_exception",
            "SanitizedException",
            "handle_errors",
            "PoolMonitor",
            "get_pool_monitor",
            "PerformanceMonitor",
            "PerformanceThreshold",
            "Alert",
            "get_performance_monitor",
        ]
    )

# Import Phase 4 utilities
try:
    from bondtrader.utils.alerting import (
        Alert,
        AlertChannel,
        AlertManager,
        AlertSeverity,
        get_alert_manager,
    )
    from bondtrader.utils.oauth import (
        OAuth2Manager,
        OAuth2Provider,
        get_oauth_manager,
    )
    from bondtrader.utils.tracing import (
        DistributedTracer,
        get_tracer,
        trace_span,
    )

    _phase4_available = True
except ImportError:
    _phase4_available = False

# Export Phase 4 utilities if available
if _phase4_available:
    __all__.extend(
        [
            "DistributedTracer",
            "get_tracer",
            "trace_span",
            "AlertSeverity",
            "AlertChannel",
            "AlertManager",
            "get_alert_manager",
            "OAuth2Manager",
            "OAuth2Provider",
            "get_oauth_manager",
        ]
    )
