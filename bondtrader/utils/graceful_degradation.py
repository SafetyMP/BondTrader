"""
Graceful Degradation System
Provides fallback mechanisms when services fail

CRITICAL: Required for high availability in Fortune 10 financial institutions
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from bondtrader.utils.utils import logger

T = TypeVar("T")


class DegradationMode:
    """Degradation mode levels"""

    FULL = "full"  # All services available
    DEGRADED = "degraded"  # Some services unavailable, using fallbacks
    MINIMAL = "minimal"  # Only critical services available


class GracefulDegradation:
    """
    Graceful Degradation Manager

    Provides fallback mechanisms when services fail.
    """

    def __init__(self):
        """Initialize graceful degradation manager"""
        self.mode = DegradationMode.FULL
        self.service_status: Dict[str, bool] = {}

    def check_service(self, service_name: str) -> bool:
        """
        Check if a service is available.

        Args:
            service_name: Name of service

        Returns:
            True if service is available
        """
        return self.service_status.get(service_name, True)

    def mark_service_down(self, service_name: str):
        """Mark a service as down"""
        self.service_status[service_name] = False
        logger.warning(f"Service {service_name} marked as down. Entering degraded mode.")
        self._update_mode()

    def mark_service_up(self, service_name: str):
        """Mark a service as up"""
        self.service_status[service_name] = True
        logger.info(f"Service {service_name} marked as up.")
        self._update_mode()

    def _update_mode(self):
        """Update degradation mode based on service status"""
        all_up = all(self.service_status.values()) if self.service_status else True

        if all_up:
            self.mode = DegradationMode.FULL
        else:
            # Check if critical services are up
            critical_services = ["database", "valuation"]
            critical_up = all(self.service_status.get(s, True) for s in critical_services)

            if critical_up:
                self.mode = DegradationMode.DEGRADED
            else:
                self.mode = DegradationMode.MINIMAL

    def with_fallback(self, fallback_func: Optional[Callable] = None, fallback_value: Any = None):
        """
        Decorator to provide fallback when function fails.

        Args:
            fallback_func: Function to call as fallback
            fallback_value: Static value to return as fallback

        Usage:
            @degradation.with_fallback(fallback_value=default_value)
            def my_function():
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"{func.__name__} failed, using fallback: {e}")

                    if fallback_func:
                        try:
                            return fallback_func(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback function also failed: {fallback_error}")
                            if fallback_value is not None:
                                return fallback_value
                            raise
                    elif fallback_value is not None:
                        return fallback_value
                    else:
                        raise

            return wrapper

        return decorator


# Global degradation manager
_degradation_manager: Optional[GracefulDegradation] = None


def get_degradation_manager() -> GracefulDegradation:
    """Get or create global degradation manager instance"""
    global _degradation_manager
    if _degradation_manager is None:
        _degradation_manager = GracefulDegradation()
    return _degradation_manager


def fallback_to_simple_model(ml_model_func: Callable):
    """
    Decorator to fallback to simpler ML model if advanced model fails.

    Usage:
        @fallback_to_simple_model
        def predict_with_advanced_model(bond):
            ...
    """

    @wraps(ml_model_func)
    def wrapper(*args, **kwargs):
        try:
            return ml_model_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Advanced ML model failed, using simple model: {e}")

            # Fallback to simple DCF calculation
            bond = args[0] if args else None
            if bond:
                from bondtrader.core.bond_valuation import BondValuator

                valuator = BondValuator()
                fair_value = valuator.calculate_fair_value(bond)
                return {
                    "theoretical_fair_value": fair_value,
                    "ml_adjusted_fair_value": fair_value,
                    "adjustment_factor": 1.0,
                    "ml_confidence": 0.0,
                    "fallback_used": True,
                }
            raise

    return wrapper


def fallback_to_cached_data(get_func: Callable):
    """
    Decorator to fallback to cached data if database fails.

    Usage:
        @fallback_to_cached_data
        def get_bond_from_db(bond_id):
            ...
    """

    @wraps(get_func)
    def wrapper(*args, **kwargs):
        try:
            return get_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Database access failed, trying cache: {e}")

            # Try to get from cache
            try:
                from bondtrader.utils.redis_cache import get_redis_cache

                cache = get_redis_cache()
                bond_id = args[0] if args else kwargs.get("bond_id")
                if bond_id:
                    cached = cache.get(f"bond:{bond_id}")
                    if cached:
                        logger.info(f"Retrieved bond {bond_id} from cache")
                        return cached
            except Exception as cache_error:
                logger.error(f"Cache fallback also failed: {cache_error}")

            raise

    return wrapper
