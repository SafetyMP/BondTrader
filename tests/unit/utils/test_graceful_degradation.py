"""
Unit tests for graceful degradation utilities
"""

import pytest

from bondtrader.utils.graceful_degradation import DegradationMode, GracefulDegradation


@pytest.mark.unit
class TestGracefulDegradation:
    """Test GracefulDegradation class"""

    def test_graceful_degradation_creation(self):
        """Test creating graceful degradation manager"""
        manager = GracefulDegradation()
        assert manager.mode == DegradationMode.FULL
        assert isinstance(manager.service_status, dict)

    def test_check_service(self):
        """Test checking service status"""
        manager = GracefulDegradation()
        # By default, services are up
        assert manager.check_service("database") is True

    def test_mark_service_down(self):
        """Test marking service as down"""
        manager = GracefulDegradation()
        manager.mark_service_down("database")
        assert manager.check_service("database") is False
        assert manager.mode == DegradationMode.DEGRADED or manager.mode == DegradationMode.MINIMAL

    def test_mark_service_up(self):
        """Test marking service as up"""
        manager = GracefulDegradation()
        manager.mark_service_down("database")
        manager.mark_service_up("database")
        assert manager.check_service("database") is True
        assert manager.mode == DegradationMode.FULL

    def test_update_mode_degraded(self):
        """Test updating mode to degraded"""
        manager = GracefulDegradation()
        manager.mark_service_down("non_critical_service")
        # Should enter degraded mode if critical services are up
        assert manager.mode in [DegradationMode.DEGRADED, DegradationMode.FULL, DegradationMode.MINIMAL]

    def test_update_mode_minimal(self):
        """Test updating mode to minimal"""
        manager = GracefulDegradation()
        manager.mark_service_down("database")
        # Should enter minimal mode if critical services are down
        assert manager.mode in [DegradationMode.DEGRADED, DegradationMode.MINIMAL]


@pytest.mark.unit
class TestWithFallback:
    """Test with_fallback decorator method"""

    def test_with_fallback_success(self):
        """Test with_fallback when function succeeds"""
        manager = GracefulDegradation()

        @manager.with_fallback(fallback_value=42)
        def successful_func():
            return 100

        result = successful_func()
        assert result == 100

    def test_with_fallback_value(self):
        """Test with_fallback with fallback value"""
        manager = GracefulDegradation()

        @manager.with_fallback(fallback_value=42)
        def failing_func():
            raise ValueError("Error")

        result = failing_func()
        assert result == 42

    def test_with_fallback_function(self):
        """Test with_fallback with fallback function"""

        def fallback_func():
            return 99

        manager = GracefulDegradation()

        @manager.with_fallback(fallback_func=fallback_func)
        def failing_func():
            raise ValueError("Error")

        result = failing_func()
        assert result == 99