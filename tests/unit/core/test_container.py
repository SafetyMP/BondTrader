"""
Unit tests for dependency injection container
"""

import pytest

from bondtrader.core.bond_valuation import BondValuator
from bondtrader.core.container import ServiceContainer, get_container


@pytest.mark.unit
class TestServiceContainer:
    """Test ServiceContainer functionality"""

    def test_container_singleton(self):
        """Test that container is singleton"""
        container1 = ServiceContainer()
        container2 = ServiceContainer()
        assert container1 is container2

    def test_get_container_function(self):
        """Test get_container function"""
        container1 = get_container()
        container2 = get_container()
        assert container1 is container2

    def test_get_valuator(self):
        """Test getting valuator"""
        container = get_container()
        valuator1 = container.get_valuator()
        valuator2 = container.get_valuator()
        assert valuator1 is valuator2
        assert isinstance(valuator1, BondValuator)

    def test_get_valuator_with_rfr(self):
        """Test getting valuator with custom risk-free rate"""
        container = get_container()
        valuator = container.get_valuator(risk_free_rate=0.04)
        assert valuator.risk_free_rate == 0.04

    def test_get_repository(self):
        """Test getting repository"""
        container = get_container()
        repo1 = container.get_repository()
        repo2 = container.get_repository()
        assert repo1 is repo2

    def test_get_bond_service(self):
        """Test getting bond service"""
        container = get_container()
        service1 = container.get_bond_service()
        service2 = container.get_bond_service()
        assert service1 is service2

    def test_get_arbitrage_detector(self):
        """Test getting arbitrage detector"""
        container = get_container()
        detector1 = container.get_arbitrage_detector()
        detector2 = container.get_arbitrage_detector()
        assert detector1 is detector2

    def test_get_risk_manager(self):
        """Test getting risk manager"""
        container = get_container()
        risk_manager1 = container.get_risk_manager()
        risk_manager2 = container.get_risk_manager()
        assert risk_manager1 is risk_manager2

    def test_config_property(self):
        """Test config property"""
        container = get_container()
        config1 = container.config
        config2 = container.config
        assert config1 is config2
