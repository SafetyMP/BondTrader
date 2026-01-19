"""
Tests for business days utilities
"""

import pytest
from datetime import datetime, timedelta

from bondtrader.utils.business_days import (
    BusinessDayCalculator,
    add_business_days,
    calculate_business_days,
)


@pytest.mark.unit
class TestBusinessDayCalculator:
    """Test BusinessDayCalculator class"""

    def test_calculator_init(self):
        """Test calculator initialization"""
        calc = BusinessDayCalculator()
        assert calc is not None

    def test_is_business_day_weekday(self):
        """Test weekday is business day"""
        calc = BusinessDayCalculator()
        # Use a known weekday (not a holiday)
        date = datetime(2024, 1, 8)  # Monday (not a holiday)
        assert calc.is_business_day(date) is True

    def test_is_business_day_weekend(self):
        """Test weekend is not business day"""
        calc = BusinessDayCalculator()
        # Saturday
        date = datetime(2024, 1, 6)  # Saturday
        assert calc.is_business_day(date) is False

        # Sunday
        date = datetime(2024, 1, 7)  # Sunday
        assert calc.is_business_day(date) is False

    def test_count_business_days(self):
        """Test counting business days"""
        calc = BusinessDayCalculator()
        start = datetime(2024, 1, 8)  # Monday (not a holiday)
        end = datetime(2024, 1, 14)  # Sunday
        count = calc.count_business_days(start, end)
        # Monday to Sunday = 5 business days (Mon-Fri)
        assert count == 5

    def test_add_business_days(self):
        """Test adding business days"""
        calc = BusinessDayCalculator()
        start = datetime(2024, 1, 8)  # Monday (not a holiday)
        result = calc.add_business_days(start, 5)
        # 5 business days from Monday = next Monday
        assert result.weekday() < 5  # Should be a weekday

    def test_get_next_business_day(self):
        """Test getting next business day"""
        calc = BusinessDayCalculator()
        # Saturday (not a business day)
        date = datetime(2024, 1, 6)  # Saturday
        next_day = calc.get_next_business_day(date)
        # Next business day should be Monday
        assert next_day.weekday() == 0  # Monday
        assert calc.is_business_day(next_day)

        # Test from a business day - should return a business day
        date_fri = datetime(2024, 1, 5)  # Friday
        next_day_fri = calc.get_next_business_day(date_fri)
        # Should be a business day (weekday 0-4)
        assert next_day_fri.weekday() < 5
        assert calc.is_business_day(next_day_fri)

    def test_get_previous_business_day(self):
        """Test getting previous business day"""
        calc = BusinessDayCalculator()
        # Monday (not a holiday)
        date = datetime(2024, 1, 8)  # Monday
        prev_day = calc.get_previous_business_day(date)
        # Previous business day should be Friday
        assert prev_day.weekday() == 4  # Friday


@pytest.mark.unit
class TestBusinessDaysFunctions:
    """Test convenience functions"""

    def test_calculate_business_days(self):
        """Test calculate_business_days function"""
        start = datetime(2024, 1, 8)  # Monday (not a holiday)
        end = datetime(2024, 1, 14)  # Sunday
        count = calculate_business_days(start, end)
        assert count == 5

    def test_add_business_days_function(self):
        """Test add_business_days function"""
        start = datetime(2024, 1, 8)  # Monday (not a holiday)
        result = add_business_days(start, 5)
        assert result.weekday() < 5  # Should be a weekday
