"""
Business Days Utilities
Handles business day calculations, market calendars, and day count conventions
"""

from datetime import datetime, timedelta
from typing import List, Optional

# Optional pandas-market-calendars for market calendars
try:
    import pandas_market_calendars as mcal
    HAS_MARKET_CALENDARS = True
except ImportError:
    HAS_MARKET_CALENDARS = False

from bondtrader.utils.utils import logger


class BusinessDayCalculator:
    """Business day calculator with market calendar support"""

    def __init__(self, calendar_name: str = "NYSE"):
        """
        Initialize business day calculator

        Args:
            calendar_name: Market calendar name ('NYSE', 'NASDAQ', 'CME', etc.)
                          Falls back to weekdays if pandas-market-calendars not available
        """
        self.calendar_name = calendar_name
        self.calendar = None

        if HAS_MARKET_CALENDARS:
            try:
                self.calendar = mcal.get_calendar(calendar_name)
            except Exception as e:
                logger.warning(f"Could not load calendar {calendar_name}: {e}. Using weekdays.")
                self.calendar = None

    def is_business_day(self, date: datetime) -> bool:
        """Check if date is a business day"""
        if self.calendar:
            # Use market calendar
            return self.calendar.valid_days(start_date=date.date(), end_date=date.date()).size > 0
        else:
            # Fallback: check if weekday (Monday=0, Sunday=6)
            return date.weekday() < 5  # Monday-Friday

    def add_business_days(self, start_date: datetime, days: int) -> datetime:
        """Add business days to a date"""
        if self.calendar:
            # Use market calendar
            valid_days = self.calendar.valid_days(start_date=start_date.date(), end_date=start_date.date() + timedelta(days=days * 2))
            if len(valid_days) >= days:
                return datetime.combine(valid_days[days - 1], datetime.min.time())
            # Fallback if not enough days
            current = start_date
            added = 0
            while added < days:
                current += timedelta(days=1)
                if self.is_business_day(current):
                    added += 1
            return current
        else:
            # Fallback: add weekdays only
            current = start_date
            added = 0
            while added < days:
                current += timedelta(days=1)
                if current.weekday() < 5:  # Monday-Friday
                    added += 1
            return current

    def count_business_days(self, start_date: datetime, end_date: datetime) -> int:
        """Count business days between two dates"""
        if self.calendar:
            # Use market calendar
            valid_days = self.calendar.valid_days(start_date=start_date.date(), end_date=end_date.date())
            return len(valid_days)
        else:
            # Fallback: count weekdays
            count = 0
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # Monday-Friday
                    count += 1
                current += timedelta(days=1)
            return count

    def get_next_business_day(self, date: datetime) -> datetime:
        """Get next business day"""
        return self.add_business_days(date, 1)

    def get_previous_business_day(self, date: datetime) -> datetime:
        """Get previous business day"""
        current = date
        while True:
            current -= timedelta(days=1)
            if self.is_business_day(current):
                return current


def calculate_business_days(start_date: datetime, end_date: datetime, calendar_name: str = "NYSE") -> int:
    """
    Convenience function to calculate business days between two dates

    Args:
        start_date: Start date
        end_date: End date
        calendar_name: Market calendar name

    Returns:
        Number of business days
    """
    calc = BusinessDayCalculator(calendar_name=calendar_name)
    return calc.count_business_days(start_date, end_date)


def add_business_days(date: datetime, days: int, calendar_name: str = "NYSE") -> datetime:
    """
    Convenience function to add business days

    Args:
        date: Start date
        days: Number of business days to add
        calendar_name: Market calendar name

    Returns:
        Date after adding business days
    """
    calc = BusinessDayCalculator(calendar_name=calendar_name)
    return calc.add_business_days(date, days)
