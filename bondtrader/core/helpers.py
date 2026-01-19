"""
Helper Utilities for Common Patterns
Provides reusable utilities for common operations
"""

from typing import Dict, List, Optional

from bondtrader.core.bond_models import Bond
from bondtrader.core.container import get_container
from bondtrader.core.exceptions import DataNotFoundError
from bondtrader.core.result import Result


def get_bond_or_error(bond_id: str) -> Result[Bond, Exception]:
    """
    Helper to get bond with consistent error handling

    Args:
        bond_id: Bond identifier

    Returns:
        Result containing Bond or error
    """
    container = get_container()
    bond_service = container.get_bond_service()
    return bond_service.get_bond(bond_id)


def get_bonds_or_error(bond_ids: List[str]) -> Result[List[Bond], Exception]:
    """
    Helper to get multiple bonds with consistent error handling

    Args:
        bond_ids: List of bond identifiers

    Returns:
        Result containing list of Bonds or error
    """
    container = get_container()
    bond_service = container.get_bond_service()

    bonds = []
    errors = []

    for bond_id in bond_ids:
        result = bond_service.get_bond(bond_id)
        if result.is_err():
            errors.append(f"Bond {bond_id}: {result.error}")
        else:
            bonds.append(result.value)

    if errors:
        return Result.err(DataNotFoundError(f"Failed to retrieve bonds: {'; '.join(errors)}"))

    return Result.ok(bonds)


def validate_bond_data(data: Dict) -> Result[Dict, Exception]:
    """
    Validate bond data dictionary

    Args:
        data: Dictionary with bond data

    Returns:
        Result containing validated data or error
    """
    from bondtrader.core.exceptions import InvalidBondError

    required_fields = ["bond_id", "bond_type", "face_value", "coupon_rate", "maturity_date", "issue_date", "current_price"]

    # Check required fields
    missing = [field for field in required_fields if field not in data]
    if missing:
        return Result.err(InvalidBondError(f"Missing required fields: {', '.join(missing)}"))

    # Validate values
    if data["face_value"] <= 0:
        return Result.err(InvalidBondError("Face value must be positive"))

    if data["current_price"] <= 0:
        return Result.err(InvalidBondError("Current price must be positive"))

    if not 0 <= data["coupon_rate"] <= 1:
        return Result.err(InvalidBondError("Coupon rate must be between 0 and 1"))

    # Validate dates
    from datetime import datetime

    try:
        maturity = data["maturity_date"]
        if isinstance(maturity, str):
            maturity = datetime.fromisoformat(maturity.replace("Z", "+00:00"))

        issue = data["issue_date"]
        if isinstance(issue, str):
            issue = datetime.fromisoformat(issue.replace("Z", "+00:00"))

        if issue >= maturity:
            return Result.err(InvalidBondError("Issue date must be before maturity date"))
    except (ValueError, TypeError) as e:
        return Result.err(InvalidBondError(f"Invalid date format: {e}"))

    return Result.ok(data)


def calculate_portfolio_value(bonds: List[Bond], weights: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Calculate portfolio value metrics

    Args:
        bonds: List of bonds
        weights: Optional portfolio weights (defaults to equal weights)

    Returns:
        Dictionary with portfolio value metrics
    """
    if not bonds:
        return {
            "total_market_value": 0.0,
            "total_fair_value": 0.0,
            "num_bonds": 0,
        }

    # Default to equal weights
    if weights is None:
        weights = [1.0 / len(bonds)] * len(bonds)

    if len(weights) != len(bonds):
        raise ValueError("Weights length must match bonds length")

    container = get_container()
    bond_service = container.get_bond_service()

    # Calculate valuations
    valuations_result = bond_service.calculate_valuations_batch(bonds)
    if valuations_result.is_err():
        # Fallback to simple calculation
        total_market_value = sum(b.current_price for b in bonds)
        return {
            "total_market_value": total_market_value,
            "total_fair_value": total_market_value,  # Approximate
            "num_bonds": len(bonds),
        }

    valuations = valuations_result.value
    total_fair_value = sum(v["fair_value"] for v in valuations)
    total_market_value = sum(b.current_price for b in bonds)

    return {
        "total_market_value": total_market_value,
        "total_fair_value": total_fair_value,
        "mismatch_percentage": (
            ((total_market_value - total_fair_value) / total_fair_value * 100) if total_fair_value > 0 else 0
        ),
        "num_bonds": len(bonds),
    }


def format_valuation_result(valuation: Dict) -> str:
    """
    Format valuation result as human-readable string

    Args:
        valuation: Valuation dictionary

    Returns:
        Formatted string
    """
    return (
        f"Bond: {valuation.get('bond_id', 'N/A')}\n"
        f"Fair Value: ${valuation.get('fair_value', 0):.2f}\n"
        f"Market Price: ${valuation.get('market_price', 0):.2f}\n"
        f"YTM: {valuation.get('ytm', 0)*100:.2f}%\n"
        f"Duration: {valuation.get('duration', 0):.2f} years\n"
        f"Mismatch: {valuation.get('mismatch_percentage', 0):.2f}%"
    )


# safe_divide moved to bondtrader.utils.utils - import from there
