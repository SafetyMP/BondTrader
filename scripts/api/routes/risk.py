"""
Risk Routes
Risk metrics and analysis
"""

from fastapi import APIRouter, Depends

from bondtrader.core.container import get_container
from scripts.api.middleware import verify_api_key
from scripts.api.models import RiskMetricsResponse
from scripts.api_helpers import handle_service_result

router = APIRouter(tags=["Risk Management"])


@router.get("/{bond_id}", response_model=RiskMetricsResponse, dependencies=[Depends(verify_api_key)])
async def get_risk_metrics(bond_id: str):
    """
    Get risk metrics for a bond

    Calculates comprehensive risk metrics including:
    - Value at Risk (VaR) using multiple methods (Historical, Parametric, Monte Carlo)
    - Credit risk analysis including default probability and expected loss

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - VaR values for different calculation methods
    - Credit risk metrics including default probability and recovery rates

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    - `500 Internal Server Error`: Risk calculation error

    **Note:** VaR calculations use a 95% confidence level and 1-day time horizon.
    """
    # Use service layer to calculate portfolio risk (single bond portfolio)
    container = get_container()
    bond_service = container.get_bond_service()
    result = bond_service.calculate_portfolio_risk([bond_id], weights=[1.0], confidence_level=0.95)
    risk_metrics = handle_service_result(result, "Risk calculation failed")

    return RiskMetricsResponse(
        bond_id=bond_id,
        var_historical=risk_metrics.get("var_historical"),
        var_parametric=risk_metrics.get("var_parametric"),
        var_monte_carlo=risk_metrics.get("var_monte_carlo"),
        credit_risk=risk_metrics.get("credit_risk", {}),
    )
