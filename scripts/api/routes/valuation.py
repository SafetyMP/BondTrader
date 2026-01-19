"""
Valuation Routes
Bond valuation and financial metrics
"""

from fastapi import APIRouter, Depends

from bondtrader.core.container import get_container
from scripts.api.middleware import verify_api_key
from scripts.api.models import ValuationResponse
from scripts.api_helpers import handle_service_result

router = APIRouter(tags=["Valuation"])


@router.get("/bonds/{bond_id}/valuation", response_model=ValuationResponse, dependencies=[Depends(verify_api_key)])
async def get_valuation(bond_id: str):
    """
    Get valuation metrics for a bond

    Calculates comprehensive valuation metrics including:
    - Fair value (theoretical price)
    - Yield to Maturity (YTM)
    - Duration (price sensitivity)
    - Convexity (second-order price sensitivity)
    - Market price mismatch percentage

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - Valuation metrics including fair value, YTM, duration, convexity, and mismatch

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    - `500 Internal Server Error`: Calculation error
    """
    # Use service layer (includes audit logging, metrics, error handling)
    container = get_container()
    bond_service = container.get_bond_service()
    result = bond_service.calculate_valuation(bond_id)
    valuation = handle_service_result(result, "Valuation calculation failed")

    return ValuationResponse(
        bond_id=valuation["bond_id"],
        fair_value=valuation["fair_value"],
        yield_to_maturity=valuation["ytm"],
        duration=valuation["duration"],
        convexity=valuation["convexity"],
        market_price=valuation["market_price"],
        mismatch_percentage=valuation["mismatch_percentage"],
    )
