"""
Arbitrage Routes
Arbitrage opportunity detection
"""

from typing import List

from fastapi import APIRouter, Depends, Query

from bondtrader.core.container import get_container
from scripts.api.middleware import verify_api_key
from scripts.api.models import ArbitrageOpportunity
from scripts.api_helpers import handle_service_result

router = APIRouter(tags=["Arbitrage"])


@router.get(
    "/opportunities",
    response_model=List[ArbitrageOpportunity],
    dependencies=[Depends(verify_api_key)],
)
async def get_arbitrage_opportunities(
    min_profit_percentage: float = Query(
        0.0,
        ge=0,
        description="Minimum profit percentage threshold (as decimal, e.g., 0.01 for 1%)",
        example=0.01,
    ),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of opportunities to return", example=10),
):
    """
    Get arbitrage opportunities

    Identifies bonds that are mispriced relative to their theoretical fair value,
    presenting potential arbitrage opportunities.

    **Query Parameters:**
    - `min_profit_percentage`: Minimum profit percentage to include (default: 0.0)
    - `limit`: Maximum number of opportunities to return (default: 10, max: 100)

    **Returns:**
    - List of arbitrage opportunities sorted by profit percentage (descending)
    - Each opportunity includes bond ID, prices, profit, and recommendation

    **Example:**
    ```
    GET /arbitrage/opportunities?min_profit_percentage=0.02&limit=20
    ```
    """
    # Use service layer to find arbitrage opportunities
    container = get_container()
    bond_service = container.get_bond_service()
    result = bond_service.find_arbitrage_opportunities(
        filters=None, min_profit_percentage=min_profit_percentage, use_ml=False, limit=limit
    )
    opportunities = handle_service_result(result, "Failed to find arbitrage opportunities")

    return [
        ArbitrageOpportunity(
            bond_id=opp["bond_id"],
            market_price=opp["market_price"],
            fair_value=opp["fair_value"],
            profit=opp["profit"],
            profit_percentage=opp["profit_percentage"],
            recommendation=opp["recommendation"],
            arbitrage_type=opp.get("arbitrage_type", "Unknown"),
        )
        for opp in opportunities
    ]
