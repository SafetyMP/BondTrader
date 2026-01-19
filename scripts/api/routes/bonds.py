"""
Bond Routes
CRUD operations for bonds
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from bondtrader.core.bond_models import Bond, BondType
from bondtrader.core.container import get_container
from scripts.api.middleware import verify_api_key
from scripts.api.models import BondCreate, BondResponse
from scripts.api_helpers import handle_service_result

router = APIRouter(tags=["Bonds"])


@router.post(
    "", response_model=BondResponse, status_code=201, dependencies=[Depends(verify_api_key)]
)
async def create_bond(bond: BondCreate):
    """
    Create a new bond

    Creates a new bond record in the system. The bond will be stored in the database
    and can be used for valuation, arbitrage detection, and risk analysis.

    **Request Body:**
    - All fields are required except `credit_rating`, `issuer`, `callable`, and `convertible`
    - `maturity_date` and `issue_date` should be in ISO format (YYYY-MM-DD)
    - `coupon_rate` should be a decimal (e.g., 0.05 for 5%)

    **Returns:**
    - Created bond object with assigned ID

    **Errors:**
    - `400 Bad Request`: Invalid bond data or validation error
    - `500 Internal Server Error`: Database or processing error
    """
    try:
        # Validate and convert string dates to datetime
        try:
            maturity = datetime.fromisoformat(bond.maturity_date.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid maturity_date format: {e}")

        try:
            issue = datetime.fromisoformat(bond.issue_date.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid issue_date format: {e}")

        # Validate dates are logical
        if issue >= maturity:
            raise HTTPException(status_code=400, detail="Issue date must be before maturity date")

        # Create Bond object
        bond_obj = Bond(
            bond_id=bond.bond_id,
            bond_type=BondType[bond.bond_type.upper()],
            face_value=bond.face_value,
            coupon_rate=bond.coupon_rate,
            maturity_date=maturity,
            issue_date=issue,
            current_price=bond.current_price,
            credit_rating=bond.credit_rating or "BBB",
            issuer=bond.issuer or "Unknown",
            frequency=bond.frequency,
            callable=bond.callable,
            convertible=bond.convertible,
        )

        # Use service layer (includes validation, audit logging, metrics)
        container = get_container()
        bond_service = container.get_bond_service()
        result = bond_service.create_bond(bond_obj)
        created_bond = handle_service_result(result, "Failed to create bond")

        return BondResponse(
            bond_id=created_bond.bond_id,
            bond_type=created_bond.bond_type.name,
            face_value=created_bond.face_value,
            coupon_rate=created_bond.coupon_rate,
            current_price=created_bond.current_price,
            credit_rating=created_bond.credit_rating,
            issuer=created_bond.issuer,
        )
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid bond data: {str(e)}")


@router.get("", response_model=List[BondResponse], dependencies=[Depends(verify_api_key)])
async def list_bonds(
    skip: int = Query(0, ge=0, description="Number of records to skip (for pagination)", example=0),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of records to return", example=100
    ),
    issuer: Optional[str] = Query(
        None, description="Filter by issuer name (case-insensitive partial match)", example="Corp"
    ),
    bond_type: Optional[str] = Query(None, description="Filter by bond type", example="CORPORATE"),
):
    """
    List bonds with optional filtering

    Retrieves a paginated list of bonds with optional filtering by issuer and bond type.

    **Query Parameters:**
    - `skip`: Number of records to skip (default: 0)
    - `limit`: Maximum number of records to return (default: 100, max: 1000)
    - `issuer`: Filter by issuer name (partial match, case-insensitive)
    - `bond_type`: Filter by bond type (exact match)

    **Returns:**
    - List of bond objects matching the criteria

    **Example:**
    ```
    GET /bonds?skip=0&limit=50&bond_type=CORPORATE&issuer=Example
    ```
    """
    # Build filters for service layer
    filters = {}
    if issuer:
        filters["issuer"] = issuer
    if bond_type:
        try:
            filters["bond_type"] = BondType[bond_type.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid bond_type: {bond_type}")

    # Use service layer (includes audit logging, metrics)
    container = get_container()
    bond_service = container.get_bond_service()
    result = bond_service.find_bonds(filters=filters if filters else None)
    bonds = handle_service_result(result, "Failed to retrieve bonds")

    # Paginate
    bonds = bonds[skip : skip + limit]

    return [
        BondResponse(
            bond_id=b.bond_id,
            bond_type=b.bond_type.name,
            face_value=b.face_value,
            coupon_rate=b.coupon_rate,
            current_price=b.current_price,
            credit_rating=b.credit_rating,
            issuer=b.issuer,
        )
        for b in bonds
    ]


@router.get("/{bond_id}", response_model=BondResponse, dependencies=[Depends(verify_api_key)])
async def get_bond(bond_id: str):
    """
    Get a specific bond by ID

    Retrieves detailed information about a specific bond.

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - Bond object with all details

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    """
    # Use service layer (includes audit logging, metrics)
    container = get_container()
    bond_service = container.get_bond_service()
    result = bond_service.get_bond(bond_id)
    bond = handle_service_result(result, "Failed to retrieve bond")

    return BondResponse(
        bond_id=bond.bond_id,
        bond_type=bond.bond_type.name,
        face_value=bond.face_value,
        coupon_rate=bond.coupon_rate,
        current_price=bond.current_price,
        credit_rating=bond.credit_rating,
        issuer=bond.issuer,
    )
