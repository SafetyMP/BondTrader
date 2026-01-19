"""
Machine Learning Routes
ML-enhanced predictions
"""

from fastapi import APIRouter, Depends

from bondtrader.core.container import get_container
from scripts.api.middleware import verify_api_key
from scripts.api.models import MLPredictionResponse
from scripts.api_helpers import handle_service_result

router = APIRouter(tags=["Machine Learning"])


@router.get(
    "/predict/{bond_id}",
    response_model=MLPredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ml_predict(bond_id: str):
    """
    Get ML-adjusted prediction for a bond

    Uses machine learning models to adjust theoretical bond valuations,
    accounting for market inefficiencies and non-linear relationships.

    **Path Parameters:**
    - `bond_id`: Unique identifier of the bond

    **Returns:**
    - ML-adjusted fair value
    - Adjustment factor applied
    - Model confidence score

    **Errors:**
    - `404 Not Found`: Bond with the specified ID does not exist
    - `503 Service Unavailable`: ML model not available or not trained
    - `500 Internal Server Error`: Prediction error

    **Note:** ML models must be trained before use. See training scripts for details.
    """
    # Use service layer for ML prediction (handles model loading internally)
    container = get_container()
    bond_service = container.get_bond_service()
    result = bond_service.predict_with_ml(bond_id, model_type="enhanced")
    prediction = handle_service_result(result, "ML prediction failed")

    return MLPredictionResponse(
        bond_id=bond_id,
        theoretical_fair_value=prediction["theoretical_fair_value"],
        ml_adjusted_fair_value=prediction["ml_adjusted_fair_value"],
        adjustment_factor=prediction.get("adjustment_factor", 1.0),
        ml_confidence=prediction.get("ml_confidence", 0.0),
    )
