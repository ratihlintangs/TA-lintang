from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from database import get_db
from services.predict_service import PredictService
from services.evaluation_service import EvaluationService

router = APIRouter(prefix="/weather", tags=["Weather Prediction"])

# ==============================
# INIT SERVICES
# ==============================
predict_service = PredictService()
evaluation_service = EvaluationService()


# ==============================
# ENDPOINT: FORECAST
# ==============================
@router.get("/forecast")
def forecast_weather(
    days_ahead: int = 7,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Endpoint prediksi cuaca N hari ke depan.
    """
    return predict_service.generate_weather_prediction(
        db=db,
        days_ahead=days_ahead
    )


# ==============================
# ENDPOINT: EVALUATION
# ==============================
@router.get("/evaluation")
def evaluate_weather_model(
    test_days: int = 7,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Endpoint evaluasi model cuaca menggunakan rolling evaluation.
    """
    return evaluation_service.evaluate_all(
        db=db,
        test_days=test_days
    )
