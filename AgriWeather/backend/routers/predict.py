from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import os

from backend.database import get_db
from backend.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/weather", tags=["Weather Prediction"])

# ==============================
# MODEL MODE SWITCH (ENV)
# ==============================
# v1 = baseline (absolute semua)
# v2 = hybrid (T/RH delta, lainnya absolute)
MODEL_MODE = os.getenv("AGRI_MODEL_MODE", "v2").strip().lower()

if MODEL_MODE == "v1":
    from backend.services.predict_service_v1 import PredictService
elif MODEL_MODE == "v2":
    from backend.services.predict_service_v2 import PredictService
else:
    # fallback aman
    from backend.services.predict_service_v2 import PredictService
    MODEL_MODE = "v2"

print(f"✅ AGRI_MODEL_MODE aktif: {MODEL_MODE}")
print(f"✅ PredictService module: {PredictService.__module__}")

# ==============================
# INIT SERVICES
# ==============================
predict_service = PredictService()
evaluation_service = EvaluationService(predict_service=predict_service)

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
    Endpoint evaluasi model cuaca (rolling evaluation).
    Mode v1/v2 mengikuti AGRI_MODEL_MODE karena evaluation_service memakai predict_service yang sama.
    """
    return evaluation_service.evaluate_all(
        db=db,
        test_days=test_days
    )
