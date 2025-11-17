from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services.predict_service import PredictService
from services.utils import WeatherPrediction
from typing import List

router = APIRouter(
    prefix="/weather/predict",
    tags=["Prediction"],
)

# Inisialisasi service
predict_service = PredictService()

@router.get("/", response_model=List[WeatherPrediction])
def get_prediction(
    db: Session = Depends(get_db),
    days_ahead: int = Query(7, ge=1, le=30, description="Jumlah hari yang akan diprediksi (1-30)"),
):
    """
    Menghasilkan prediksi cuaca untuk N hari ke depan menggunakan model AR placeholder.
    """
    try:
        predictions = predict_service.generate_weather_prediction(db, days_ahead=days_ahead)
        
        if not predictions:
             raise HTTPException(status_code=404, detail="Data historis tidak ditemukan untuk memulai prediksi")
             
        return predictions
    except HTTPException:
        # Melewatkan HTTPException jika sudah dinaikkan di service
        raise
    except Exception as e:
        print(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail="Kesalahan saat menjalankan model prediksi")