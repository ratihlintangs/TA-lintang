from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from database import get_db
# Import kelas PredictService yang baru
from services.predict_service import PredictService 
from services.utils import WeatherPrediction
from typing import List

router = APIRouter(
    prefix="/weather/predict",
    tags=["Prediction"],
)

# Inisialisasi service
# Perubahan: Inisialisasi service di luar handler agar model di-cache
predict_service = PredictService() 

@router.get("/", response_model=List[WeatherPrediction])
def get_prediction(
    db: Session = Depends(get_db),
    days_ahead: int = Query(7, ge=1, le=30, description="Jumlah hari yang akan diprediksi (1-30)"),
):
    """
    Menghasilkan prediksi cuaca untuk N hari ke depan menggunakan 4 model ML TA Anda.
    """
    try:
        # Panggil metode dari instance service
        predictions = predict_service.generate_weather_prediction(db, days_ahead=days_ahead)
        
        if not predictions:
             raise HTTPException(status_code=404, detail="Data historis tidak ditemukan atau tidak mencukupi untuk memulai prediksi")
             
        return predictions
    except FileNotFoundError as e:
        # Tangani jika file .pkl tidak ditemukan
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        # Melewatkan HTTPException jika sudah dinaikkan di service
        raise
    except Exception as e:
        print(f"Error generating prediction: {e}")
        # Tangani error umum lainnya
        raise HTTPException(status_code=500, detail="Kesalahan saat menjalankan model prediksi: " + str(e))