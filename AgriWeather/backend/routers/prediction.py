from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
# Import kelas PredictService
from backend.services.predict_service import PredictService 
# KOREKSI IMPOR: Gunakan ForecastData, ForecastResponse, dan ModelEvaluation dari utils
from backend.services.utils import ForecastData, ForecastResponse, ModelEvaluation 
from typing import List, Tuple

router = APIRouter(
    prefix="/weather/predict",
    tags=["Prediction"],
)

# Inisialisasi service
predict_service = PredictService() 

# response_model: Menggunakan ForecastResponse
@router.get("/", response_model=ForecastResponse)
def get_prediction(
    db: Session = Depends(get_db),
    days_ahead: int = Query(7, ge=1, le=30, description="Jumlah hari yang akan diprediksi (1-30)"),
):
    """
    Menghasilkan prediksi cuaca untuk N hari ke depan dan informasi evaluasi.
    """
    try:
        # KOREKSI KRITIS: ASUMSI service hanya mengembalikan LIST PREDIKSI
        predictions: List[ForecastData] = predict_service.generate_weather_prediction(db, days_ahead=days_ahead)
        
        # 2. DEFINISIKAN EVALUATIONS (Mengambil dari dummy data yang sudah ada di utils)
        # Kami menempatkan hardcoded dummy evaluation di sini karena service Anda hanya mengembalikan prediksi.
        evaluations = [
            ModelEvaluation(target="Temperature", rmse=2.15, mae=1.58, model_name="MLP"),
            ModelEvaluation(target="Humidity", rmse=4.51, mae=3.10, model_name="MLP"),
            ModelEvaluation(target="Pressure", rmse=0.89, mae=0.62, model_name="MLP"),
            ModelEvaluation(target="Wind Speed", rmse=0.76, mae=0.55, model_name="MLP"),
        ]
        
        if not predictions:
            # Jika service mengembalikan list kosong (karena database kosong atau error), tampilkan 404
            raise HTTPException(status_code=404, detail="Data historis tidak ditemukan atau tidak mencukupi untuk memulai prediksi")
            
        # Mengembalikan objek ForecastResponse yang sesuai dengan schema
        return ForecastResponse(
            predictions=predictions,
            evaluations=evaluations
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="Kesalahan Server: File model prediksi (.pkl) tidak ditemukan.")
        
    except HTTPException:
        # Melewatkan HTTPException jika sudah dinaikkan (misalnya error 404 di atas)
        raise
        
    except Exception as e:
        print(f"Error generating prediction: {e}")
        # Tangani error umum lainnya
        raise HTTPException(status_code=500, detail="Kesalahan umum saat menjalankan prediksi: " + str(e))