from fastapi import APIRouter
from pydantic import BaseModel
from database import load_data_from_db
from services.predict_service import get_forecast_and_evaluation

router = APIRouter()

# Skema respons Pydantic (opsional, tapi baik untuk struktur)
class Evaluation(BaseModel):
    Kombinasi: str
    Model: str
    Periode_Awal: str
    Periode_Akhir: str
    RMSE: float
    MAE: float
    R2: float
    Fitur: str

class PredictionResult(BaseModel):
    tanggal: str
    Aktual: float
    Prediksi: float

class ForecastResponse(BaseModel):
    predictions: list[PredictionResult]
    evaluations: list[Evaluation]

@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast():
    """
    Endpoint untuk menjalankan model peramalan PS, mengambil data dari DB, 
    dan mengembalikan hasil prediksi serta evaluasi bergulir.
    """
    try:
        # 1. Muat Data Mentah dari Database
        df_raw = load_data_from_db()
        
        if df_raw.empty or 'PS' not in df_raw.columns:
            raise ValueError("Gagal memuat data dari database atau kolom 'PS' hilang.")

        # 2. Jalankan Prediction Service
        results = get_forecast_and_evaluation(df_raw)
        
        # 3. Kembalikan Hasil
        return results

    except Exception as e:
        # Pengecualian akan ditangani oleh FastAPI dan dikembalikan sebagai status 500
        print(f"Error pada endpoint /forecast: {e}")
        # Kembalikan pesan error yang jelas (walaupun response_model akan memvalidasi)
        raise Exception(f"Terjadi kesalahan saat memproses peramalan: {str(e)}")