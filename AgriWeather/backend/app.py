from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from services.utils import load_all_models_and_data, ForecastResponse, ModelEvaluation, get_latest_predictions

# ----------------------------------------------------
# 1. SETUP APLIKASI DAN KONFIGURASI CORS
# ----------------------------------------------------

app = FastAPI(
    title="AgriWeather Prediction API",
    description="API untuk ramalan cuaca (Pressure, PS) menggunakan model Time Series MLP.",
    version="1.0.0"
)

# Konfigurasi CORS (Cross-Origin Resource Sharing)
# Ini wajib agar frontend (yang berjalan di port 5500) bisa mengakses backend (port 8000)
origins = [
    "http://localhost",
    "http://localhost:5500",  # Izinkan koneksi dari server frontend Anda
    "http://127.0.0.1:5500",  # Juga izinkan 127.0.0.1
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # Daftar asal yang diizinkan
    allow_credentials=True,             # Izinkan cookies/header otentikasi
    allow_methods=["*"],                # Izinkan semua metode (GET, POST, dll.)
    allow_headers=["*"],                # Izinkan semua header
)


# ----------------------------------------------------
# 2. LIFESPAN EVENT HANDLERS (LOAD MODEL PADA STARTUP)
# ----------------------------------------------------

# Struktur untuk menyimpan hasil prediksi dan evaluasi (akan diisi di startup)
class AppState:
    forecast_data: List[ForecastResponse] = []
    evaluation_data: List[ModelEvaluation] = []

app_state = AppState()

@app.on_event("startup")
async def startup_event():
    print("INFO: Memuat model dan data pada startup...")
    try:
        # Panggil fungsi dari services.utils untuk memuat semua yang diperlukan
        all_forecasts, all_evaluations = load_all_models_and_data()
        
        # Simpan data yang dimuat ke dalam state aplikasi
        app_state.forecast_data = all_forecasts
        app_state.evaluation_data = all_evaluations
        print("INFO: Model dan data berhasil dimuat.")
        
    except Exception as e:
        print(f"ERROR: Gagal memuat model/data saat startup: {e}")
        # Jika model gagal dimuat, aplikasi bisa tetap berjalan tapi API akan mengembalikan error
        # Untuk kasus ini, kita akan biarkan API endpoint yang menangani error jika data kosong.


@app.on_event("shutdown")
def shutdown_event():
    print("INFO: Aplikasi dimatikan.")


# ----------------------------------------------------
# 3. ENDPOINT API
# ----------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Selamat datang di AgriWeather API. Akses /weather/predict/ untuk data ramalan."}

@app.get("/weather/predict/", response_model=ForecastResponse)
async def get_weather_prediction():
    """
    Mengembalikan data ramalan cuaca (Pressure PS) dan evaluasi model.
    Data ini dimuat sekali saat startup aplikasi.
    """
    if not app_state.forecast_data or not app_state.evaluation_data:
        # Kasus ini akan terjadi jika load_all_models_and_data gagal saat startup
        raise HTTPException(
            status_code=503, 
            detail="Layanan tidak tersedia. Model atau data gagal dimuat pada startup server."
        )

    # Mengembalikan data terbaru yang sudah dimuat (sesuai format ForecastResponse)
    return {
        "predictions": app_state.forecast_data,
        "evaluations": app_state.evaluation_data
    }

# ----------------------------------------------------
# 4. Fungsi Utility (untuk mendapatkan 7 hari terakhir - opsional)
# ----------------------------------------------------

# Tidak perlu membuat fungsi get_latest_predictions di sini karena data sudah ada di app_state
# dan frontend hanya mengambil 7 hari terakhir dari array 'predictions'