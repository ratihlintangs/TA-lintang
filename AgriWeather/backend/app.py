from fastapi import FastAPI
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware # BARIS BARU: Import CORS
from routers import weather, predict 
from database import get_db, Base 
from models import WeatherHistoryModel  

# --- FUNGSI UNTUK MEMBUAT TABEL DATABASE ---
def create_tables():
    """Membuat tabel dalam database jika belum ada."""
    try:
        Base.metadata.create_all(bind=Base.engine) 
        print("INFO: Tabel database berhasil dibuat (jika belum ada).")
    except Exception as e:
        print(f"FATAL ERROR: Gagal membuat tabel database: {e}") 

# --- FASTAPI LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    print("INFO: Aplikasi FastAPI dimulai.")
    yield
    print("INFO: Aplikasi FastAPI dimatikan.")

# --- INISIALISASI APLIKASI ---
app = FastAPI(
    title="AgriWeather API",
    description="API untuk prediksi cuaca pertanian.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- KONFIGURASI CORS ---
origins = [
    "http://localhost:5500", 
    "http://127.0.0.1:5500", 
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MENDAFTARKAN ROUTER ---
app.include_router(weather.router)
app.include_router(predict.router)

# --- ENDPOINT TEST SEDERHANA ---
@app.get("/")
def read_root():
    return {"message": "Selamat datang di AgriWeather API! Akses /docs untuk dokumentasi."}