import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session # Import Session di sini
from typing import Generator
import pandas as pd # Import Pandas untuk load_data_from_db

logger = logging.getLogger(__name__)

# --- KONFIGURASI PATH DATABASE ---
# Asumsi kita berada di folder 'backend/'
# Path DB adalah 'backend/data/weather_data.db'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "weather_data.db")
SQLITE_URL = f"sqlite:///{DB_PATH}"

# Logging jalur DB (Anda melihat ini di output terminal)
logger.debug(f"Mencoba menghubungkan ke DB di jalur: {DB_PATH}")

# --- KONFIGURASI SQLALCHEMY ---
# create_engine: Objek ini bertanggung jawab untuk berinteraksi dengan DB
# check_same_thread=False diperlukan untuk SQLite agar dapat digunakan di FastAPI/multithreading
engine = create_engine(
    SQLITE_URL, 
    connect_args={"check_same_thread": False}, 
    echo=False # Ubah ke True untuk melihat semua query SQL di terminal
)

# SessionLocal: Kelas yang akan digunakan untuk membuat setiap Session DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base: Kelas dasar tempat model SQLAlchemy (WeatherHistoryModel) akan mewarisi
Base = declarative_base()

# --- FASTAPI DEPENDENCY INJECTION ---
def get_db() -> Generator:
    """
    Generator yang menghasilkan sesi database.
    Digunakan oleh FastAPI Depends() untuk manajemen sesi.
    """
    db = SessionLocal()
    try:
        # Menyerahkan sesi ke endpoint (contoh: get_prediction)
        yield db
    finally:
        # Menutup sesi setelah request selesai (penting!)
        db.close()

# --- TAMBAHAN UNTUK LOAD DATA PADA PREDICTION SERVICE ---
def load_data_from_db(db: Session):
    """
    Fungsi utilitas untuk memuat semua data dari database ke Pandas DataFrame.
    """
    try:
        # Menggunakan SQL query untuk memuat data
        from models import WeatherHistoryModel # Import di dalam untuk menghindari circular import
        
        # Query: Ambil semua data, diurutkan berdasarkan tanggal
        data = db.query(WeatherHistoryModel).order_by(WeatherHistoryModel.date.asc()).all()
        
        # Konversi ke DataFrame
        df = pd.DataFrame([vars(d) for d in data])
        
        # Hapus kolom internal SQLAlchemy yang tidak diperlukan
        if '_sa_instance_state' in df.columns:
            df = df.drop(columns=['_sa_instance_state'])
            
        logger.info(f"Berhasil memuat {len(df)} catatan historis dari DB.")
        return df
    except Exception as e:
        logger.error(f"Gagal memuat data dari DB: {e}")
        return pd.DataFrame() # Kembalikan DataFrame kosong jika gagal