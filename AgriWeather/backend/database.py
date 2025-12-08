import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session 
from typing import Generator
import pandas as pd 

logger = logging.getLogger(__name__)

# --- KONFIGURASI PATH DATABASE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "weather_data.db")
SQLITE_URL = f"sqlite:///{DB_PATH}"

logger.debug(f"Mencoba menghubungkan ke DB di jalur: {DB_PATH}")

# --- KONFIGURASI SQLALCHEMY ---
engine = create_engine(
    SQLITE_URL, 
    connect_args={"check_same_thread": False}, 
    echo=False 
)

# SessionLocal: Kelas yang akan digunakan untuk membuat setiap Session DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base: Kelas dasar tempat model SQLAlchemy akan mewarisi
Base = declarative_base()

# PERBAIKAN: Lampirkan engine ke Base
Base.engine = engine 

# --- FASTAPI DEPENDENCY INJECTION ---
def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- TAMBAHAN UNTUK LOAD DATA PADA PREDICTION SERVICE ---
def load_data_from_db(db: Session):
    """
    Fungsi utilitas untuk memuat semua data dari database ke Pandas DataFrame.
    """
    try:
        # Menggunakan SQL query untuk memuat data
        from models import WeatherHistoryModel 
        
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
        # 🚨 PERUBAHAN KRITIS: Jangan sembunyikan error!
        logger.error(f"FATAL DB READ ERROR: Gagal memuat data dari DB: {e}")
        raise # <-- MELEMPAR ERROR ASLI KE KONSOLE UVICORN