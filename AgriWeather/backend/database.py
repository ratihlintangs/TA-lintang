import os
import logging
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

logger = logging.getLogger(__name__)

# --- KONFIGURASI PATH DATABASE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "weather_data.db")
SQLITE_URL = f"sqlite:///{DB_PATH}"

logger.debug(f"Menghubungkan ke DB di: {DB_PATH}")

# --- KONFIGURASI SQLALCHEMY ---
engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# --- FASTAPI DEPENDENCY ---
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- UTIL UNTUK PREDICTION SERVICE ---
def load_data_from_db(db: Session, only_valid: bool = False) -> pd.DataFrame:
    """
    Load semua data historis cuaca dari database menjadi DataFrame.

    only_valid=False:
        - perilaku lama (ambil semua baris)

    only_valid=True:
        - hanya ambil baris yang nilai utamanya lengkap (non-null)
          untuk mencegah 'tail null' (tanggal ada tapi datanya belum tersedia).
    """
    try:
        from backend.models import WeatherHistoryModel

        q = db.query(WeatherHistoryModel).order_by(WeatherHistoryModel.date.asc())

        # Filter baris valid (hindari NULL yang bikin last_date salah)
        if only_valid:
            q = q.filter(
                WeatherHistoryModel.temperature.isnot(None),
                WeatherHistoryModel.humidity.isnot(None),
                WeatherHistoryModel.pressure.isnot(None),
                WeatherHistoryModel.wind_speed.isnot(None),
            )

        data = q.all()
        df = pd.DataFrame([vars(d) for d in data])

        if df.empty:
            logger.info("DB load: 0 baris (kosong)")
            return df

        if "_sa_instance_state" in df.columns:
            df = df.drop(columns=["_sa_instance_state"])

        logger.info(f"DB load sukses: {len(df)} baris (only_valid={only_valid})")
        return df

    except Exception as e:
        logger.error(f"FATAL DB ERROR: {e}")
        raise
