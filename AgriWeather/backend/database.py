from sqlalchemy import create_engine, Column, Integer, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Tentukan lokasi database SQLite Anda
# Asumsi path relatif ke folder `data` seperti di struktur file
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/weather_data.db"

# Engine SQLAlchemy
# connect_args={"check_same_thread": False} diperlukan hanya untuk SQLite
# karena FastAPI berjalan di thread terpisah.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Pembuat sesi (SessionLocal)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class untuk model ORM
Base = declarative_base()

# --- Model Database (SQLAlchemy) ---
class WeatherHistoryModel(Base):
    """
    Model SQLAlchemy untuk tabel 'weather_history'.
    """
    __tablename__ = "weather_history"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Text, index=True)  # Menyimpan tanggal sebagai teks
    temperature = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    source = Column(Text)

# Fungsi utilitas untuk mendapatkan sesi DB
def get_db():
    """Dependency untuk mendapatkan sesi DB di FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Membuat tabel (hanya perlu dijalankan sekali saat inisialisasi, 
# atau jika Anda ingin memastikan tabel ada)
# Base.metadata.create_all(bind=engine)