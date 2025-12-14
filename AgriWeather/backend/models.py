from sqlalchemy import Column, Integer, Float, String, Text, Date
from backend.database import Base  # ✅ lebih aman untuk mode -m

class WeatherHistoryModel(Base):
    __tablename__ = "weather_history"

    id = Column(Integer, primary_key=True, index=True)

    # ✅ Samakan dengan DB: TEXT "YYYY-MM-DD"
    date = Column(String, unique=True, index=True)

    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)

    source = Column(Text, nullable=True)

    def __repr__(self):
        return f"<WeatherHistory(date={self.date}, temp={self.temperature})>"


class EvaluationHistoryModel(Base):
    __tablename__ = "evaluation_history"

    id = Column(Integer, primary_key=True, index=True)
    eval_date = Column(Date, index=True)      # tanggal evaluasi (as_of)
    horizon_days = Column(Integer, default=7)
    parameter = Column(String, index=True)    # temperature/humidity/pressure/wind_speed

    mse = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    r2 = Column(Float)

    n_samples = Column(Integer, default=0)