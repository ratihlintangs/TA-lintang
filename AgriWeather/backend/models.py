from sqlalchemy import Column, Integer, Float, String, Text
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
