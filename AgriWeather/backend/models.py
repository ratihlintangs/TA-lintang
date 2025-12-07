from sqlalchemy import Column, Integer, Float, Date, Text
from database import Base

class WeatherHistoryModel(Base):
    """
    Model Database untuk menyimpan data historis cuaca harian.
    Model ini berfungsi sebagai 'peta' yang menghubungkan Python ke tabel database 'weather_history'.
    """
    __tablename__ = "weather_history"
    
    # Kunci utama otomatis
    id = Column(Integer, primary_key=True, index=True)
    
    # Tanggal data cuaca (digunakan sebagai indeks dalam time series)
    date = Column(Date, unique=True, index=True)
    
    # Variabel cuaca dengan nama kolom yang sudah sesuai dengan DB Anda:
    temperature = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    
    # Kolom sumber data
    source = Column(Text)

    def __repr__(self):
        return f"<WeatherHistory(date={self.date}, temp={self.temperature})>"