from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta

# --- Pydantic Schemas (Untuk I/O API) ---

class WeatherHistory(BaseModel):
    """Skema untuk data cuaca historis yang diambil dari DB."""
    id: int
    date: str
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    source: str

    class Config:
        # Memungkinkan Pydantic membaca dari objek ORM (SQLAlchemy)
        from_attributes = True

class WeatherPrediction(BaseModel):
    """Skema untuk data prediksi cuaca."""
    date: str = Field(..., description="Tanggal prediksi (misalnya YYYY-MM-DD)")
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    model: str = Field(default="AR Model V0.1", description="Nama/versi model yang digunakan")

# --- Fungsi Utilitas Model (Placeholder) ---

def simulate_prediction(
    last_known_data: WeatherHistory, 
    days_ahead: int = 7
) -> List[WeatherPrediction]:
    """
    Fungsi placeholder untuk simulasi prediksi AR.
    Saat ini hanya menggeser data terakhir dengan sedikit noise.
    """
    
    # Ambil nilai terakhir sebagai basis
    temp = last_known_data.temperature
    hum = last_known_data.humidity
    pres = last_known_data.pressure
    wind = last_known_data.wind_speed
    
    # Parsing tanggal terakhir
    try:
        start_date = datetime.strptime(last_known_data.date, '%Y-%m-%d')
    except ValueError:
        # Jika format DB tidak YYYY-MM-DD, gunakan tanggal hari ini
        start_date = datetime.now()

    predictions: List[WeatherPrediction] = []

    for i in range(1, days_ahead + 1):
        # Tanggal prediksi
        pred_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        
        # Simulasi pergeseran/noise sederhana
        # Nanti, logika AR yang sebenarnya akan menggantikan ini
        import numpy as np
        
        # Tambahkan sedikit noise (simulasi tren atau perubahan)
        temp_pred = temp + np.random.uniform(-0.5, 0.5)
        hum_pred = hum + np.random.uniform(-1.0, 1.0)
        pres_pred = pres + np.random.uniform(-0.5, 0.5)
        wind_pred = wind + np.random.uniform(-0.1, 0.1)

        predictions.append(
            WeatherPrediction(
                date=pred_date,
                temperature=round(temp_pred, 2),
                humidity=round(hum_pred, 2),
                pressure=round(pres_pred, 2),
                wind_speed=round(wind_pred, 2),
            )
        )
        
        # Update "basis" untuk simulasi AR sederhana (data hari ini mempengaruhi besok)
        temp = temp_pred
        hum = hum_pred
        # ... dan seterusnya
        
    return predictions