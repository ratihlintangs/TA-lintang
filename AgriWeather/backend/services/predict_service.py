from sqlalchemy.orm import Session
from database import WeatherHistoryModel
from services.utils import WeatherPrediction
from typing import List, Optional, Dict
from datetime import datetime, timedelta

# Impor pustaka untuk model AR
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.sm_exceptions import HessianInversionWarning
import warnings

# Abaikan peringatan inversi Hessian yang sering muncul saat training model AR sederhana
warnings.filterwarnings('ignore', category=HessianInversionWarning)

# Orde Model AR yang Digunakan. 
AR_LAG_ORDER = 5 
# Batas jumlah data historis yang diambil untuk training
DATA_LIMIT = 100 

# --- Implementasi Logika Model AR ---

def _run_ar_forecast(series: pd.Series, days_ahead: int, var_name: str) -> List[float]:
    """
    Melakukan training dan forecasting menggunakan model AutoReg untuk satu variabel.
    """
    
    # KUNCI PERBAIKAN: Selalu bersihkan data terlebih dahulu
    # Ambil 30 data terakhir yang BERSIH (dropna)
    train_data = series.tail(30).dropna()
    
    # Jika train_data yang bersih masih kosong
    if train_data.empty:
         print(f"FATAL: Data {var_name} kosong setelah pembersihan. Mengembalikan nilai default 0.")
         return [0.0] * days_ahead

    # Cek apakah data yang bersih sudah cukup untuk model AR
    if len(train_data) < AR_LAG_ORDER + 1:
        # Jika data terlalu sedikit, kembalikan nilai terakhir
        last_value = train_data.iloc[-1]
        print(f"Peringatan: Data {var_name} ({len(train_data)} hari) tidak cukup untuk AR({AR_LAG_ORDER}). Menggunakan nilai terakhir: {last_value}.")
        return [last_value] * days_ahead

    try:
        # 1. Training Model AR
        model = AutoReg(train_data, lags=AR_LAG_ORDER, trend='c')
        model_fit = model.fit()

        # 2. Forecasting dengan Indeks Numerik yang Aman
        # Indeks train_data berakhir di len(train_data) - 1. 
        # Kita ingin memprediksi hari berikutnya (start=len(train_data))
        forecast_start_index = len(train_data)
        forecast_end_index = len(train_data) + days_ahead - 1

        forecast = model_fit.predict(start=forecast_start_index, end=forecast_end_index)
        
        # KUNCI PERBAIKAN LAGI: Lakukan pengecekan pada hasil prediksi
        if forecast.isnull().any():
            # Ini adalah inti dari error "NoneType"
            print(f"Peringatan Kritis: Prediksi {var_name} menghasilkan NaN (NoneType). Menggunakan nilai terakhir.")
            return [train_data.iloc[-1]] * days_ahead
            
        return forecast.tolist()
    except Exception as e:
        print(f"Error AR forecast untuk {var_name}: {e}. Menggunakan nilai terakhir.")
        # Kembalikan nilai terakhir jika model gagal karena alasan lain
        return [train_data.iloc[-1]] * days_ahead


class PredictService:
    """
    Layanan yang menangani pengambilan data dan logika prediksi.
    Sekarang terintegrasi dengan Model AR untuk keempat variabel cuaca.
    """

    def get_all_historical_data_for_model(self, db: Session) -> Optional[List[WeatherHistoryModel]]:
        """
        Mengambil semua data historis, diurutkan berdasarkan ID, untuk training model.
        """
        # Batasi pengambilan data untuk efisiensi
        data = db.query(WeatherHistoryModel).order_by(
            WeatherHistoryModel.id.asc()
        ).limit(DATA_LIMIT).all() 
        return data


    def generate_weather_prediction(
        self, 
        db: Session, 
        days_ahead: int = 7
    ) -> List[WeatherPrediction]:
        """
        Menghasilkan prediksi cuaca untuk N hari ke depan menggunakan 4 model AR terpisah.
        """
        # 1. Ambil data historis
        all_data = self.get_all_historical_data_for_model(db)
        
        if not all_data:
            return []

        # 2. Persiapan Data untuk Model AR
        df = pd.DataFrame([vars(d) for d in all_data])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # PERBAIKAN PENTING: Lakukan pembersihan (dropna) pada seluruh dataset
        # sebelum diproses oleh model. Ini adalah baris yang memecahkan masalah 'NoneType'.
        df = df.dropna()

        if df.empty:
             print("Peringatan: DataFrame kosong setelah menghapus nilai NaN. Tidak bisa melakukan prediksi.")
             return []

        # Tentukan tanggal terakhir untuk memulai prediksi
        last_date_obj = df.index[-1].to_pydatetime().date()
        
        # 3. Jalankan Model AR untuk SEMUA 4 VARIABEL
        print("Mulai training dan prediksi 4 model AR...")
        
        forecasts: Dict[str, List[float]] = {
            # Sekarang _run_ar_forecast menerima data yang sudah bersih
            'temperature': _run_ar_forecast(df['temperature'], days_ahead, 'Suhu'),
            'humidity': _run_ar_forecast(df['humidity'], days_ahead, 'Kelembapan'),
            'pressure': _run_ar_forecast(df['pressure'], days_ahead, 'Tekanan'),
            'wind_speed': _run_ar_forecast(df['wind_speed'], days_ahead, 'Kecepatan Angin'),
        }

        # 4. Finalisasi Hasil Prediksi
        predictions: List[WeatherPrediction] = []

        for i in range(days_ahead):
            pred_date = (last_date_obj + timedelta(days=i + 1)).strftime('%Y-%m-%d')
            
            # Ambil hasil dari masing-masing model
            temp_pred = forecasts['temperature'][i]
            hum_pred = forecasts['humidity'][i]
            pres_pred = forecasts['pressure'][i]
            wind_pred = forecasts['wind_speed'][i]

            predictions.append(
                WeatherPrediction(
                    date=pred_date,
                    temperature=round(temp_pred, 2),
                    # Pastikan Kelembapan antara 0 dan 100
                    humidity=round(np.clip(hum_pred, 0, 100), 2), 
                    pressure=round(pres_pred, 2),
                    # Pastikan Kecepatan Angin non-negatif
                    wind_speed=round(np.clip(wind_pred, 0, 50), 2), 
                    model=f"4x AR Model V1.0 (Lags={AR_LAG_ORDER})"
                )
            )

        return predictions