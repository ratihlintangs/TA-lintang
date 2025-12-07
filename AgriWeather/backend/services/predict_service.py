import pandas as pd
import numpy as np
import joblib
import datetime
from math import sin, cos, pi
import sys
import os
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException
from database import load_data_from_db # Memuat dari database.py yang sudah ada

# === KONFIGURASI MODEL ===
TARGET_COLUMNS = [
    'temperature',
    'humidity',
    'pressure',
    'wind_speed'
]
# Lokasi file model diasumsikan berada di direktori induk (backend/)
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..') 

class PredictService:
    """
    Layanan yang bertanggung jawab untuk memuat model, melakukan prediksi rekursif 7 hari, 
    dan mengelola data historis untuk fitur lag.
    """
    def __init__(self):
        # Struktur untuk menyimpan aset model yang dimuat
        self.model_assets = {}
        self._load_all_model_assets()
        
    def _load_model_assets(self, target_col: str):
        """Memuat model dan scalers untuk kolom target tertentu."""
        model_filename = f'{target_col.lower()}_mlp_model.pkl'
        scaler_x_filename = f'{target_col.lower()}_scaler_X.pkl'
        scaler_y_filename = f'{target_col.lower()}_scaler_y.pkl'
        
        try:
            # Gunakan jalur relatif yang benar ke direktori model (backend/)
            model = joblib.load(os.path.join(MODEL_DIR, model_filename))
            scaler_X = joblib.load(os.path.join(MODEL_DIR, scaler_x_filename))
            scaler_y = joblib.load(os.path.join(MODEL_DIR, scaler_y_filename))
            
            self.model_assets[target_col] = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }
            print(f"✅ INFO: Aset model untuk {target_col.upper()} berhasil dimuat.")
            return True
        except FileNotFoundError:
            raise FileNotFoundError(f"File model atau scaler untuk '{target_col}' tidak ditemukan di jalur: {MODEL_DIR}. Pastikan .pkl ada.")
        except Exception as e:
            print(f"❌ ERROR: Gagal memuat aset model untuk '{target_col}': {e}")
            return False

    def _load_all_model_assets(self):
        """Memuat semua 4 set model saat layanan diinisialisasi."""
        for target_col in TARGET_COLUMNS:
            try:
                self._load_model_assets(target_col)
            except FileNotFoundError as e:
                # Menghentikan inisialisasi jika ada model yang hilang
                print(f"❌ KRITIS: Gagal inisialisasi layanan. {e}")
                # Kosongkan aset agar generate_weather_prediction tahu bahwa inisialisasi gagal
                self.model_assets = {} 
                break
    
    def _prepare_initial_data(self, df_raw: pd.DataFrame, target_col: str) -> List[float]:
        """
        Melakukan pra-pemrosesan dan mendapatkan 30 nilai smoothed terakhir 
        dari data historis untuk memulai fitur lag.
        """
        target_col_smooth = f'{target_col}_smooth'
        df = df_raw.copy()
        
        # Lakukan smoothing (window=3)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce') 
        df = df.dropna(subset=[target_col])
        df[target_col_smooth] = df[target_col].rolling(window=3, center=True).mean()
        df = df.dropna(subset=[target_col_smooth])
        
        # Kembalikan 30 hari terakhir sebagai basis untuk lag 7, 14, 30
        if len(df) < 30:
            raise HTTPException(status_code=404, detail=f"Data historis tidak cukup ({len(df)} hari). Minimal 30 hari dibutuhkan untuk fitur lag.")
            
        historic_values = df[target_col_smooth].tail(30).tolist()
        return historic_values

    def generate_weather_prediction(self, db: Session, days_ahead: int) -> List[Dict[str, Any]]:
        """
        Menghasilkan ramalan cuaca rekursif untuk N hari ke depan.
        """
        if not self.model_assets:
            # Jika _load_all_model_assets gagal, beri tahu pengguna
            raise HTTPException(status_code=500, detail="Inisialisasi model prediksi gagal. Periksa log server untuk masalah FileNotFoundError.")

        # 1. Muat Data Historis
        df_raw = load_data_from_db(db) # Menggunakan get_db dari FastAPI
        
        if df_raw.empty:
            raise HTTPException(status_code=404, detail="Data historis kosong dari database.")

        # Pastikan 'date' adalah datetime dan data diurutkan
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['date']).sort_values('date')
        
        # Tentukan tanggal ramalan
        last_date = df_raw['date'].max().date() # Ambil hanya tanggalnya
        forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days_ahead + 1)]
        
        # 2. Persiapkan Struktur Hasil
        forecast_results = pd.DataFrame({'date': forecast_dates})
        
        # 3. Looping untuk setiap Target Cuaca
        for target_col in TARGET_COLUMNS:
            
            assets = self.model_assets.get(target_col)
            if not assets:
                continue # Lanjut ke kolom target berikutnya jika asetnya hilang
                
            model = assets['model']
            scaler_X = assets['scaler_X']
            scaler_y = assets['scaler_y']

            print(f"⏳ Memulai Prediksi Rekursif untuk: {target_col.upper()}")

            try:
                # Dapatkan nilai 30 hari terakhir yang sudah di-smoothing
                historic_values = self._prepare_initial_data(df_raw, target_col)
            except HTTPException as e:
                # Menangkap error data tidak cukup
                raise e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Kesalahan pra-pemrosesan data untuk {target_col}: {e}")

            predicted_values = []
            
            # 4. Prediksi Rekursif (N hari ke depan)
            for i in range(days_ahead):
                current_date = forecast_dates[i]
                
                # --- Tentukan Fitur Lag ---
                lag_7 = historic_values[-7]
                lag_14 = historic_values[-14]
                lag_30 = historic_values[-30]
                
                # --- Tentukan Fitur Musiman dan Waktu ---
                dayofyear = current_date.timetuple().tm_yday
                sin_day = sin(2 * pi * dayofyear / 365.25)
                cos_day = cos(2 * pi * dayofyear / 365.25)
                dayofweek = current_date.weekday()
                is_weekday = 1 if dayofweek < 5 else 0
                
                # Features: ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day', 'is_weekday']
                X_input = np.array([[lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]])
                
                # --- Prediksi ---
                X_scaled = scaler_X.transform(X_input)
                y_scaled_pred = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))[0][0]
                
                # Pembulatan
                y_pred_rounded = round(y_pred, 2)
                
                predicted_values.append(y_pred_rounded)
                
                # Tambahkan hasil prediksi ke 'historic_values' untuk langkah rekursif berikutnya
                historic_values.append(y_pred) 

            # 5. Gabungkan Hasil Prediksi
            forecast_results[target_col] = predicted_values
            print(f"✅ Prediksi {days_ahead}-hari untuk {target_col.upper()} selesai.")
        
        # 6. Format Hasil Akhir (List of Dictionaries)
        final_forecast_list = []
        for index, row in forecast_results.iterrows():
            final_forecast_list.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "temperature_c": row.get('temperature', None),
                "humidity_percent": row.get('humidity', None),
                "pressure_hpa": row.get('pressure', None),
                "wind_speed_kmh": row.get('wind_speed', None),
            })
            
        return final_forecast_list