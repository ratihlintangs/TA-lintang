import os
import sys
import joblib
import pandas as pd
import numpy as np
import traceback # Import untuk mencetak detail error
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from math import sin, cos, pi

# Tambahkan path induk agar bisa mengimpor model database (tetap dipertahankan, tapi menggunakan dummy)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 1. KONFIGURASI DAN STRUKTUR DATA (Pydantic) ---

TARGET_COLUMNS = ['temperature', 'humidity', 'pressure', 'wind_speed']
DAYS_TO_FORECAST = 7

class ModelEvaluation(BaseModel):
    target: str
    rmse: float
    mae: float
    model_name: str

class ForecastData(BaseModel):
    date: str 
    temperature_c: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_kmh: float

class ForecastResponse(BaseModel):
    predictions: List[ForecastData]
    evaluations: List[ModelEvaluation]


# --- DUMMY DATA LOADER ---

def load_dummy_data() -> pd.DataFrame:
    """
    Fungsi ini digunakan sebagai pengganti load_data_from_db 
    untuk memastikan server dapat berjalan tanpa database.
    """
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    return pd.DataFrame({
        'tanggal': dates,
        'temperature': np.random.uniform(25.0, 32.0, 30),
        'humidity': np.random.uniform(70.0, 90.0, 30),
        'pressure': np.random.uniform(1000.0, 1015.0, 30),
        'wind_speed': np.random.uniform(1.0, 5.0, 30),
    }).rename(columns={'tanggal': 'tanggal'})


# --- 2. LOGIKA UTILITY ---

def _load_model_and_scalers(target_col: str) -> Tuple[Any, Any, Any]:
    """Memuat model dan scalers yang tersimpan menggunakan joblib."""
    
    model_file = f'{target_col.lower()}_mlp_model.pkl'
    scaler_x_file = f'{target_col.lower()}_scaler_X.pkl'
    scaler_y_file = f'{target_col.lower()}_scaler_y.pkl'
    
    # Dapatkan path absolut dari direktori induk (seharusnya backend/)
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../backend/services
    base_dir = os.path.abspath(os.path.join(current_dir, '..')) # .../backend/

    # Path Absolut ke file
    model_path = os.path.join(base_dir, model_file)
    scaler_x_path = os.path.join(base_dir, scaler_x_file)
    scaler_y_path = os.path.join(base_dir, scaler_y_file)

    try:
        if not os.path.exists(model_path):
             print(f"DEBUG PATH: File TIDAK DITEMUKAN: {model_path}")
             raise FileNotFoundError(f"Model {model_file} tidak ditemukan di path yang diharapkan.")
             
        model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        print(f"INFO: Model {target_col.upper()} berhasil dimuat dari: {model_path}")
        return model, scaler_X, scaler_y
    except FileNotFoundError as e:
        print(f"FATAL ERROR: File model HILANG atau salah path: {e}")
        traceback.print_exc() # Cetak traceback untuk FileNotFoundError
        raise 
    except Exception as e:
        # Menangkap error deserialisasi atau kompatibilitas
        print(f"FATAL ERROR: Gagal memuat model/scaler {target_col} (Joblib/Kompatibilitas): {e}")
        print(f"           PATH ABSOLUT YANG DICARI: {model_path}")
        print("           ----- FULL TRACEBACK -----")
        traceback.print_exc() # Ini penting untuk Joblib/Pickle error
        print("           --------------------------")
        raise Exception(f"Gagal memuat model {target_col}: {e}")

# ... (Fungsi _create_features dan _predict_single_variable sama seperti sebelumnya) ...
def _create_features(df: pd.DataFrame, target_col: str, forecast_date: datetime):
    """
    Membuat fitur (lag dan musiman) untuk tanggal prediksi tertentu.
    """
    
    # 1. Fitur Musiman
    dayofyear = forecast_date.timetuple().tm_yday
    sin_day = sin(2 * pi * dayofyear / 365.25)
    cos_day = cos(2 * pi * dayofyear / 365.25)
    
    # 2. Fitur Hari Kerja
    is_weekday = 1 if forecast_date.weekday() < 5 else 0
    
    # 3. Fitur Lag 
    target_col_smooth = f'{target_col}_smooth'
    
    if target_col_smooth not in df.columns:
        df[target_col_smooth] = df[target_col].rolling(window=3, center=True).mean()
        df[target_col_smooth].fillna(df[target_col].mean(), inplace=True)
        
    df = df.sort_values('tanggal').reset_index(drop=True)
    
    # Tentukan tanggal yang diperlukan untuk fitur lag
    date_lag_7 = (forecast_date - timedelta(days=7)).strftime('%Y-%m-%d')
    date_lag_14 = (forecast_date - timedelta(days=14)).strftime('%Y-%m-%d')
    date_lag_30 = (forecast_date - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Ambil nilai lag
    def get_lag_value(df, date_str):
        result = df.loc[df['tanggal'].dt.strftime('%Y-%m-%d') == date_str, target_col_smooth]
        if not result.empty:
            return result.iloc[0]
        # Fallback: jika tanggal tidak ada, gunakan nilai rata-rata
        return df[target_col_smooth].mean() 
        
    lag_7 = get_lag_value(df, date_lag_7)
    lag_14 = get_lag_value(df, date_lag_14)
    lag_30 = get_lag_value(df, date_lag_30)

    # Mengembalikan array fitur [lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]
    return np.array([lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]).reshape(1, -1)


def _predict_single_variable(df_hist: pd.DataFrame, target_col: str) -> List[Tuple[datetime, float]]:
    """Melakukan prediksi 7 hari ke depan untuk satu variabel cuaca."""
    
    # Langkah ini memanggil _load_model_and_scalers dan akan melempar error jika gagal
    model, scaler_X, scaler_y = _load_model_and_scalers(target_col)

    # 1. Persiapan data historis (hanya 30 hari terakhir)
    df_hist_clean = df_hist[['tanggal', target_col]].copy()
    df_hist_clean['tanggal'] = pd.to_datetime(df_hist_clean['tanggal'])
    df_hist_clean = df_hist_clean.sort_values('tanggal').tail(30).reset_index(drop=True)
    
    # 2. Inisialisasi DataFrame sementara untuk auto-regression
    df_temp = df_hist_clean.copy()
    last_date = df_temp['tanggal'].iloc[-1]
    predictions_output = []

    # 3. Lakukan iterasi prediksi 7 hari
    for i in range(1, DAYS_TO_FORECAST + 1):
        forecast_date = last_date + timedelta(days=i)
        
        # a. Buat fitur
        X_features = _create_features(df_temp, target_col, forecast_date)
        
        # b. Skala fitur
        X_scaled = scaler_X.transform(X_features)
        
        # c. Prediksi
        y_scaled = model.predict(X_scaled)
        
        # d. Inverse Transform 
        y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]
        
        # e. Simpan hasil
        predictions_output.append((forecast_date, y_pred))
        
        # f. Tambahkan hasil prediksi ke df_temp untuk auto-regression
        new_row = {
            'tanggal': forecast_date,
            target_col: y_pred,
            f'{target_col}_smooth': y_pred
        }
        df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)

    return predictions_output

# --- 3. FUNGSI UTAMA ---

def load_all_models_and_data() -> Tuple[List[ForecastData], List[ModelEvaluation]]:
    """
    Fungsi ini adalah alur kerja utama.
    """
    print("\n========================================================")
    print(f"INFO: Memulai alur kerja prediksi 7 hari. Cek {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')} untuk model.")
    print("========================================================\n")
    
    # 1. Muat Data Historis (menggunakan DUMMY)
    try:
        df_hist = load_dummy_data() 
    except Exception as e:
        print(f"ERROR: Gagal memuat data historis (DUMMY): {e}")
        return [], []
        
    if df_hist.empty:
        print("PERINGATAN: Data historis kosong. Tidak dapat melakukan prediksi.")
        return [], []

    # 2. Lakukan Prediksi untuk Semua Target
    results_by_target = {}
    for target in TARGET_COLUMNS:
        try:
            forecasts = _predict_single_variable(df_hist, target)
            results_by_target[target] = forecasts
            print(f"INFO: Prediksi 7 hari untuk {target.upper()} selesai.")
        except Exception as e:
            # Jika ada kegagalan, pesan error detail sudah dicetak di _load_model_and_scalers
            print(f"FATAL ERROR: Prediksi gagal total untuk {target}. Server akan mengembalikan 503.")
            return [], []


    # 3. Gabungkan Hasil Prediksi Harian & 4. Evaluasi Dummy
    combined_forecast: List[ForecastData] = []
    
    if results_by_target:
        for i in range(DAYS_TO_FORECAST):
            date_str = results_by_target[TARGET_COLUMNS[0]][i][0].strftime('%Y-%m-%d')
            daily_forecast = ForecastData(
                date=date_str,
                temperature_c=results_by_target['temperature'][i][1],
                humidity_percent=results_by_target['humidity'][i][1],
                pressure_hpa=results_by_target['pressure'][i][1],
                wind_speed_kmh=results_by_target['wind_speed'][i][1]
            )
            combined_forecast.append(daily_forecast)

    dummy_evaluations = [
        ModelEvaluation(target="Temperature", rmse=2.15, mae=1.58, model_name="MLP"),
        ModelEvaluation(target="Humidity", rmse=4.51, mae=3.10, model_name="MLP"),
        ModelEvaluation(target="Pressure", rmse=0.89, mae=0.62, model_name="MLP"),
        ModelEvaluation(target="Wind Speed", rmse=0.76, mae=0.55, model_name="MLP"),
    ]
    
    print("\n========================================================")
    print("INFO: Startup berhasil! Server siap melayani permintaan.")
    print("========================================================\n")
    return combined_forecast, dummy_evaluations

# --- 4. Fungsi Tambahan (Sesuai Kerangka Awal) ---

def get_latest_predictions():
    """Fungsi placeholder, tidak digunakan di app.py."""
    pass