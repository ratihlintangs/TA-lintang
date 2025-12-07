import os
import pickle
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from datetime import date

# Import dari file proyek Anda
from database import engine, SessionLocal
from models import WeatherHistoryModel

# --- KONFIGURASI MODEL ---
MODEL_FILE_PATH = "data/weather_predictor_mlp.pkl"
LAG_DAYS = 7 # Jumlah hari historis yang digunakan untuk memprediksi 1 hari ke depan
# --- END KONFIGURASI ---

def create_sequences(df: pd.DataFrame, lag_days: int):
    """
    Membuat fitur time-lag (X) dan target (y) dari data time series.
    """
    X, y = [], []
    # PERUBAHAN: Nama fitur disesuaikan dengan models.py dan skema DB Anda
    features = ['temperature', 'humidity', 'pressure', 'wind_speed']
    
    # Data harus diurutkan secara kronologis
    data = df[features].values 
    
    for i in range(lag_days, len(data)):
        # X: 7 hari data historis sebelum hari ini
        X.append(data[i - lag_days:i, :]) 
        # y: Nilai hari ke-8 (target)
        y.append(data[i, :]) 
        
    return np.array(X), np.array(y)


def train_and_save_model(db: Session):
    """
    Fungsi utama untuk mengambil data, melatih model MLP, dan menyimpannya.
    """
    print("--- Memulai Proses Pelatihan Model MLP ---")

    # 1. Ambil Data dari Database
    # Ambil semua data historis, urutkan berdasarkan tanggal
    data = db.query(WeatherHistoryModel).order_by(WeatherHistoryModel.date.asc()).all()
    
    if len(data) < LAG_DAYS + 1:
        print(f"ERROR: Data historis terlalu sedikit ({len(data)} hari). Minimal {LAG_DAYS + 1} hari diperlukan.")
        return

    df = pd.DataFrame([record.__dict__ for record in data])
    df = df.sort_values(by='date').set_index('date')
    
    print(f"INFO: Berhasil memuat {len(df)} catatan data historis.")
    
    # Daftar fitur yang disesuaikan
    features_list = ['temperature', 'humidity', 'pressure', 'wind_speed']

    # 2. Scaling Data
    # Gunakan scaler untuk semua fitur
    scaler = MinMaxScaler()
    # PERUBAHAN: Nama kolom disesuaikan di sini juga
    df[features_list] = scaler.fit_transform(
        df[features_list]
    )

    # 3. Buat Sequence (Fitur Lag)
    X, y = create_sequences(df, LAG_DAYS)
    
    # MLPRegressor membutuhkan input X yang 2D (samples, features)
    # Kita harus meratakan (flatten) fitur lag
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape((n_samples, n_timesteps * n_features))
    
    print(f"INFO: Data siap. X shape: {X_flat.shape}, y shape: {y.shape}")

    # 4. Inisialisasi dan Latih Model MLP
    # Sesuaikan parameter (hidden_layer_sizes, max_iter, activation) sesuai kebutuhan TA Anda.
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        activation='relu', 
        solver='adam', 
        max_iter=500, 
        random_state=42
    )
    
    print("INFO: Mulai pelatihan model...")
    model.fit(X_flat, y)
    print("INFO: Pelatihan selesai.")

    # 5. Simpan Model dan Scaler
    # Simpan model DAN scaler (penting untuk inverse transform di fase prediksi!)
    model_data = {
        'model': model,
        'scaler': scaler,
        'lag_days': LAG_DAYS,
        # PERUBAHAN: Nama fitur disesuaikan
        'features': features_list
    }
    
    # Pastikan folder data ada
    os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
    
    with open(MODEL_FILE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"BERHASIL: Model dan Scaler disimpan ke {MODEL_FILE_PATH}")


if __name__ == "__main__":
    db_session = SessionLocal()
    try:
        train_and_save_model(db_session)
    finally:
        db_session.close()