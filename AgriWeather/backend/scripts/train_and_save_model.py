import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from math import sin, cos, pi
from scipy.stats import zscore
import joblib
import warnings
import sys
import os

from backend.config_location import (
    LATITUDE,
    LONGITUDE,
    LOCATION_NAME,
    INVALID_VALUES,
    NASA_SAFE_LAG_DAYS
)


# === [PENTING] Penyesuaian Jalur Impor ===
# Memastikan skrip dapat mengimpor dari folder induk (backend)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Impor fungsi muat data nyata dari file database.py
from backend.database import load_data_from_db 

# Mengabaikan peringatan saat training
warnings.filterwarnings('ignore')

# === Konfigurasi Target ===
# Daftar KOLOM TARGET yang akan diprediksi (Sesuai dengan skema database Anda)
TARGET_COLUMNS = [
    'temperature',
    'humidity',
    'pressure',
    'wind_speed'
]

def create_model():
    """Mengembalikan model MLPRegressor yang akan dilatih."""
    return MLPRegressor(hidden_layer_sizes=(100, 50), 
                        activation='relu', 
                        solver='adam', 
                        max_iter=500, 
                        random_state=42)

def preprocess_data(df_raw: pd.DataFrame, target_col: str):
    """Melakukan preprocessing data untuk kolom target tertentu."""
    df = df_raw.copy()
    
    # Pembersihan Data & Konversi
    df[target_col] = df[target_col].replace('-', np.nan)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce') 
    
    df = df.dropna(subset=[target_col, 'tanggal'])
    df = df.sort_values('tanggal')

    # Smoothing (Moving Average)
    target_col_smooth = f'{target_col}_smooth'
    df[target_col_smooth] = df[target_col].rolling(window=3, center=True).mean()
    df = df.dropna(subset=[target_col_smooth])

    # Tambahkan Fitur Deret Waktu (Lag features)
    df['lag_7'] = df[target_col_smooth].shift(7)
    df['lag_14'] = df[target_col_smooth].shift(14)
    df['lag_30'] = df[target_col_smooth].shift(30)

    # Fitur Musiman (Sinus & Cosinus)
    # Asumsi kolom 'tanggal' sudah berformat datetime
    df['dayofyear'] = df['tanggal'].dt.dayofyear
    df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365.25)

    # Fitur Hari Kerja/Hari Libur
    df['dayofweek'] = df['tanggal'].dt.dayofweek
    df['is_weekday'] = np.where(df['dayofweek'] < 5, 1, 0)
    df = df.dropna()
    
    return df

def train_and_save_single_model(df_raw: pd.DataFrame, target_col: str):
    """Melatih dan menyimpan model untuk satu variabel target."""
    print(f"\n=======================================================")
    print(f"⏳ Mulai Training untuk TARGET: {target_col.upper()}")
    print(f"=======================================================")
    
    # 1. Preprocessing Data
    df = preprocess_data(df_raw, target_col)
    
    if df.empty:
        print(f"❌ Gagal: Setelah preprocessing, DataFrame untuk {target_col} kosong.")
        return

    features = ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day', 'is_weekday']
    target_col_smooth = f'{target_col}_smooth'
    
    # 2. Persiapan Data Training/Test (80% Training)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]

    X_train = df_train[features].values
    y_train = df_train[target_col_smooth].values.reshape(-1, 1)

    # 3. Inisialisasi dan fit Scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # 4. Latih Model
    model = create_model()
    model.fit(X_train_scaled, y_train_scaled.ravel())
    print(f"✅ Training {target_col.upper()} Selesai.")

    # 5. Simpan Model dan Scalers
    MODEL_FILENAME = f'{target_col.lower()}_mlp_model.pkl'
    SCALER_X_FILENAME = f'{target_col.lower()}_scaler_X.pkl'
    SCALER_Y_FILENAME = f'{target_col.lower()}_scaler_y.pkl'
    
    # File model akan disimpan di direktori 'backend/'
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler_X, SCALER_X_FILENAME)
    joblib.dump(scaler_y, SCALER_Y_FILENAME)
    
    print(f"✅ Model dan Scalers untuk {target_col} berhasil disimpan sebagai {MODEL_FILENAME}.")

def run_all_training():
    """Mengambil data mentah dan menjalankan training untuk semua target."""
    print("⏳ Memuat data nyata DENGAN SEMUA KOLOM dari database...")
    raw_data = load_data_from_db() 
    
    if raw_data.empty:
        print(f"❌ Gagal: Data yang dimuat kosong. Gagal memulai training multi-target.")
        return

    # Pastikan kolom tanggal sudah berupa datetime
    raw_data['tanggal'] = pd.to_datetime(raw_data['tanggal'], errors='coerce')
    raw_data = raw_data.dropna(subset=['tanggal'])


    print(f"Total {len(raw_data)} baris data historis dimuat.")

    for target in TARGET_COLUMNS:
        if target in raw_data.columns:
            train_and_save_single_model(raw_data, target)
        else:
            print(f"❌ Peringatan: Kolom '{target}' tidak ditemukan di data mentah. Melewati training untuk kolom ini.")

if __name__ == '__main__':
    run_all_training()
    print("\n>>> SEMUA PROSES TRAINING SELESAI. Empat set model (.pkl) telah dibuat.")