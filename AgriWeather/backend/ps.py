import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sin, cos, pi
from scipy.stats import zscore
import warnings

# Mengabaikan peringatan (terutama dari MLPRegressor yang sering muncul saat training)
warnings.filterwarnings('ignore')

def create_model(model_name: str):
    """
    Membuat dan mengembalikan model machine learning untuk peramalan.
    
    Args:
        model_name (str): Nama model yang akan digunakan ('MLP' atau 'LR').
        
    Returns:
        object: Model ML yang sudah diinisialisasi.
    """
    if model_name == 'MLP':
        # Multilayer Perceptron (MLP) Regressor - Pilihan populer untuk Tugas Akhir
        # Anda dapat mengubah hidden_layer_sizes untuk tuning model (misalnya: (50, 50))
        return MLPRegressor(hidden_layer_sizes=(100, 50), 
                            activation='relu', 
                            solver='adam', 
                            max_iter=500, 
                            random_state=42)
    elif model_name == 'LR':
        # Linear Regression (sebagai baseline/perbandingan)
        return LinearRegression()
    else:
        # Default jika nama model tidak dikenali
        return MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)

def run_ps_forecast(df_raw: pd.DataFrame, model_name: str = 'MLP', target_col: str = 'pressure'):
    """
    Melakukan Preprocessing, Modeling, dan Evaluasi Peramalan Tekanan Udara (PS).

    Args:
        df_raw (pd.DataFrame): DataFrame mentah yang dimuat dari database (dari database.py).
        model_name (str): Nama model yang akan digunakan ('MLP' atau 'LR').
        target_col (str): Nama kolom target ('pressure' dari DB, yang merupakan 'PS' di data Anda).

    Returns:
        dict: Berisi hasil prediksi (Aktual vs Prediksi) dan list evaluasi.
    """
    evaluasi_list = []
    
    # === 1. PREPROCESSING DATA ===
    df = df_raw.copy()
    
    # Pastikan data yang dimuat dari DB memiliki kolom 'date'
    df['tanggal'] = df['date'] 
    
    # Konversi kolom target ke numerik dan hapus baris NaN yang mungkin terjadi
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di data mentah.")
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col, 'tanggal'])
    df = df.sort_values('tanggal')

    # a. Smoothing (Moving Average 3 hari)
    # Ini membantu model menangkap tren jangka panjang
    df['PS_smooth'] = df[target_col].rolling(window=3, center=True).mean()
    df = df.dropna(subset=['PS_smooth'])
    target_col_smooth = 'PS_smooth'

    # b. Outlier Removal (Z-Score)
    # Menghapus nilai ekstrem (di luar 3 standar deviasi)
    df['zscore'] = zscore(df[target_col_smooth])
    df = df[df['zscore'].abs() < 3]

    # c. Tambahkan Fitur Autoregressive (LAG) 
    # Inti dari Time Series: menggunakan nilai masa lalu (lag) untuk prediksi masa depan.
    df['lag_7'] = df[target_col_smooth].shift(7)   # Nilai 7 hari lalu
    df['lag_14'] = df[target_col_smooth].shift(14)  # Nilai 14 hari lalu
    df['lag_30'] = df[target_col_smooth].shift(30)  # Nilai 30 hari lalu

    # d. Tambahkan Fitur Musiman (Sinus & Cosinus)
    # Mengkodekan siklus tahunan agar model tahu posisi musim 
    df['dayofyear'] = df['tanggal'].dt.dayofyear
    df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365.25)

    # e. Fitur Hari Kerja/Hari Libur
    df['dayofweek'] = df['tanggal'].dt.dayofweek
    df['is_weekday'] = np.where(df['dayofweek'] < 5, 1, 0)

    # Hapus baris NaN yang muncul akibat fitur lag
    df = df.dropna()

    # Tentukan fitur/variabel yang akan dimasukkan ke model (X)
    features = ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day', 'is_weekday']
    kombinasi_name = "Kombinasi D: Lag + Musiman + Hari Kerja"

    # === 2. MODELING (Training & Rolling Window Evaluation) ===
    
    model = create_model(model_name)
    train_size = int(len(df) * 0.8) # 80% data untuk training
    
    scaler_X = StandardScaler() # Scaler untuk fitur (X)
    scaler_y = StandardScaler() # Scaler untuk target (Y)

    X = df[features].values
    y = df[target_col_smooth].values.reshape(-1, 1)

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    all_results = []
    window_size = 7 # Evaluasi per 7 hari (mingguan)
    
    # Loop untuk Rolling Window Evaluation
    for i in range(train_size, len(df), window_size):
        train_end = i
        test_start = i
        test_end = min(i + window_size, len(df))
        
        if test_start >= len(df):
            break

        X_train, X_test = X_scaled[:train_end], X_scaled[test_start:test_end]
        y_train, y_test = y_scaled[:train_end], y_scaled[test_start:test_end]
        
        if len(X_train) == 0:
            continue 
            
        # Training Model
        model.fit(X_train, y_train.ravel())
        
        # Prediksi
        y_pred_scaled = model.predict(X_test)
        
        # Inverse Transform (Mengembalikan ke Skala Asli)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = scaler_y.inverse_transform(y_test).flatten()
        
        # === 3. EVALUASI HASIL ===
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        # Simpan evaluasi minggu ini
        evaluasi_list.append({
            'Kombinasi': kombinasi_name,
            'Model': f"{model_name} (AR-3 Lags)", 
            'Periode_Awal': df['tanggal'].iloc[test_start].strftime('%Y-%m-%d'),
            'Periode_Akhir': df['tanggal'].iloc[test_end-1].strftime('%Y-%m-%d'),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2': round(r2, 4),
            'Fitur': ', '.join(features)
        })

        # Simpan hasil prediksi (tanggal, aktual, prediksi)
        results = pd.DataFrame({
            'tanggal': df['tanggal'].iloc[test_start:test_end].dt.strftime('%Y-%m-%d'), 
            'Aktual': y_actual.round(4),
            'Prediksi': y_pred.round(4)
        })
        all_results.append(results)

    final_results = pd.concat(all_results).to_dict('records') if all_results else []
    
    # Mengembalikan hasil sesuai format ForecastResponse
    return {
        "predictions": final_results,
        "evaluations": sorted(evaluasi_list, key=lambda x: x['RMSE'])
    }