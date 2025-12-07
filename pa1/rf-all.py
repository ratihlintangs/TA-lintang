import pandas as pd
import numpy as np
import sys
# Pastikan openpyxl sudah terinstal: pip install openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import pi

# === Konfigurasi Umum ===
TARGETS = ['T2M', 'RH2M', 'WS2M', 'PS']
LAG_DAYS = [7, 14, 30]
TEST_SIZE_WEEKS = 4 
TEST_SIZE = TEST_SIZE_WEEKS * 7 # 28 hari

def run_rf_forecast_for_target(file_path, target_col, lag_days, test_size):
    """
    Melatih dan mengevaluasi model Single-Output RandomForestRegressor untuk satu target.
    """
    
    try:
        # === 1. Load & Preprocessing ===
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Jika file tidak ada, lewati
        return None

    df['tanggal'] = pd.to_datetime(df['tanggal'], dayfirst=True, errors='coerce')
    df = df.sort_values('tanggal').dropna(subset=['tanggal'])

    # Bersihkan dan konversi Target saat ini
    df[target_col] = df[target_col].replace('-', np.nan).astype(float)

    # Tambah fitur lag dan musiman
    FEATURES = []

    # Buat fitur lag HANYA untuk target saat ini
    for lag in lag_days:
        col_name = f'lag_{lag}'
        df[col_name] = df[target_col].shift(lag)
        FEATURES.append(col_name)

    # Tambahkan fitur musiman
    df['dayofyear'] = df['tanggal'].dt.dayofyear
    df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365)
    df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365)
    FEATURES.extend(['sin_day', 'cos_day'])

    # Hapus baris NaN yang dihasilkan oleh fitur shifting
    df_clean = df.dropna(subset=FEATURES + [target_col]).copy()
    
    # --- 2. PENCEGAHAN ERROR: Cek Panjang Data ---
    MIN_REQUIRED_ROWS = test_size + 1 

    if len(df_clean) < MIN_REQUIRED_ROWS:
        print(f"   [-- {target_col} --] ❌ Data terlalu pendek ({len(df_clean)} baris).")
        return None
    
    # === 3. Split Data Training dan Testing ===
    df_train = df_clean[:-test_size].copy()
    df_test = df_clean[-test_size:].copy().reset_index(drop=True)

    X_train_raw = df_train[FEATURES]
    Y_train_raw = df_train[[target_col]]
    X_test_raw = df_test[FEATURES]
    
    # Random Forest tidak wajib scaling, tapi kita lakukan hanya pada Y
    scaler_Y = MinMaxScaler()
    
    # Scaling HANYA pada target (Y), X tidak perlu di-scale untuk RF
    Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)

    # === 4. Pelatihan Model ===
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print(f"   [-- {target_col} --] -> Pelatihan model Random Forest dimulai...")
    # Gunakan .ravel() karena Random Forest Regressor adalah single-output
    model_rf.fit(X_train_raw, Y_train_scaled.ravel()) 
    print(f"   [-- {target_col} --] -> Pelatihan selesai.")

    # === 5. Prediksi dan Evaluasi ===
    Y_pred_scaled = model_rf.predict(X_test_raw)
    
    # Ubah ke bentuk 2D untuk Inverse Transform
    Y_pred_scaled = Y_pred_scaled.reshape(-1, 1) 
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    actual = df_test[target_col].values
    predicted = Y_pred.flatten()
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print(f"   [-- {target_col} --] ✅ {target_col}: RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # Simpan data prediksi dan aktual ke dalam df_test_results
    df_test['Prediksi'] = predicted
    
    # Kembalikan ringkasan evaluasi
    return {
        'File': file_path, 
        'Target': target_col, 
        'MSE': f"{mse:.4f}", 
        'RMSE': f"{rmse:.4f}", 
        'MAE': f"{mae:.4f}", 
        'R2': f"{r2:.4f}"
    }

# === BAGIAN EKSEKUSI UTAMA ===

# Daftar file yang ingin Anda proses (asumsi: kumpulan1.csv hingga kumpulan12.csv)
files_to_process = [f"kumpulan{i}.csv" for i in range(1, 13)]
all_results = []

for file_name in files_to_process:
    print(f"\n=======================================================")
    print(f"       Memproses File: {file_name}")
    print(f"=======================================================")
    
    for target in TARGETS:
        result = run_rf_forecast_for_target(file_name, target, LAG_DAYS, TEST_SIZE)
        if result is not None:
            all_results.append(result)

# === Rangkuman Akhir: Disimpan ke Excel ===
if all_results:
    print("\n\n--- RANGKUMAN EVALUASI SELURUH BULAN ---")
    df_final_summary = pd.DataFrame(all_results)
    
    excel_file_name = "rangkuman_evaluasi_random_forest.xlsx"
    df_final_summary.to_excel(excel_file_name, index=False)
    
    print(f"✅ File Excel '{excel_file_name}' berhasil disimpan, berisi 48 baris hasil evaluasi.")
    
    # Tampilkan rata-rata performa di konsol
    print("\n--- RATA-RATA EVALUASI SEMUA FILE ---")
    for col in ['MSE', 'RMSE', 'MAE', 'R2']:
        df_final_summary[col] = pd.to_numeric(df_final_summary[col], errors='coerce')
        
    df_avg = df_final_summary.groupby('Target')[['MSE', 'RMSE', 'MAE', 'R2']].mean().reset_index()
    
    print(df_avg.to_string(index=False))