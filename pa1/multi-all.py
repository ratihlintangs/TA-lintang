import pandas as pd
import numpy as np
import sys
# Pastikan openpyxl sudah terinstal: pip install openpyxl
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import pi

# === Konfigurasi Umum ===
TARGETS = ['T2M', 'RH2M', 'WS2M', 'PS']
LAG_DAYS = [7, 14, 30]
TEST_SIZE_WEEKS = 4 
TEST_SIZE = TEST_SIZE_WEEKS * 7 # 28 hari

def run_multi_output_forecast(file_path, targets, lag_days, test_size):
    """
    Melatih dan mengevaluasi model Multi-Output MLPRegressor.
    """
    print(f"\n=======================================================")
    print(f"       Memulai Pemrosesan File: {file_path}")
    print(f"=======================================================")
    
    try:
        # === 1. Load & Preprocessing ===
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} tidak ditemukan. Melewatkan pemrosesan.")
        return None

    df['tanggal'] = pd.to_datetime(df['tanggal'], dayfirst=True, errors='coerce')
    df = df.sort_values('tanggal').dropna(subset=['tanggal'])

    # Bersihkan dan konversi SEMUA 4 TARGET
    for col in targets:
        df[col] = df[col].replace('-', np.nan).astype(float)

    # Tambah fitur lag dan musiman
    FEATURES = []
    # Buat fitur lag untuk SEMUA 4 TARGET
    for target in targets:
        for lag in lag_days:
            col_name = f'lag_{lag}_{target}'
            df[col_name] = df[target].shift(lag)
            FEATURES.append(col_name)

    # Tambahkan fitur musiman
    df['dayofyear'] = df['tanggal'].dt.dayofyear
    df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365)
    df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365)
    FEATURES.extend(['sin_day', 'cos_day'])

    # Hapus baris NaN yang dihasilkan oleh fitur shifting dan nilai data yang kotor
    df_clean = df.dropna(subset=FEATURES + targets).copy()
    
    # --- 2. PENCEGAHAN ERROR: Cek Panjang Data ---
    MIN_REQUIRED_ROWS = test_size + 1 

    if len(df_clean) < MIN_REQUIRED_ROWS:
        print(f"❌ Peringatan: Data yang sudah bersih ({len(df_clean)} baris) terlalu pendek.")
        print(f"Diperlukan minimal {MIN_REQUIRED_ROWS} baris (Train + Test).")
        return None
    
    # === 3. Split Data Training dan Testing ===
    df_train = df_clean[:-test_size].copy()
    df_test = df_clean[-test_size:].copy().reset_index(drop=True)

    X_train_raw = df_train[FEATURES]
    Y_train_raw = df_train[targets]
    X_test_raw = df_test[FEATURES]

    # === 4. Scaling Data ===
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    # === 5. Pelatihan Model ===
    model_multi_4 = MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        max_iter=500, 
        random_state=42, 
        early_stopping=True,
        verbose=False
    )
    print("-> Pelatihan model Multi-Output (MLP) dimulai...")
    model_multi_4.fit(X_train_scaled, Y_train_scaled)
    print("-> Pelatihan selesai.")

    # === 6. Prediksi dan Evaluasi ===
    Y_pred_scaled = model_multi_4.predict(X_test_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    df_test_results = df_test[['tanggal'] + targets].copy()
    evaluasi_summary = []

    print("\n========== HASIL EVALUASI (4 TARGET) ==========")
    for i, target in enumerate(targets):
        actual = df_test_results[target].values
        predicted = Y_pred[:, i]
        
        df_test_results[f'Prediksi_{target}'] = predicted
        
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        print(f"✅ {target}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        evaluasi_summary.append({
            'File': file_path, 
            'Target': target, 
            'MSE': f"{mse:.4f}", 
            'RMSE': f"{rmse:.4f}", 
            'MAE': f"{mae:.4f}", 
            'R2': f"{r2:.4f}"
        })

    # Simpan data prediksi dan aktual per file ke Excel (Optional, tapi membantu)
    df_test_results.to_excel(f"hasil_prediksi_{file_path.replace('.csv', '.xlsx')}", index=False)
    print(f"✅ Data prediksi untuk {file_path} berhasil disimpan ke Excel.")

    return evaluasi_summary

# === BAGIAN EKSEKUSI (Ubah nama file sesuai data Anda) ===

# Daftar file yang ingin Anda proses (asumsi: kumpulan1.csv hingga kumpulan12.csv)
files_to_process = [f"kumpulan{i}.csv" for i in range(1, 13)]
all_results = []

for file_name in files_to_process:
    results = run_multi_output_forecast(file_name, TARGETS, LAG_DAYS, TEST_SIZE)
    if results is not None:
        all_results.extend(results)

# === Rangkuman Akhir: Disimpan ke Excel ===
if all_results:
    print("\n\n--- RANGKUMAN EVALUASI SELURUH BULAN ---")
    df_final_summary = pd.DataFrame(all_results)
    
    excel_file_name = "rangkuman_evaluasi_multi_output.xlsx"
    # Menggunakan to_excel() untuk menyimpan ringkasan hasil evaluasi akhir
    df_final_summary.to_excel(excel_file_name, index=False)
    
    print(f"✅ File Excel '{excel_file_name}' berhasil disimpan.")
    
    # Tampilkan rata-rata performa di konsol
    print("\n--- RATA-RATA EVALUASI SEMUA FILE ---")
    for col in ['MSE', 'RMSE', 'MAE', 'R2']:
        df_final_summary[col] = pd.to_numeric(df_final_summary[col], errors='coerce')
        
    df_avg = df_final_summary.groupby('Target')[['MSE', 'RMSE', 'MAE', 'R2']].mean().reset_index()
    
    print(df_avg.to_string(index=False))