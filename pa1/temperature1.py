import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import math

# === 1. BACA DATA EXCEL ===
file_path = "data 3 bulan1.xlsx"  # Pastikan file ada di folder yang sama
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
df.columns = ['Tavg']

# === 2. MODEL AUTOREGRESSIVE ===
tavg_values = df['Tavg'].values

# Gunakan lag=7 (1 minggu)
model = AutoReg(tavg_values, lags=7, old_names=False)
model_fit = model.fit()

# Prediksi 7 hari pertama bulan April
forecast = model_fit.predict(start=len(tavg_values), end=len(tavg_values) + 6)

# Tampilkan hasil prediksi
print("=== Prediksi Tavg 7 Hari Pertama April ===")
for i, val in enumerate(forecast, 1):
    print(f"Hari {i}: {val:.2f}")

# === 3. EVALUASI MODEL ===
# Nilai aktual suhu Tavg 7 hari pertama bulan April
actual_april = [29.2, 29.5, 29.4, 28.6, 28.6, 28.3, 29.7]

# Hitung metrik evaluasi
mse = mean_squared_error(actual_april, forecast)
rmse = math.sqrt(mse)
mae = mean_absolute_error(actual_april, forecast)
r2 = r2_score(actual_april, forecast)

# Tampilkan hasil evaluasi
print("\n=== Evaluasi Model ===")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# === 4. VISUALISASI ===
plt.figure(figsize=(10, 6))
days = np.arange(1, 8)

plt.plot(days, actual_april, marker='o', linestyle='-', label='Aktual', color='blue')
plt.plot(days, forecast, marker='o', linestyle='--', label='Prediksi', color='orange')

plt.title("Prediksi vs Aktual Tavg (7 Hari Pertama April)")
plt.xlabel("Hari di Bulan April")
plt.ylabel("Tavg (°C)")
plt.xticks(days)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
