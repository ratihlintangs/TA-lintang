import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# 1. Data harga saham aktual (25 hari)
data_harga = [
    100.496714, 100.358450, 101.006138, 102.529168, 102.295015,
    102.060878, 103.640091, 104.407525, 103.938051, 104.480611,
    104.017193, 103.551464, 103.793426, 101.880146, 100.155228,
    99.592940, 98.580109, 98.894357, 97.986332, 96.574029,
    98.039678, 97.813901, 97.881429, 96.456681, 95.912299
]

hari = list(range(1, 26))  # Hari ke-1 sampai ke-25

# 2. Buat DataFrame
df = pd.DataFrame({'Hari': hari, 'Harga': data_harga})
print("Data Harga Saham (Hari 1–25):")
print(df)

# 3. Visualisasi data
plt.figure(figsize=(10, 4))
plt.plot(df['Hari'], df['Harga'], marker='o', label='Data Aktual')
plt.title('Harga Saham Hari ke-1 sampai ke-25')
plt.xlabel('Hari')
plt.ylabel('Harga')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 4. Bangun dan latih model AR
model = AutoReg(df['Harga'], lags=3)  # gunakan 3 lag
model_fit = model.fit()
print("\nKoefisien Model AR:")
print(model_fit.params)

# 5. Prediksi harga saham hari ke-26 sampai ke-30
prediksi = model_fit.predict(start=len(df), end=len(df)+4)
hari_prediksi = list(range(26, 31))
df_prediksi = pd.DataFrame({'Hari': hari_prediksi, 'Harga': prediksi})

print("\nHasil Prediksi Harga Saham Hari ke-26 sampai ke-30:")
for hari, harga in zip(hari_prediksi, prediksi):
    print(f"Hari ke-{hari}: {harga:.2f}")

# 6. Visualisasi gabungan aktual + prediksi
plt.figure(figsize=(10, 4))
plt.plot(df['Hari'], df['Harga'], marker='o', label='Data Aktual')
plt.plot(df_prediksi['Hari'], df_prediksi['Harga'], marker='s', linestyle='--', color='red', label='Prediksi')
plt.title('Prediksi Harga Saham Hari ke-26 sampai ke-30')
plt.xlabel('Hari')
plt.ylabel('Harga')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
