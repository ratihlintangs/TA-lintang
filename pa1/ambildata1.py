import json
import pandas as pd
from datetime import datetime

# Muat file JSON
with open('karangploso_power.json', 'r') as f:
    data = json.load(f)

# Ambil semua parameter dari data
parameters = data['properties']['parameter']

# Tentukan rentang waktu
start_date = pd.to_datetime('2025-04-06')
end_date = pd.to_datetime('2025-06-28')

# Buat DataFrame kosong untuk gabungan
merged_df = None

# Iterasi setiap parameter (misal: T2M, RH2M, WS2M, dll.)
for param_name, param_data in parameters.items():
    # Konversi parameter menjadi DataFrame
    df = pd.DataFrame(list(param_data.items()), columns=['tanggal', param_name])
    df['tanggal'] = pd.to_datetime(df['tanggal'], format='%Y%m%d')
    
    # Filter berdasarkan rentang tanggal
    df = df[(df['tanggal'] >= start_date) & (df['tanggal'] <= end_date)]
    
    # Gabungkan semua parameter ke satu DataFrame berdasarkan tanggal
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on='tanggal', how='outer')

# Urutkan tanggal dari terbaru ke terlama
merged_df = merged_df.sort_values(by='tanggal', ascending=False).reset_index(drop=True)

# Tampilkan preview
print(merged_df.head())

# Simpan ke CSV
merged_df.to_csv('dataactual_all.csv', index=False)
print("Data semua parameter telah disimpan ke: dataactual_all.csv")
