import pandas as pd

# Baca file Excel
df = pd.read_excel('klasifikasi.xlsx')

# Gunakan baris ke-1 sebagai header dan buang baris atas
df_cleaned = df[1:].copy()
df_cleaned.columns = df.iloc[0]
df_cleaned = df_cleaned.dropna(subset=['T2M', 'WS2M', 'RH2M', 'PS'])

# Konversi kolom numerik
df_cleaned['T2M'] = df_cleaned['T2M'].astype(float)
df_cleaned['RH2M'] = df_cleaned['RH2M'].astype(float)
df_cleaned['WS2M'] = df_cleaned['WS2M'].astype(float)
df_cleaned['PS'] = df_cleaned['PS'].astype(float)

# Fungsi klasifikasi cuaca lanjutan
def klasifikasi_cuaca(row):
    suhu = row['T2M']
    rh = row['RH2M']
    angin = row['WS2M']
    tekanan = row['PS']

    if rh > 85 and suhu < 27 and tekanan < 100.8:
        return "Hujan"
    elif angin > 3:
        return "Berangin"
    elif suhu > 30 and rh < 60:
        return "Cerah"
    elif 25 <= suhu <= 30 and 60 <= rh <= 85:
        return "Berawan"
    elif rh > 80 and 27 <= suhu <= 29 and 100.8 <= tekanan <= 101.2:
        return "Hujan Ringan"
    elif rh >= 75 and suhu < 27:
        return "Berawan Tebal"
    elif suhu > 30 and rh >= 70:
        return "Panas Lembap"
    else:
        return "Lainnya"

# Tambahkan klasifikasi
df_cleaned['klasifikasi_model'] = df_cleaned.apply(klasifikasi_cuaca, axis=1)

# Simpan hasil ke file baru
df_cleaned.to_excel('klasifikasi_output_lanjutan.xlsx', index=False)

print("Klasifikasi cuaca lanjutan selesai dan disimpan ke 'klasifikasi_output_lanjutan.xlsx'")
