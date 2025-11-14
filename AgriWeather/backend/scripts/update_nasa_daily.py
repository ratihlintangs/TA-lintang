import requests
import sqlite3
from datetime import datetime, timedelta
import os

# === Lokasi Stasiun Klimatologi Karangploso, Malang ===
LATITUDE = -7.9135
LONGITUDE = 112.6214

# === Tentukan path database ===
base_dir = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(base_dir, "data", "weather_data.db")

# === Koneksi ke database ===
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# === Cari tanggal terakhir di database ===
cursor.execute("SELECT MAX(date) FROM weather_history WHERE source = 'NASA'")
last_date_row = cursor.fetchone()
conn.commit()

if last_date_row and last_date_row[0]:
    last_date = datetime.strptime(last_date_row[0], "%Y-%m-%d").date()
else:
    raise Exception("Database belum memiliki data NASA. Jalankan fetch_nasa_history.py dulu.")

# === Tentukan rentang tanggal baru ===
start_date = last_date + timedelta(days=1)
end_date = datetime.now().date() - timedelta(days=1)

if start_date > end_date:
    print("⏸️ Tidak ada data baru untuk diperbarui.")
    conn.close()
    exit()

# Format tanggal untuk API NASA (YYYYMMDD)
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

# === URL NASA POWER API ===
url = (
    f"https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters=T2M,RH2M,PS,WS2M"
    f"&community=AG"
    f"&longitude={LONGITUDE}&latitude={LATITUDE}"
    f"&start={start_str}&end={end_str}"
    f"&format=JSON"
)

print(f"🔄 Mengambil data NASA dari {start_str} sampai {end_str}...")
response = requests.get(url)
data = response.json()

# === Cek hasil response ===
if "properties" not in data or "parameter" not in data["properties"]:
    raise Exception("Gagal mengambil data dari NASA API.")

params = data["properties"]["parameter"]

# === Simpan data baru ke database ===
count_new = 0
for date_str in params["T2M"].keys():
    date_obj = datetime.strptime(date_str, "%Y%m%d").date()
    t2m = params["T2M"].get(date_str)
    rh2m = params["RH2M"].get(date_str)
    ps = params["PS"].get(date_str)
    ws2m = params["WS2M"].get(date_str)

    cursor.execute("""
        INSERT OR IGNORE INTO weather_history (date, temperature, humidity, pressure, wind_speed, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date_obj, t2m, rh2m, ps, ws2m, "NASA"))
    count_new += 1

conn.commit()
conn.close()

print(f"✅ Update selesai! {count_new} data baru ditambahkan dari NASA.")
