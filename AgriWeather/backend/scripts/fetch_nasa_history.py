import requests
import sqlite3
from datetime import datetime, timedelta

# === Konfigurasi Lokasi & Database ===
LAT = -7.9227   # Koordinat Stasiun Klimatologi Karangploso, Malang
LON = 112.6255
DB_PATH = "backend/data/weather_data.db"

# === Hitung Rentang Waktu (3 tahun ke belakang dari kemarin) ===
end_date = datetime.now().date() - timedelta(days=1)
start_date = end_date - timedelta(days=3 * 365)

start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

print(f"Mengambil data NASA dari {start_str} sampai {end_str}...")

# === URL API NASA POWER ===
URL = (
    f"https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"start={start_str}&end={end_str}&latitude={LAT}&longitude={LON}"
    f"&parameters=T2M,RH2M,PS,WS2M&community=RE&format=JSON"
)

# === Ambil Data dari NASA POWER ===
response = requests.get(URL)
if response.status_code != 200:
    raise Exception(f"Gagal mengambil data dari NASA: {response.status_code}")

data = response.json()
params = data["properties"]["parameter"]

# === Koneksi ke Database ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# === Buat Tabel (jika belum ada) ===
cursor.execute("""
CREATE TABLE IF NOT EXISTS weather_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE,
    temperature REAL,
    humidity REAL,
    pressure REAL,
    wind_speed REAL,
    source TEXT
)
""")

# === Simpan Data NASA ke Database ===
count = 0
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
    count += 1

conn.commit()
conn.close()

print(f"✅ Data NASA 3 tahun ke belakang berhasil dimasukkan ke database! ({count} baris)")
