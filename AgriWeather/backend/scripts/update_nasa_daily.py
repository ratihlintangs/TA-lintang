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
log_path = os.path.join(base_dir, "nasa_update.log")

def log(msg):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {msg}\n")
    print(msg)

# === Validasi Nilai NASA ===
def is_invalid(value):
    if value is None:
        return True
    if isinstance(value, str):
        if value.strip() in ["-999", "-999.0", "-9999"]:
            return True
    if value in [-999, -999.0, -9999]:
        return True
    return False

# === Koneksi ke database ===
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# === Cari tanggal terakhir di database ===
cursor.execute("SELECT MAX(date) FROM weather_history WHERE source = 'NASA'")
last_date_row = cursor.fetchone()

if last_date_row and last_date_row[0]:
    last_date = datetime.strptime(last_date_row[0], "%Y-%m-%d").date()
else:
    raise Exception("Database belum memiliki data NASA. Jalankan fetch_nasa_history.py dulu.")

# === Tentukan rentang update ===
start_date = last_date + timedelta(days=1)
end_date = datetime.now().date() - timedelta(days=2)  # aman H-2 NASA release

if start_date > end_date:
    log("⏸️ Tidak ada data baru untuk diperbarui.")
    conn.close()
    exit()

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

log(f"🔄 Mengambil data NASA dari {start_str} sampai {end_str}...")
response = requests.get(url)

if response.status_code != 200:
    log(f"❌ NASA API error: {response.status_code}")
    exit()

data = response.json()

# === Validasi struktur response ===
if "properties" not in data or "parameter" not in data["properties"]:
    log("❌ Response NASA tidak valid.")
    exit()

params = data["properties"]["parameter"]

# Pastikan parameter tersedia
for key in ["T2M", "RH2M", "PS", "WS2M"]:
    if key not in params:
        log(f"⚠️ Parameter {key} tidak tersedia dalam response NASA (kosong).")
        params[key] = {}

count_new = 0

# === Simpan data ===
for date_str in params.get("T2M", {}):
    date_obj = datetime.strptime(date_str, "%Y%m%d").date()

    t2m = params["T2M"].get(date_str)
    rh2m = params["RH2M"].get(date_str)
    ps = params["PS"].get(date_str)
    ws2m = params["WS2M"].get(date_str)

    # === Skip jika data belum valid ===
    if any(is_invalid(v) for v in [t2m, rh2m, ps, ws2m]):
        log(f"⚠️ {date_obj} dilewati (data belum lengkap / masih -999).")
        continue

    cursor.execute("""
        INSERT OR IGNORE INTO weather_history 
        (date, temperature, humidity, pressure, wind_speed, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date_obj, t2m, rh2m, ps, ws2m, "NASA"))

    count_new += 1
    log(f"✔ Data {date_obj} ditambahkan.")

conn.commit()
conn.close()

log(f"✅ Update selesai! {count_new} data baru ditambahkan.")
