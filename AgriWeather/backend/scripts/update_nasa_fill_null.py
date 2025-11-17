import requests
import sqlite3
from datetime import datetime
import os

# === Lokasi ===
LATITUDE = -7.9135
LONGITUDE = 112.6214

# === Path DB ===
base_dir = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(base_dir, "data", "weather_data.db")

# === Koneksi DB ===
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# === Ambil semua tanggal NASA yang masih NULL ===
cursor.execute("""
    SELECT date FROM weather_history
    WHERE source = 'NASA' AND (
        temperature IS NULL OR
        humidity IS NULL OR
        pressure IS NULL OR
        wind_speed IS NULL
    )
""")

rows = cursor.fetchall()
dates_to_fix = [r[0] for r in rows]

if not dates_to_fix:
    print("✅ Tidak ada baris NULL yang perlu diperbaiki.")
    conn.close()
    exit()

print(f"🔧 Ditemukan {len(dates_to_fix)} baris NULL yang akan diperbaiki.")
print(dates_to_fix)

for d in dates_to_fix:

    date_str = datetime.strptime(d, "%Y-%m-%d").strftime("%Y%m%d")

    # === API NASA untuk 1 tanggal ===
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M,PS,WS2M"
        f"&community=AG"
        f"&longitude={LONGITUDE}&latitude={LATITUDE}"
        f"&start={date_str}&end={date_str}"
        f"&format=JSON"
    )

    print(f"🔄 Mengambil data NASA untuk {d}...")
    r = requests.get(url)
    data = r.json()

    if "properties" not in data or "parameter" not in data["properties"]:
        print(f"⚠️ Gagal mengambil data NASA untuk {d}. Dilewati.")
        continue

    params = data["properties"]["parameter"]

    # Ambil values
    t2m = params["T2M"].get(date_str)
    rh2m = params["RH2M"].get(date_str)
    ps = params["PS"].get(date_str)
    ws2m = params["WS2M"].get(date_str)

    # Skip jika NASA masih -999
    if any(v in [-999, -999.0, -9999] for v in [t2m, rh2m, ps, ws2m]):
        print(f"⚠️ Data NASA untuk {d} masih -999. Dilewati.")
        continue

    # UPDATE row
    cursor.execute("""
        UPDATE weather_history
        SET temperature = ?, humidity = ?, pressure = ?, wind_speed = ?
        WHERE date = ? AND source = 'NASA'
    """, (t2m, rh2m, ps, ws2m, d))

    print(f"✅ Berhasil update {d}")

conn.commit()
conn.close()

print("\n🎉 Semua baris NULL sudah dicoba diperbarui.")