import requests
import sqlite3
from datetime import datetime, timedelta
import os

from backend.config_location import (
    LATITUDE,
    LONGITUDE,
    LOCATION_NAME,
    INVALID_VALUES,
    NASA_SAFE_LAG_DAYS
)


# === Path DB ===
base_dir = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(base_dir, "data", "weather_data.db")

# === Rentang waktu ===
end_date = datetime.now().date() - timedelta(days=NASA_SAFE_LAG_DAYS)
start_date = end_date - timedelta(days=3 * 365)

start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

print(f"📍 Lokasi: {LOCATION_NAME} ({LATITUDE}, {LONGITUDE})")
print(f"📥 Ambil NASA: {start_str} s/d {end_str}")
print(f"📦 DB: {DB_PATH}")

URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?start={start_str}&end={end_str}"
    f"&latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&parameters=T2M,RH2M,PS,WS2M"
    f"&community=AG&format=JSON"
)

response = requests.get(URL, timeout=60)
response.raise_for_status()
data = response.json()
params = data["properties"]["parameter"]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

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

def clean(v):
    if v is None or v in INVALID_VALUES:
        return None
    return v

count = 0
for date_str in params["T2M"]:
    date_db = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    t2m = clean(params["T2M"].get(date_str))
    rh2m = clean(params["RH2M"].get(date_str))
    ps = clean(params["PS"].get(date_str))
    ws2m = clean(params["WS2M"].get(date_str))

    cursor.execute("""
        INSERT INTO weather_history (date, temperature, humidity, pressure, wind_speed, source)
        VALUES (?, ?, ?, ?, ?, 'NASA')
        ON CONFLICT(date) DO UPDATE SET
            temperature=excluded.temperature,
            humidity=excluded.humidity,
            pressure=excluded.pressure,
            wind_speed=excluded.wind_speed
    """, (date_db, t2m, rh2m, ps, ws2m))

    count += 1

conn.commit()
conn.close()

print(f"✅ History selesai: {count} hari")
