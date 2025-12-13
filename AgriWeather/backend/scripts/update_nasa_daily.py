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


base_dir = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(base_dir, "data", "weather_data.db")
LOG_PATH = os.path.join(base_dir, "nasa_update.log")

def log(msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {msg}\n")
    print(msg)

log(f"📍 Lokasi: {LOCATION_NAME}")

today = datetime.now().date()
end_date = today - timedelta(days=NASA_SAFE_LAG_DAYS)
start_date = end_date - timedelta(days=7)

start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

log(f"🔄 Update harian NASA {start_str} → {end_str}")

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters=T2M,RH2M,PS,WS2M"
    f"&community=AG"
    f"&latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start={start_str}&end={end_str}"
    f"&format=JSON"
)

r = requests.get(url, timeout=60)
r.raise_for_status()
params = r.json()["properties"]["parameter"]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

upsert = 0
skip = 0

for d in params["T2M"]:
    date_db = datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")

    vals = (
        params["T2M"].get(d),
        params["RH2M"].get(d),
        params["PS"].get(d),
        params["WS2M"].get(d),
    )

    if any(v is None or v in INVALID_VALUES for v in vals):
        skip += 1
        continue

    cursor.execute("""
        INSERT INTO weather_history (date, temperature, humidity, pressure, wind_speed, source)
        VALUES (?, ?, ?, ?, ?, 'NASA')
        ON CONFLICT(date) DO UPDATE SET
            temperature=excluded.temperature,
            humidity=excluded.humidity,
            pressure=excluded.pressure,
            wind_speed=excluded.wind_speed
    """, (date_db, *vals))

    upsert += 1

conn.commit()
conn.close()

log(f"✅ Selesai | Upsert: {upsert} | Skip: {skip}")
