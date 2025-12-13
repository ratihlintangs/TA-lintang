import requests
import sqlite3
from datetime import datetime
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

print(f"📍 Lokasi: {LOCATION_NAME}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
SELECT date FROM weather_history
WHERE temperature IS NULL OR humidity IS NULL OR pressure IS NULL OR wind_speed IS NULL
""")

dates = [r[0] for r in cursor.fetchall()]
print(f"🔧 Perlu diperbaiki: {len(dates)} hari")

def fetch_one(date_yyyymmdd):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M,PS,WS2M"
        f"&community=AG"
        f"&latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start={date_yyyymmdd}&end={date_yyyymmdd}"
        f"&format=JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    return (
        p["T2M"].get(date_yyyymmdd),
        p["RH2M"].get(date_yyyymmdd),
        p["PS"].get(date_yyyymmdd),
        p["WS2M"].get(date_yyyymmdd),
    )

for d in dates:
    ds = datetime.strptime(d, "%Y-%m-%d").strftime("%Y%m%d")
    try:
        vals = fetch_one(ds)
        if any(v is None or v in INVALID_VALUES for v in vals):
            print(f"⚠️ {d} masih invalid")
            continue

        cursor.execute("""
            UPDATE weather_history
            SET temperature=?, humidity=?, pressure=?, wind_speed=?
            WHERE date=?
        """, (*vals, d))

        print(f"✅ Update {d}")
    except Exception as e:
        print(f"❌ {d} gagal: {e}")

conn.commit()
conn.close()
