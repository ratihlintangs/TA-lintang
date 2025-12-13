import requests
import sqlite3
from datetime import datetime, timedelta
import os
import sys
import traceback

# ===========================================================
# FIX PATH agar kompatibel dengan Task Scheduler
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # backend/scripts
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))   # backend
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))  # AgriWeather root

# Tambahkan project root ke sys.path supaya import backend.* aman
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config_location import (
    LATITUDE,
    LONGITUDE,
    LOCATION_NAME,
    INVALID_VALUES,
    NASA_SAFE_LAG_DAYS
)

# ===========================================================
# Path DB & log
# ===========================================================
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "data", "weather_data.db")
LOG_PATH = os.path.join(BACKEND_DIR, "nasa_update.log")


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] {msg}"
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def is_invalid(v):
    if v is None:
        return True
    if isinstance(v, str) and v.strip() in {"-999", "-999.0", "-9999", "999"}:
        return True
    return v in INVALID_VALUES


def fetch_range(start_yyyymmdd: str, end_yyyymmdd: str):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M,PS,WS2M"
        f"&community=AG"
        f"&longitude={LONGITUDE}&latitude={LATITUDE}"
        f"&start={start_yyyymmdd}&end={end_yyyymmdd}"
        f"&format=JSON"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    params = data.get("properties", {}).get("parameter", None)
    return params


def main():
    log(f"📍 Lokasi: {LOCATION_NAME} (lat={LATITUDE}, lon={LONGITUDE})")
    log(f"📦 DB: {DEFAULT_DB_PATH}")

    if not os.path.exists(DEFAULT_DB_PATH):
        log("❌ Database tidak ditemukan. Jalankan fetch_nasa_history.py dulu.")
        return 1

    # =======================================================
    # Tentukan rentang update (sliding window + H-2)
    # =======================================================
    today = datetime.now().date()
    end_date = today - timedelta(days=NASA_SAFE_LAG_DAYS)   # aman H-2
    start_date = end_date - timedelta(days=7)              # sapu 7 hari terakhir

    if start_date > end_date:
        log("⏸ Rentang update tidak valid.")
        return 0

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    log(f"🔄 Scheduler update NASA: {start_str} → {end_str}")

    params = fetch_range(start_str, end_str)
    if not params:
        log("❌ Response NASA tidak valid (properties.parameter kosong).")
        return 1

    for p in ["T2M", "RH2M", "PS", "WS2M"]:
        if p not in params:
            log(f"⚠ Parameter {p} tidak tersedia, dianggap kosong.")
            params[p] = {}

    # =======================================================
    # Upsert ke DB
    # =======================================================
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cursor = conn.cursor()

    upsert = 0
    skip = 0

    for date_str in params.get("T2M", {}):
        date_db = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

        t2m = params["T2M"].get(date_str)
        rh2m = params["RH2M"].get(date_str)
        ps = params["PS"].get(date_str)
        ws2m = params["WS2M"].get(date_str)

        if any(is_invalid(v) for v in [t2m, rh2m, ps, ws2m]):
            skip += 1
            continue

        cursor.execute("""
            INSERT INTO weather_history (date, temperature, humidity, pressure, wind_speed, source)
            VALUES (?, ?, ?, ?, ?, 'NASA')
            ON CONFLICT(date) DO UPDATE SET
                temperature=excluded.temperature,
                humidity=excluded.humidity,
                pressure=excluded.pressure,
                wind_speed=excluded.wind_speed,
                source=excluded.source
        """, (date_db, t2m, rh2m, ps, ws2m))

        upsert += 1

    conn.commit()
    conn.close()

    log(f"✅ Scheduler selesai | Upsert: {upsert} | Skip (invalid/delay): {skip}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log("❌ ERROR FATAL:")
        log(str(e))
        log(traceback.format_exc())
        sys.exit(1)
