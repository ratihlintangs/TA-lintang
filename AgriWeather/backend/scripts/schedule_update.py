import requests
import sqlite3
from datetime import datetime, timedelta
import os
import sys
import traceback

# ===========================================================
#  FIX PATH agar kompatibel dengan Task Scheduler
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/scripts
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # backend/

# === Lokasi default DB ===
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "weather_data.db")

# === Lokasi log ===
LOG_PATH = os.path.join(PROJECT_ROOT, "nasa_update.log")


# ===========================================================
#  Logging helper
# ===========================================================
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] {msg}"
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# ===========================================================
#  Auto-detect database path
# ===========================================================
def get_database_path():
    possible_paths = [
        DEFAULT_DB_PATH,
        os.path.join(BASE_DIR, "data", "weather_data.db"),
        os.path.join(PROJECT_ROOT, "..", "data", "weather_data.db"),
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return os.path.abspath(p)

    return None  # biarkan main() yang handle error


# ===========================================================
#  Cek nilai NASA valid / tidak
# ===========================================================
def is_invalid(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ["-999", "-999.0", "-9999"]:
        return True
    if value in [-999, -999.0, -9999]:
        return True
    return False


# ===========================================================
#  MAIN PROCESS
# ===========================================================
def main():
    LATITUDE = -7.9135
    LONGITUDE = 112.6214

    # === Cari path DB yang benar ===
    DB_PATH = get_database_path()
    if not DB_PATH:
        log("❌ Database weather_data.db tidak ditemukan di folder manapun.")
        log(f"Dicari terakhir di: {DEFAULT_DB_PATH}")
        return 1

    log(f"📁 Menggunakan database: {DB_PATH}")

    # ===============================================
    # Koneksi DB
    # ===============================================
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Cari tanggal terakhir NASA
    cursor.execute("SELECT MAX(date) FROM weather_history WHERE source = 'NASA'")
    last_date_row = cursor.fetchone()

    if not last_date_row or not last_date_row[0]:
        log("❌ Database belum punya data awal NASA. Jalankan fetch_nasa_history.py dulu.")
        return 1

    last_date = datetime.strptime(last_date_row[0], "%Y-%m-%d").date()

    # Tentukan rentang update
    start_date = last_date + timedelta(days=1)
    end_date = datetime.now().date() - timedelta(days=2)  # H-2

    if start_date > end_date:
        log("⏸ Tidak ada data baru.")
        return 0

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # NASA API
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M,PS,WS2M"
        f"&community=AG"
        f"&longitude={LONGITUDE}&latitude={LATITUDE}"
        f"&start={start_str}&end={end_str}"
        f"&format=JSON"
    )

    log(f"🔄 Fetch NASA {start_str} → {end_str}")

    response = requests.get(url)
    if response.status_code != 200:
        log(f"❌ NASA API error: HTTP {response.status_code}")
        return 1

    data = response.json()
    if "properties" not in data or "parameter" not in data["properties"]:
        log("❌ NASA response corrupt / tidak lengkap.")
        return 1

    params = data["properties"]["parameter"]

    # Pastikan parameter lengkap
    for p in ["T2M", "RH2M", "PS", "WS2M"]:
        if p not in params:
            log(f"⚠ Parameter {p} kosong, diisi dict kosong.")
            params[p] = {}

    count_new = 0

    # Insert data
    for date_str in params["T2M"]:
        date_obj = datetime.strptime(date_str, "%Y%m%d").date()

        t2m = params["T2M"].get(date_str)
        rh2m = params["RH2M"].get(date_str)
        ps = params["PS"].get(date_str)
        ws2m = params["WS2M"].get(date_str)

        # Skip jika ada nilai buruk
        if any(is_invalid(v) for v in [t2m, rh2m, ps, ws2m]):
            log(f"⚠ {date_obj} dilewati (data NASA masih -999 / incomplete).")
            continue

        cursor.execute("""
            INSERT OR IGNORE INTO weather_history 
            (date, temperature, humidity, pressure, wind_speed, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (date_obj, t2m, rh2m, ps, ws2m, "NASA"))

        count_new += 1
        log(f"✔ Insert {date_obj}")

    conn.commit()
    conn.close()

    log(f"🎉 Update selesai! Total data baru: {count_new}")
    return 0


# ===========================================================
#  ENTRY POINT (untuk Task Scheduler)
# ===========================================================
if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception as e:
        log("❌ ERROR FATAL:")
        log(str(e))
        log(traceback.format_exc())
        sys.exit(1)
