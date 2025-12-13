import sqlite3
import os

from backend.config_location import (
    LATITUDE,
    LONGITUDE,
    LOCATION_NAME,
    INVALID_VALUES,
    NASA_SAFE_LAG_DAYS
)


# === Tentukan path database ===
base_dir = os.path.dirname(os.path.dirname(__file__))  # sesuaikan jika perlu
db_path = os.path.join(base_dir, "data", "weather_data.db")

print(f"📂 Mengakses database: {db_path}")

# === Koneksi ke database ===
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# === List kolom yang ingin dibersihkan ===
columns = ["temperature", "humidity", "pressure", "wind_speed"]

# === Nilai-nilai yang dianggap invalid oleh NASA ===
invalid_values = [-999, -999.0, -9999]

total_cleaned = 0

for col in columns:
    for invalid in invalid_values:
        cursor.execute(
            f"""
            UPDATE weather_history
            SET {col} = NULL
            WHERE {col} = ? AND source = 'NASA';
            """,
            (invalid,)
        )
        cleaned = cursor.rowcount
        total_cleaned += cleaned

        if cleaned > 0:
            print(f"🧹 Membersihkan {cleaned} nilai {invalid} pada kolom '{col}'")

conn.commit()
conn.close()

print(f"\n✅ Pembersihan selesai! Total nilai -999 yang diubah menjadi NULL: {total_cleaned}")
