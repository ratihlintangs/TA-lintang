import os
import sqlite3

# --- Tentukan path database (otomatis tersimpan di folder data/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # ke folder backend/
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "weather_data.db")

# Pastikan folder data/ ada
os.makedirs(DATA_DIR, exist_ok=True)

# Koneksi ke SQLite (akan otomatis membuat file kalau belum ada)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# === Buat tabel-tabel utama ===
cursor.executescript("""
-- 1. Data historis (NASA + OpenWeather)
CREATE TABLE IF NOT EXISTS weather_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,                -- NASA atau OpenWeather
    date DATE NOT NULL,                  -- tanggal observasi
    latitude REAL NOT NULL,              -- lintang lokasi
    longitude REAL NOT NULL,             -- bujur lokasi
    temperature REAL,                    -- suhu rata-rata (°C)
    humidity REAL,                       -- kelembapan rata-rata (%)
    pressure REAL,                       -- tekanan udara (hPa)
    wind_speed REAL,                     -- kecepatan angin (m/s)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Hasil prediksi model
CREATE TABLE IF NOT EXISTS weather_prediction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_predicted DATE NOT NULL,
    model_type TEXT NOT NULL,            -- jenis model (Linear, MLP, Polynomial)
    predicted_temperature REAL,
    predicted_humidity REAL,
    predicted_pressure REAL,
    predicted_wind_speed REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Performa model
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    r2_score REAL,
    mae REAL,
    rmse REAL,
    date_evaluated DATE DEFAULT CURRENT_DATE,
    notes TEXT
);

-- 4. (Opsional) Data lokasi
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL
);
""")

# Simpan dan tutup koneksi
conn.commit()
conn.close()

print(f"✅ Database berhasil dibuat di: {DB_PATH}")
print("   Semua tabel siap digunakan untuk NASA dan OpenWeather.")
