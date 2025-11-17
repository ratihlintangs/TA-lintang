import os
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import random

# Impor konfigurasi database dan model dari file lain
# PASTIKAN SEMUA INI ADA DI DATABASE.PY
from database import engine, Base, WeatherHistoryModel, SessionLocal 

# --- Fungsi untuk mengisi database dengan data palsu ---
# ... (Fungsi seed_database tidak berubah)

def seed_database(db: Session, num_days: int = 30):
    """
    Mengisi database dengan catatan cuaca palsu untuk N hari terakhir.
    """
    print(f"Mulai mengisi database dengan data untuk {num_days} hari terakhir...")
    
    # Hapus semua data lama (opsional, untuk memastikan fresh start)
    # db.query(WeatherHistoryModel).delete()
    # db.commit()

    # Dapatkan tanggal hari ini (UTC)
    current_date = datetime.now().date()
    
    # Tentukan tanggal mulai (misalnya 30 hari yang lalu)
    start_date = current_date - timedelta(days=num_days - 1)

    for i in range(num_days):
        date_to_seed = start_date + timedelta(days=i)
        
        # Simulasi data yang realistis (dengan sedikit variasi)
        temp_base = 30 + (i % 5) * 0.2 + random.uniform(-1, 1) # ~30-32 C
        humidity_base = 75 + (i % 7) * 0.5 + random.uniform(-2, 2) # ~75-80%
        pressure_base = 1010 + (i % 3) * 0.1 + random.uniform(-0.5, 0.5) # ~1010 hPa
        wind_base = 5 + (i % 4) * 0.3 + random.uniform(-1, 1) # ~5-6 Km/j

        # Buat objek data baru
        db_data = WeatherHistoryModel(
            date=date_to_seed.strftime('%Y-%m-%d'),
            temperature=round(temp_base, 2),
            humidity=round(humidity_base, 2),
            pressure=round(pressure_base, 2),
            wind_speed=round(wind_base, 2),
            source="Mock Data Seeder"
        )
        db.add(db_data)
        
    db.commit()
    print(f"Selesai! Berhasil menambahkan {num_days} catatan cuaca ke database.")


if __name__ == "__main__":
    # Pastikan direktori 'data' ada
    os.makedirs('data', exist_ok=True)
    
    # Pastikan tabel dibuat sebelum diisi
    Base.metadata.create_all(bind=engine)
    
    # Dapatkan sesi DB dan jalankan seeder
    db = SessionLocal()
    try:
        seed_database(db, num_days=50) # Mengisi data untuk 50 hari
    except Exception as e:
        print(f"Gagal mengisi database: {e}")
    finally:
        db.close()