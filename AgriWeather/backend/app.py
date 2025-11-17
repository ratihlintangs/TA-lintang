from fastapi import FastAPI
# IMPORT BARU UNTUK FILE STATIS
from fastapi.staticfiles import StaticFiles 
from starlette.responses import HTMLResponse 

from database import Base, engine, WeatherHistoryModel
from routers import weather, prediction
from fastapi.middleware.cors import CORSMiddleware 

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="AGRIWEATHER API",
    description="API untuk mengelola data historis cuaca dan prediksi AR untuk area pertanian.",
    version="1.0.0",
)

# --- KONFIGURASI CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Inisialisasi Database (Membuat tabel jika belum ada) ---
try:
    Base.metadata.create_all(bind=engine)
    print("Database tables ensured.")
except Exception as e:
    print(f"Error creating database tables: {e}")

# --- Registrasi Routers ---
app.include_router(weather.router)
app.include_router(prediction.router)

# --- KONFIGURASI FILE STATIS (FRONTEND) ---
# 1. Mount StaticFiles: Melayani semua file di folder 'static'
#    Pastikan Anda membuat folder 'static' dan memindahkan index.html ke sana.
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Endpoint Root untuk Melayani index.html
#    Saat pengguna mengakses http://127.0.0.1:8000/, mereka akan disajikan index.html.
@app.get("/", tags=["Root"], response_class=HTMLResponse)
async def read_root():
    """Endpoint root yang melayani frontend index.html."""
    try:
        # Baca konten index.html dari folder static
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse("<h1>404 Not Found</h1><p>Frontend file (static/index.html) not found. Pastikan index.html ada di folder 'static'.</p>", status_code=404)


# --- Catatan untuk menjalankan aplikasi: ---
# Anda dapat menjalankan aplikasi ini dari terminal menggunakan Uvicorn:
# uvicorn app:app --reload