from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend.database import Base, engine
from backend.routers import weather, predict
from backend.routers.evaluation_history import router as evaluation_history_router


def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        print("INFO: Tabel database berhasil dibuat (jika belum ada).")
    except Exception as e:
        print(f"FATAL ERROR: Gagal membuat tabel database: {e}")


@asynccontextmanager
async def lifespan(app_: FastAPI):
    create_tables()
    print("INFO: Aplikasi FastAPI dimulai.")
    yield
    print("INFO: Aplikasi FastAPI dimatikan.")


# ✅ variabel bernama `app` harus ada di level module
app = FastAPI(
    title="AgriWeather API",
    description="API untuk prediksi cuaca pertanian.",
    version="1.0.0",
    lifespan=lifespan,
)

# ======= PATHS (AMAN, ABSOLUT) =======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../AgriWeather/backend
STATIC_DIR = os.path.join(BASE_DIR, "static")                # .../AgriWeather/backend/static
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")


# ======= CORS (opsional) =======
# Kalau frontend kamu diserve dari FastAPI yang sama (root /),
# biasanya CORS tidak diperlukan. Tapi tetap aman disiapkan.
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Tambahkan origin LAN kamu jika perlu (contoh):
    # "http://192.168.1.10:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= ROUTERS (API) =======
app.include_router(weather.router)
app.include_router(predict.router)
app.include_router(evaluation_history_router)

# ======= STATIC (PWA assets) =======
# Pastikan folder backend/static memang ada.
if not os.path.isdir(STATIC_DIR):
    raise RuntimeError(
        f"Static directory not found: {STATIC_DIR}\n"
        f"Pastikan folder 'backend/static' ada dan berisi index.html, manifest.json, icons/"
    )

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ======= WEB ROOT (PWA entry) =======
# Root "/" akan serve index.html agar PWA bisa dibuka sebagai web-app.
@app.get("/", include_in_schema=False)
def serve_index():
    if os.path.isfile(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return {
        "message": "index.html tidak ditemukan. Pastikan ada di backend/static/index.html"
    }


# (Opsional) healthcheck ringan
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}
