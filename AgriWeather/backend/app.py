from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

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


app = FastAPI(
    title="AgriWeather API",
    description="API untuk prediksi cuaca pertanian.",
    version="1.0.0",
    lifespan=lifespan,
)

# ===================== PATHS =====================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
MANIFEST_FILE = STATIC_DIR / "manifest.json"
ICON_192 = STATIC_DIR / "icons" / "icon-192.png"

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== ROUTERS =====================
app.include_router(weather.router)
app.include_router(predict.router)
app.include_router(evaluation_history_router)

# ===================== STATIC =====================
if not STATIC_DIR.is_dir():
    raise RuntimeError(
        f"Static directory not found: {STATIC_DIR}\n"
        f"Pastikan folder 'backend/static' ada dan berisi index.html, manifest.json, icons/"
    )

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ===================== ROOT =====================
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_index(request: Request):
    if INDEX_FILE.is_file():
        return FileResponse(str(INDEX_FILE))
    return JSONResponse(
        status_code=500,
        content={"message": "index.html tidak ditemukan. Pastikan ada di backend/static/index.html"},
    )

# ===================== MANIFEST =====================
@app.api_route("/manifest.json", methods=["GET", "HEAD"], include_in_schema=False)
def serve_manifest_json():
    if MANIFEST_FILE.is_file():
        return FileResponse(str(MANIFEST_FILE), media_type="application/manifest+json")
    return JSONResponse(status_code=404, content={"message": "manifest.json tidak ditemukan"})

@app.api_route("/manifest.webmanifest", methods=["GET", "HEAD"], include_in_schema=False)
def serve_manifest_webmanifest():
    if MANIFEST_FILE.is_file():
        return FileResponse(str(MANIFEST_FILE), media_type="application/manifest+json")
    return JSONResponse(status_code=404, content={"message": "manifest.json tidak ditemukan"})

# ===================== FAVICON =====================
@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
def favicon():
    if ICON_192.is_file():
        return FileResponse(str(ICON_192))
    return JSONResponse(status_code=404, content={"message": "favicon not found"})

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}
