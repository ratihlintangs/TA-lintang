from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

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


# ✅ INI YANG DICARI UVICORN: variabel bernama `app` harus ada di level module
app = FastAPI(
    title="AgriWeather API",
    description="API untuk prediksi cuaca pertanian.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(weather.router)
app.include_router(predict.router)
app.include_router(evaluation_history_router)


@app.get("/")
def read_root():
    return {"message": "Selamat datang di AgriWeather API! Akses /docs untuk dokumentasi."}
