from fastapi import APIRouter
from services.nasa_service import run_update_script, read_log_tail

router = APIRouter()


@router.post("/update")
def update_nasa():
    """Trigger the existing update_nasa_daily.py script (non-blocking wrapper)."""
    ok, out = run_update_script()
    # Harus 4 spasi ke dalam
    return {"success": ok, "message": out}


@router.get("/log")
def nasa_log():
    text = read_log_tail()
    # Harus 4 spasi ke dalam
    return {"log_tail": text}