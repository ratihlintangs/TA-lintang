from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List

from backend.database import get_db
from backend.models import WeatherHistoryModel
from backend.services.utils import WeatherHistory  # ✅ FIX DI SINI

router = APIRouter(
    prefix="/weather/history",
    tags=["History"],
)


@router.get("/", response_model=List[WeatherHistory])
def get_historical_data(
    db: Session = Depends(get_db),
    limit: int = Query(10, description="Jumlah catatan terakhir yang akan diambil"),
    skip: int = Query(0, description="Lewati N catatan"),
):
    """
    Mengambil data cuaca historis terbaru dari database.
    """
    try:
        data = (
            db.query(WeatherHistoryModel)
            .order_by(WeatherHistoryModel.id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

        return [WeatherHistory.model_validate(item) for item in data]

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Kesalahan saat mengambil data historis",
        )
