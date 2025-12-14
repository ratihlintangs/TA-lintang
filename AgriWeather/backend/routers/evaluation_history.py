import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from backend.database import get_db
from backend.services.evaluation_history_service import EvaluationHistoryService

router = APIRouter(prefix="/weather/evaluation-history", tags=["Evaluation History"])

svc = EvaluationHistoryService()

@router.post("/backfill")
@router.get("/backfill")
def backfill_history(
    start: str = Query("2025-09-01"),
    end: str = Query("2025-11-30"),
    step_days: int = Query(7, description="Evaluasi tiap berapa hari (7=mingguan)"),
    horizon_days: int = Query(7, description="Horizon prediksi untuk evaluasi"),
    overwrite: bool = Query(False, description="Jika true, timpa data lama di periode itu"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    start_date = datetime.date.fromisoformat(start)
    end_date = datetime.date.fromisoformat(end)

    return svc.run_backfill(
        db=db,
        start_date=start_date,
        end_date=end_date,
        step_days=step_days,
        horizon_days=horizon_days,
        overwrite=overwrite,
    )


@router.get("/list")
def list_history(
    start: str = Query("2025-09-01"),
    end: str = Query("2025-11-30"),
    parameter: Optional[str] = Query(None, description="temperature/humidity/pressure/wind_speed"),
    horizon_days: int = Query(7),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    start_date = datetime.date.fromisoformat(start)
    end_date = datetime.date.fromisoformat(end)

    return svc.list_history(
        db=db,
        start_date=start_date,
        end_date=end_date,
        parameter=parameter,
        horizon_days=horizon_days,
    )
@router.get("/latest")
def latest_history(
    horizon_days: int = Query(7),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    return svc.get_latest_valid(db=db, horizon_days=horizon_days)


