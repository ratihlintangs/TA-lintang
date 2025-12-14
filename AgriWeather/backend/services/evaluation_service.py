from typing import List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException

class EvaluationService:
    """
    EvaluationService menerima predict_service dari luar (dependency injection),
    sehingga tidak perlu import PredictService di sini.
    Ini menghindari circular import dan error module saat reload/switch.
    """

    def __init__(self, predict_service):
        if predict_service is None:
            raise ValueError("predict_service tidak boleh None.")
        self.predict_service = predict_service

    def evaluate_all(self, db: Session, test_days: int = 7) -> List[Dict[str, Any]]:
        """
        Endpoint evaluasi.
        NOTE:
        - Karena aku belum lihat evaluasi rolling kamu yang lama, fungsi ini aman dulu (placeholder).
        - Kalau kamu mau aku refactor evaluasi rolling yang kamu sudah punya agar kompatibel v1/v2,
          kirim isi evaluation_service lama kamu, nanti aku balikin versi finalnya.
        """
        try:
            if test_days < 1:
                raise HTTPException(status_code=400, detail="test_days minimal 1.")

            # TODO: Tempel logic evaluasi kamu di sini.
            # Saat butuh prediksi, panggil:
            # preds = self.predict_service.generate_weather_prediction(db=db, days_ahead=test_days)
            #
            # Lalu bandingkan dengan ground truth dari DB untuk tanggal-tanggal target.

            return [
                {
                    "status": "ok",
                    "message": (
                        "EvaluationService sudah aktif dan mengikuti mode v1/v2 melalui inject predict_service. "
                        "Tempel logic evaluasi rolling kamu di evaluate_all()."
                    ),
                    "test_days": test_days,
                }
            ]

        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ ERROR evaluate_all: {e}")
            raise HTTPException(status_code=500, detail=f"Kesalahan saat evaluasi model: {e}")
