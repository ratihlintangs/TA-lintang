from typing import List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException
import pandas as pd
import numpy as np
import datetime
import math

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.database import load_data_from_db


def _safe_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


class EvaluationService:
    """
    EvaluationService menerima predict_service dari luar (dependency injection),
    sehingga tidak perlu import PredictService di sini.
    """

    def __init__(self, predict_service):
        if predict_service is None:
            raise ValueError("predict_service tidak boleh None.")
        self.predict_service = predict_service

    def evaluate_all(self, db: Session, test_days: int = 7) -> List[Dict[str, Any]]:
        """
        Rolling evaluation (anti-NaN):
        - Evaluasi dilakukan pada N hari terakhir yang punya data aktual lengkap.
        - Untuk tiap parameter: hitung MSE, RMSE, MAE, R2.
        - Jika data tidak cukup / variansi nol / ada NaN -> metrik dibuat None (bukan NaN).
        """
        if test_days < 1:
            raise HTTPException(status_code=400, detail="test_days minimal 1.")

        # 1) Load historis
        df = load_data_from_db(db)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Data historis kosong dari database.")

        if "date" not in df.columns:
            raise HTTPException(status_code=500, detail="Kolom 'date' tidak ditemukan pada data historis.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        required = ["temperature", "humidity", "pressure", "wind_speed"]
        for c in required:
            if c not in df.columns:
                raise HTTPException(status_code=500, detail=f"Kolom '{c}' tidak ditemukan pada data historis.")

        # coerce numeric
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 2) Tentukan window evaluasi: N hari terakhir yang punya actual lengkap
        df_valid = df.dropna(subset=required).copy()
        if df_valid.empty:
            raise HTTPException(status_code=404, detail="Tidak ada data aktual lengkap untuk evaluasi.")

        last_actual_date = df_valid["date"].max().date()
        start_date = last_actual_date - datetime.timedelta(days=test_days - 1)

        # actual window
        df_window = df_valid[(df_valid["date"].dt.date >= start_date) & (df_valid["date"].dt.date <= last_actual_date)].copy()
        if df_window.empty or len(df_window) < 3:
            # minimal 3 titik supaya metrik bermakna
            return [
                {"parameter": p, "MSE": None, "RMSE": None, "MAE": None, "R2": None,
                 "note": "Data aktual tidak cukup untuk evaluasi."}
                for p in required
            ]

        # 3) Generate prediksi untuk periode yang sama
        #    Kita minta predict_service menghasilkan test_days ke depan dari "posisi sekarang".
        #    Tapi karena predict_service kamu memotong output mulai dari today, kita ambil prediksi lalu match by date.
        preds = self.predict_service.generate_weather_prediction(db=db, days_ahead=test_days)
        if not preds:
            return [
                {"parameter": p, "MSE": None, "RMSE": None, "MAE": None, "R2": None,
                 "note": "Prediksi kosong, evaluasi tidak dapat dilakukan."}
                for p in required
            ]

        df_pred = pd.DataFrame(preds)
        df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")
        df_pred = df_pred.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # map key forecast -> kolom aktual
        mapping = {
            "temperature": "temperature_c",
            "humidity": "humidity_percent",
            "pressure": "pressure_hpa",
            "wind_speed": "wind_speed_kmh",
        }

        # 4) Match by date intersection
        # actual date column: df_window['date']
        # pred date column: df_pred['date']
        # Use date only (no time)
        df_window["d"] = df_window["date"].dt.date
        df_pred["d"] = df_pred["date"].dt.date

        merged = pd.merge(df_window, df_pred, on="d", how="inner", suffixes=("_act", "_pred"))
        if merged.empty or len(merged) < 3:
            return [
                {"parameter": p, "MSE": None, "RMSE": None, "MAE": None, "R2": None,
                 "note": "Tidak ada irisan tanggal actual vs prediksi (kemungkinan NASA delay atau slicing forecast)."}
                for p in required
            ]

        results: List[Dict[str, Any]] = []

        for p in required:
            pred_col = mapping[p]
            y_true = merged[p].to_numpy(dtype=float)
            y_pred = pd.to_numeric(merged[pred_col], errors="coerce").to_numpy(dtype=float)

            # drop NaN pairs
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) < 3:
                results.append({"parameter": p, "MSE": None, "RMSE": None, "MAE": None, "R2": None})
                continue

            mse = mean_squared_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_true, y_pred)

            # safe R2
            if float(np.std(y_true)) == 0.0:
                r2 = None
            else:
                try:
                    r2 = r2_score(y_true, y_pred)
                except Exception:
                    r2 = None

            results.append(
                {
                    "parameter": p,
                    "MSE": _safe_float(mse),
                    "RMSE": _safe_float(rmse),
                    "MAE": _safe_float(mae),
                    "R2": _safe_float(r2) if r2 is not None else None,
                    "range_start": str(start_date),
                    "range_end": str(last_actual_date),
                    "n_points": int(len(y_true)),
                }
            )

        return results
