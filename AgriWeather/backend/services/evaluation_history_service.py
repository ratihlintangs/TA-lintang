import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.database import load_data_from_db
from backend.models import EvaluationHistoryModel

from backend.services.predict_service import PredictService


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # handle kosong
    if len(y_true) == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R2 manual (hindari sklearn dependency)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


class EvaluationHistoryService:
    """
    Rolling evaluation untuk periode tertentu, simpan hasil ke DB (SQLite).
    """

    def __init__(self):
        self.predict_service = PredictService()

    def _generate_forecast_as_of(self, db: Session, as_of_date: datetime.date, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Prediksi 7 hari ke depan dengan asumsi 'hari terakhir observasi' = as_of_date.
        NOTE: Ini mirip PredictService yang kamu punya, tapi outputnya harus mulai as_of_date+1,
              bukan 'today'. Jadi kita lakukan trick: sementara set 'today' = as_of_date+1
              dengan cara memanggil versi predict_service yang sudah ada, tetapi kita butuh
              kontrol output date.

        Solusi paling simpel & aman:
        - Ambil semua data dari DB
        - Pakai mekanisme model yang sama, tapi kita "potong" data sampai as_of_date
        - Jalankan prediksi rekursif untuk days_ahead mulai as_of_date+1
        """
        # load data full
        df_raw = load_data_from_db(db)
        if df_raw.empty:
            raise HTTPException(status_code=404, detail="Data historis kosong dari database.")

        # rapiin
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"]).sort_values("date")

        # potong histori sampai as_of_date
        df_hist = df_raw[df_raw["date"].dt.date <= as_of_date].copy()
        if df_hist.empty:
            raise HTTPException(status_code=404, detail=f"Tidak ada data historis <= {as_of_date}")

        # Kita “re-use” internal predict logic dengan memanggil fungsi private-nya secara aman:
        # Cara paling aman tanpa refactor besar: kita bikin instance baru sementara dan pakai
        # metode public generate_weather_prediction, tetapi kita butuh start dari as_of_date+1.
        #
        # Jadi kita buat forecast manual di sini untuk 4 target dengan memanggil helper di PredictService.
        ps = self.predict_service

        # target columns harus sama dengan predict_service
        target_cols = ["temperature", "humidity", "pressure", "wind_speed"]

        base_start_date = as_of_date + datetime.timedelta(days=1)
        forecast_dates = [base_start_date + datetime.timedelta(days=i) for i in range(days_ahead)]
        forecast_results = pd.DataFrame({"date": pd.to_datetime(forecast_dates)})

        # ambil aset model dari PredictService
        for target_col in target_cols:
            assets = ps.model_assets.get(target_col)
            if not assets:
                raise HTTPException(status_code=500, detail=f"Aset model {target_col} tidak ditemukan.")

            model = assets["model"]
            scaler_X = assets["scaler_X"]
            scaler_y = assets["scaler_y"]

            # gunakan helper smoothing + ambil 30 nilai terakhir (dari histori yg dipotong)
            historic_values = ps._prepare_initial_data(df_hist, target_col)

            preds = []
            for i in range(days_ahead):
                current_date = forecast_dates[i]

                lag_7 = historic_values[-7]
                lag_14 = historic_values[-14]
                lag_30 = historic_values[-30]

                dayofyear = current_date.timetuple().tm_yday
                sin_day = np.sin(2 * np.pi * dayofyear / 365.25)
                cos_day = np.cos(2 * np.pi * dayofyear / 365.25)
                dayofweek = current_date.weekday()
                is_weekday = 1 if dayofweek < 5 else 0

                X_input = np.array([[lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]])
                X_scaled = scaler_X.transform(X_input)
                y_scaled_pred = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))[0][0]

                y_pred = float(y_pred)
                preds.append(round(y_pred, 2))
                historic_values.append(y_pred)

            forecast_results[target_col] = preds

        # format output sama dengan API forecast kamu
        out = []
        for _, row in forecast_results.iterrows():
            out.append({
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "temperature_c": row.get("temperature", None),
                "humidity_percent": row.get("humidity", None),
                "pressure_hpa": row.get("pressure", None),
                "wind_speed_kmh": row.get("wind_speed", None),
            })
        return out

    def run_backfill(
        self,
        db: Session,
        start_date: datetime.date,
        end_date: datetime.date,
        step_days: int = 7,
        horizon_days: int = 7,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Jalankan rolling evaluation dari start_date..end_date, tiap step_days.
        Simpan ke tabel evaluation_history.
        """
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="end_date harus >= start_date")

        # load semua data sekali untuk ambil aktual
        df_all = load_data_from_db(db)
        if df_all.empty:
            raise HTTPException(status_code=404, detail="Data historis kosong dari database.")

        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all.dropna(subset=["date"]).sort_values("date")
        df_all["d"] = df_all["date"].dt.date

        params_map = {
            "temperature": ("temperature_c", "temperature"),
            "humidity": ("humidity_percent", "humidity"),
            "pressure": ("pressure_hpa", "pressure"),
            "wind_speed": ("wind_speed_kmh", "wind_speed"),
        }

        created = 0
        skipped = 0
        eval_dates = []

        cur = start_date
        while cur <= end_date:
            eval_dates.append(cur)
            cur = cur + datetime.timedelta(days=step_days)

        for as_of in eval_dates:
            # cek apakah sudah ada
            existing = db.query(EvaluationHistoryModel).filter(
                EvaluationHistoryModel.eval_date == as_of,
                EvaluationHistoryModel.horizon_days == horizon_days,
            ).count()

            if existing > 0 and not overwrite:
                skipped += 1
                continue

            if overwrite and existing > 0:
                db.query(EvaluationHistoryModel).filter(
                    EvaluationHistoryModel.eval_date == as_of,
                    EvaluationHistoryModel.horizon_days == horizon_days,
                ).delete()

            # prediksi 7 hari setelah as_of
            preds = self._generate_forecast_as_of(db, as_of_date=as_of, days_ahead=horizon_days)
            pred_df = pd.DataFrame(preds)
            pred_df["d"] = pd.to_datetime(pred_df["date"]).dt.date

            # ambil aktual dari DB untuk horizon tersebut
            actual_start = as_of + datetime.timedelta(days=1)
            actual_end = as_of + datetime.timedelta(days=horizon_days)

            actual_df = df_all[(df_all["d"] >= actual_start) & (df_all["d"] <= actual_end)].copy()
            if actual_df.empty:
                # tidak ada aktual, skip
                skipped += 1
                continue

            # join on date
            merged = pd.merge(
                pred_df,
                actual_df,
                how="inner",
                left_on="d",
                right_on="d",
                suffixes=("_pred", "_act"),
            )

            # hitung metric per parameter
            for param, (api_key, db_col) in params_map.items():
                # actual col = db_col, pred col = api_key
                y_true = pd.to_numeric(merged[db_col], errors="coerce")
                y_pred = pd.to_numeric(merged[api_key], errors="coerce")

                mask = (~y_true.isna()) & (~y_pred.isna())
                y_true = y_true[mask].to_numpy(dtype=float)
                y_pred = y_pred[mask].to_numpy(dtype=float)

                m = _metrics(y_true, y_pred)

                rec = EvaluationHistoryModel(
                    eval_date=as_of,
                    horizon_days=horizon_days,
                    parameter=param,
                    mse=m["mse"],
                    rmse=m["rmse"],
                    mae=m["mae"],
                    r2=m["r2"],
                    n_samples=int(len(y_true)),
                )
                db.add(rec)
                created += 1

            db.commit()

        return {
            "status": "ok",
            "start_date": str(start_date),
            "end_date": str(end_date),
            "step_days": step_days,
            "horizon_days": horizon_days,
            "records_created": created,
            "eval_points": len(eval_dates),
            "skipped_points": skipped,
        }

    def list_history(
        self,
        db: Session,
        start_date: datetime.date,
        end_date: datetime.date,
        parameter: Optional[str] = None,
        horizon_days: int = 7,
    ) -> List[Dict[str, Any]]:
        q = db.query(EvaluationHistoryModel).filter(
            EvaluationHistoryModel.eval_date >= start_date,
            EvaluationHistoryModel.eval_date <= end_date,
            EvaluationHistoryModel.horizon_days == horizon_days,
        )
        if parameter:
            q = q.filter(EvaluationHistoryModel.parameter == parameter)

        rows = q.order_by(EvaluationHistoryModel.eval_date.asc(), EvaluationHistoryModel.parameter.asc()).all()
        out = []
        for r in rows:
            out.append({
                "eval_date": r.eval_date.strftime("%Y-%m-%d"),
                "horizon_days": r.horizon_days,
                "parameter": r.parameter,
                "MSE": r.mse,
                "RMSE": r.rmse,
                "MAE": r.mae,
                "R2": r.r2,
                "n_samples": r.n_samples,
            })
        return out
