import os
import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException
import math


from backend.database import load_data_from_db
from backend.models import EvaluationHistoryModel

MODEL_MODE = os.getenv("AGRI_MODEL_MODE", "v2").strip().lower()
if MODEL_MODE == "v1":
    from backend.services.predict_service_v1 import PredictService
else:
    from backend.services.predict_service_v2 import PredictService


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ========= helpers (independen dari PredictService private methods) =========

def _numeric_clean(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].copy()
    s = s.replace("-", np.nan)
    s = pd.to_numeric(s, errors="coerce")
    return s


def _abs_smooth_tail_30(df_hist: pd.DataFrame, target_col: str) -> List[float]:
    """
    Ambil 30 nilai terakhir dari seri ABS yang sudah smoothing rolling window=3 center=True.
    Ini dipakai untuk model absolute (v1) dan juga sebagai base reconstruct delta.
    """
    df = df_hist.copy()
    df[target_col] = _numeric_clean(df, target_col)
    df = df.dropna(subset=["date", target_col]).sort_values("date")

    smooth_col = f"{target_col}_smooth"
    df[smooth_col] = df[target_col].rolling(window=3, center=True).mean()
    df = df.dropna(subset=[smooth_col])

    series = df[smooth_col].reset_index(drop=True)
    if len(series) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"Data historis tidak cukup untuk {target_col} (smooth abs={len(series)}), butuh minimal 30.",
        )
    return series.tail(30).tolist()


def _delta_tail_30_from_abs_smooth(df_hist: pd.DataFrame, target_col: str) -> List[float]:
    """
    Bangun delta dari abs_smooth.diff(), lalu ambil 30 delta terakhir.
    """
    abs_30_plus = _abs_smooth_tail_30(df_hist, target_col)
    # abs_30_plus itu hanya 30 poin terakhir, delta butuh 31 poin untuk punya 30 delta,
    # jadi kita ambil dari full series agar benar.
    df = df_hist.copy()
    df[target_col] = _numeric_clean(df, target_col)
    df = df.dropna(subset=["date", target_col]).sort_values("date")

    smooth_col = f"{target_col}_smooth"
    df[smooth_col] = df[target_col].rolling(window=3, center=True).mean()
    df = df.dropna(subset=[smooth_col])

    abs_series = df[smooth_col].reset_index(drop=True)
    delta_series = abs_series.diff().dropna().reset_index(drop=True)

    if len(delta_series) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"Data historis tidak cukup untuk {target_col} (delta={len(delta_series)}), butuh minimal 30.",
        )
    return delta_series.tail(30).tolist()


def _season_features(current_date: datetime.date):
    dayofyear = current_date.timetuple().tm_yday
    sin_day = float(np.sin(2 * np.pi * dayofyear / 365.25))
    cos_day = float(np.cos(2 * np.pi * dayofyear / 365.25))
    is_weekday = 1 if current_date.weekday() < 5 else 0
    return sin_day, cos_day, is_weekday


class EvaluationHistoryService:
    """
    Rolling evaluation untuk periode tertentu, simpan hasil ke DB (SQLite).
    """

    def __init__(self):
        self.predict_service = PredictService()

    def _generate_forecast_as_of(
        self,
        db: Session,
        as_of_date: datetime.date,
        days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Forecast mulai as_of_date+1 sampai as_of_date+days_ahead.
        Support hybrid:
        - delta: temperature & humidity (reconstruct abs)
        - absolute: pressure & wind_speed (abs)
        """
        df_raw = load_data_from_db(db)
        if df_raw.empty:
            raise HTTPException(status_code=404, detail="Data historis kosong dari database.")

        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"]).sort_values("date")

        df_hist = df_raw[df_raw["date"].dt.date <= as_of_date].copy()
        if df_hist.empty:
            raise HTTPException(status_code=404, detail=f"Tidak ada data historis <= {as_of_date}")

        ps = self.predict_service
        target_cols = ["temperature", "humidity", "pressure", "wind_speed"]

        base_start_date = as_of_date + datetime.timedelta(days=1)
        forecast_dates = [base_start_date + datetime.timedelta(days=i) for i in range(days_ahead)]
        forecast_results = pd.DataFrame({"date": pd.to_datetime(forecast_dates)})

        for target_col in target_cols:
            assets = ps.model_assets.get(target_col)
            if not assets:
                raise HTTPException(status_code=500, detail=f"Aset model {target_col} tidak ditemukan.")

            model = assets["model"]
            scaler_X = assets["scaler_X"]
            scaler_y = assets["scaler_y"]
            model_type = assets.get("type", "absolute")

            preds_abs: List[float] = []

            if model_type == "delta":
                # historic_abs untuk base reconstruct + historic_delta untuk lag
                historic_abs = _abs_smooth_tail_30(df_hist, target_col)
                historic_delta = _delta_tail_30_from_abs_smooth(df_hist, target_col)
            else:
                # absolute model: gunakan abs smooth tail 30
                historic_abs = _abs_smooth_tail_30(df_hist, target_col)

            for i in range(days_ahead):
                current_date = forecast_dates[i]
                sin_day, cos_day, is_weekday = _season_features(current_date)

                if model_type == "delta":
                    lag_7 = historic_delta[-7]
                    lag_14 = historic_delta[-14]
                    lag_30 = historic_delta[-30]
                else:
                    lag_7 = historic_abs[-7]
                    lag_14 = historic_abs[-14]
                    lag_30 = historic_abs[-30]

                X_input = np.array([[lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]])
                X_scaled = scaler_X.transform(X_input)
                y_scaled_pred = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))[0][0]
                y_pred = float(y_pred)

                if model_type == "delta":
                    # reconstruct abs
                    new_abs = float(historic_abs[-1]) + y_pred
                    preds_abs.append(round(new_abs, 2))
                    historic_abs.append(new_abs)
                    historic_delta.append(y_pred)
                else:
                    preds_abs.append(round(y_pred, 2))
                    historic_abs.append(y_pred)

            forecast_results[target_col] = preds_abs

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
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="end_date harus >= start_date")
        if step_days < 1:
            raise HTTPException(status_code=400, detail="step_days minimal 1")
        if horizon_days < 1:
            raise HTTPException(status_code=400, detail="horizon_days minimal 1")

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

            preds = self._generate_forecast_as_of(db, as_of_date=as_of, days_ahead=horizon_days)
            pred_df = pd.DataFrame(preds)
            pred_df["d"] = pd.to_datetime(pred_df["date"]).dt.date

            actual_start = as_of + datetime.timedelta(days=1)
            actual_end = as_of + datetime.timedelta(days=horizon_days)

            actual_df = df_all[(df_all["d"] >= actual_start) & (df_all["d"] <= actual_end)].copy()
            if actual_df.empty:
                skipped += 1
                continue

            merged = pd.merge(
                pred_df,
                actual_df,
                how="inner",
                left_on="d",
                right_on="d",
                suffixes=("_pred", "_act"),
            )

            # pastikan horizon lengkap
            if merged["d"].nunique() < horizon_days:
                skipped += 1
                continue

            for param, (api_key, db_col) in params_map.items():
                y_true = pd.to_numeric(merged[db_col], errors="coerce")
                y_pred = pd.to_numeric(merged[api_key], errors="coerce")

                mask = (~y_true.isna()) & (~y_pred.isna())
                y_true_arr = y_true[mask].to_numpy(dtype=float)
                y_pred_arr = y_pred[mask].to_numpy(dtype=float)

                if len(y_true_arr) == 0:
                    skipped += 1
                    continue

                m = _metrics(y_true_arr, y_pred_arr)

                rec = EvaluationHistoryModel(
                    eval_date=as_of,
                    horizon_days=horizon_days,
                    parameter=param,
                    mse=m["mse"],
                    rmse=m["rmse"],
                    mae=m["mae"],
                    r2=m["r2"],
                    n_samples=int(len(y_true_arr)),
                )
                db.add(rec)
                created += 1

            db.commit()

        return {
            "status": "ok",
            "model_mode": MODEL_MODE,
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
    def get_latest_valid(self, db: Session, horizon_days: int = 7) -> List[Dict[str, Any]]:
        q = db.query(EvaluationHistoryModel).filter(
            EvaluationHistoryModel.horizon_days == horizon_days,
            EvaluationHistoryModel.mse.isnot(None),
            EvaluationHistoryModel.rmse.isnot(None),
            EvaluationHistoryModel.mae.isnot(None),
        ).order_by(EvaluationHistoryModel.eval_date.desc())

        rows = q.all()
        if not rows:
            return []

        def ok(x):
            return (x is not None) and math.isfinite(float(x))

        latest_date = None
        for r in rows:
            if ok(r.mse) and ok(r.rmse) and ok(r.mae):
                latest_date = r.eval_date
                break

        if latest_date is None:
            return []

        rows2 = db.query(EvaluationHistoryModel).filter(
            EvaluationHistoryModel.eval_date == latest_date,
            EvaluationHistoryModel.horizon_days == horizon_days,
        ).order_by(EvaluationHistoryModel.parameter.asc()).all()

        return [
            {
                "eval_date": r.eval_date.strftime("%Y-%m-%d"),
                "horizon_days": r.horizon_days,
                "parameter": r.parameter,
                "MSE": r.mse,
                "RMSE": r.rmse,
                "MAE": r.mae,
                "R2": r.r2,
                "n_samples": r.n_samples,
            }
            for r in rows2
        ]


