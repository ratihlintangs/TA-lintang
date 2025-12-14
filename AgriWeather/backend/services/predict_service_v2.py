# backend/services/predict_service.py

import pandas as pd
import numpy as np
import joblib
import datetime
from math import sin, cos, pi
import os
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.database import load_data_from_db  # dari database.py


# === KONFIGURASI MODEL ===
TARGET_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed"]
FEATURES = ["lag_7", "lag_14", "lag_30", "sin_day", "cos_day", "is_weekday"]

# Lokasi file model berada di direktori induk (backend/)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")

# Smoothing mode:
# - "none"  : tidak smoothing (paling sederhana)
# - "rolling": rolling mean causal (tanpa leakage)
# - "ewm"   : EWMA causal (recommended untuk produksi)
SMOOTHING_MODE = os.getenv("AGRI_SMOOTHING_MODE", "ewm").lower()  # default ewm
ROLLING_WINDOW = int(os.getenv("AGRI_ROLLING_WINDOW", "3"))
EWM_SPAN = int(os.getenv("AGRI_EWM_SPAN", "3"))

# Missing dates handling:
# - jika tanggal bolong -> kita asfreq('D') lalu interpolate time
FILL_MISSING_DATES = os.getenv("AGRI_FILL_MISSING_DATES", "true").lower() == "true"


class PredictService:
    """
    Layanan memuat model, melakukan prediksi rekursif, dan mengelola data historis untuk fitur lag.
    v2 hybrid:
      - temperature, humidity => model DELTA (suffix _v2)
      - pressure, wind_speed  => model absolute (v1)
    """

    def __init__(self):
        self.model_assets = {}
        self._load_all_model_assets()

    # -------------------------
    # MODEL LOADING
    # -------------------------
    def _load_model_assets(self, target_col: str) -> bool:
        try:
            if target_col in ["temperature", "humidity"]:
                model_filename = f"{target_col.lower()}_mlp_model_v2.pkl"
                scaler_x_filename = f"{target_col.lower()}_scaler_X_v2.pkl"
                scaler_y_filename = f"{target_col.lower()}_scaler_y_v2.pkl"
                model_type = "delta"
            else:
                model_filename = f"{target_col.lower()}_mlp_model.pkl"
                scaler_x_filename = f"{target_col.lower()}_scaler_X.pkl"
                scaler_y_filename = f"{target_col.lower()}_scaler_y.pkl"
                model_type = "absolute"

            model = joblib.load(os.path.join(MODEL_DIR, model_filename))
            scaler_X = joblib.load(os.path.join(MODEL_DIR, scaler_x_filename))
            scaler_y = joblib.load(os.path.join(MODEL_DIR, scaler_y_filename))

            self.model_assets[target_col] = {
                "model": model,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "type": model_type,
            }

            print(f"✅ INFO: Aset model untuk {target_col.upper()} berhasil dimuat ({model_type}).")
            return True

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File model/scaler untuk '{target_col}' tidak ditemukan di: {MODEL_DIR}. Pastikan .pkl ada."
            )
        except Exception as e:
            print(f"❌ ERROR: Gagal memuat aset model untuk '{target_col}': {e}")
            return False

    def _load_all_model_assets(self) -> None:
        for target_col in TARGET_COLUMNS:
            try:
                self._load_model_assets(target_col)
            except FileNotFoundError as e:
                print(f"❌ KRITIS: Gagal inisialisasi layanan. {e}")
                self.model_assets = {}
                break

    # -------------------------
    # DATA PREP / CLEAN
    # -------------------------
    def _ensure_daily_series(
        self, df_raw: pd.DataFrame, target_col: str
    ) -> pd.DataFrame:
        """
        Pastikan data sorted, date valid, dan (opsional) dibuat harian kontinyu.
        Missing dates diisi lewat interpolate time (aman untuk fitur lag).
        """
        df = df_raw.copy()
        if "date" not in df.columns:
            raise HTTPException(status_code=500, detail="Kolom 'date' tidak ditemukan pada data historis.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        # Coerce numeric target
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        # Optional: make daily frequency to stabilize lag meaning (7 hari = 7 hari beneran)
        if FILL_MISSING_DATES:
            df = df.set_index("date")
            df = df.asfreq("D")  # tambah baris missing date
            if target_col in df.columns:
                # Interpolate time untuk missing values; lalu fallback ffill/bfill
                df[target_col] = df[target_col].interpolate(method="time", limit_direction="both")
                df[target_col] = df[target_col].ffill().bfill()
            df = df.reset_index()

        return df

    def _apply_smoothing(self, series: pd.Series) -> pd.Series:
        """
        Smoothing causal (tanpa leakage).
        - rolling: mean window (pakai masa lalu + hari ini)
        - ewm: EWMA
        - none: no smoothing
        """
        s = series.copy()

        if SMOOTHING_MODE == "none":
            return s

        if SMOOTHING_MODE == "rolling":
            # causal rolling (tanpa center=True)
            return s.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

        # default ewm
        return s.ewm(span=EWM_SPAN, adjust=False).mean()

    # -------------------------
    # INITIAL HISTORY BUILDERS
    # -------------------------
    def _prepare_initial_data_absolute(self, df_raw: pd.DataFrame, target_col: str) -> List[float]:
        """
        Untuk model absolute (v1):
        - ambil 30 nilai terakhir (bisa smoothed sesuai konfigurasi)
        """
        df = self._ensure_daily_series(df_raw, target_col)

        if target_col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Kolom '{target_col}' tidak ditemukan pada data historis.")

        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Data historis kosong untuk {target_col}.")

        # smoothing optional (causal)
        sm = self._apply_smoothing(df[target_col])
        df["__val__"] = sm
        df = df.dropna(subset=["__val__"]).reset_index(drop=True)

        if len(df) < 30:
            raise HTTPException(
                status_code=404,
                detail=f"Data historis tidak cukup ({len(df)} hari). Minimal 30 hari dibutuhkan untuk fitur lag.",
            )

        return df["__val__"].tail(30).tolist()

    def _prepare_initial_data_delta(self, df_raw: pd.DataFrame, target_col: str) -> Tuple[List[float], List[float]]:
        """
        Untuk model delta (v2):
        - bangun seri ABS (bisa smoothed sesuai config) untuk base rekonstruksi
        - delta = abs.diff()
        Return: (historic_abs_30, historic_delta_30)
        """
        df = self._ensure_daily_series(df_raw, target_col)

        if target_col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Kolom '{target_col}' tidak ditemukan pada data historis.")

        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Data historis kosong untuk {target_col}.")

        abs_series = self._apply_smoothing(df[target_col])
        abs_series = abs_series.dropna().reset_index(drop=True)

        delta_series = abs_series.diff().dropna().reset_index(drop=True)

        if len(abs_series) < 31:
            # butuh minimal 31 abs agar delta >=30
            raise HTTPException(
                status_code=404,
                detail=f"Data historis tidak cukup untuk delta ({target_col}). abs={len(abs_series)}; minimal 31 dibutuhkan.",
            )

        if len(delta_series) < 30:
            raise HTTPException(
                status_code=404,
                detail=f"Data historis tidak cukup untuk delta lag ({target_col}). delta={len(delta_series)}; minimal 30 dibutuhkan.",
            )

        historic_abs = abs_series.tail(30).tolist()
        historic_delta = delta_series.tail(30).tolist()

        return historic_abs, historic_delta

    # -------------------------
    # LAST VALID DATE (anti-null tail)
    # -------------------------
    def _get_last_valid_date_all_targets(self, df: pd.DataFrame) -> datetime.date:
        required_cols = ["date", "temperature", "humidity", "pressure", "wind_speed"]
        for c in required_cols:
            if c not in df.columns:
                raise HTTPException(status_code=500, detail=f"Kolom '{c}' tidak ditemukan pada data historis.")

        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        df2 = df2.dropna(subset=["date"]).sort_values("date")

        valid = df2.dropna(subset=["temperature", "humidity", "pressure", "wind_speed"])
        if valid.empty:
            return df2["date"].max().date()

        return valid["date"].max().date()

    # -------------------------
    # MAIN FORECAST
    # -------------------------
    def generate_weather_prediction(self, db: Session, days_ahead: int) -> List[Dict[str, Any]]:
        """
        Menghasilkan ramalan cuaca N hari ke depan.
        Output tanggal selalu dimulai dari 'hari ini' (today),
        meskipun data NASA POWER memiliki delay 1-3 hari.
        """
        if not self.model_assets:
            raise HTTPException(
                status_code=500,
                detail="Inisialisasi model prediksi gagal. Periksa log server untuk masalah FileNotFoundError.",
            )

        if days_ahead < 1:
            raise HTTPException(status_code=400, detail="days_ahead minimal 1.")

        df_raw = load_data_from_db(db)
        if df_raw is None or df_raw.empty:
            raise HTTPException(status_code=404, detail="Data historis kosong dari database.")

        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"]).sort_values("date")

        last_valid_date = self._get_last_valid_date_all_targets(df_raw)
        base_start_date = last_valid_date + datetime.timedelta(days=1)

        today = datetime.date.today()
        end_date = today + datetime.timedelta(days=days_ahead - 1)

        # gap handling (NASA delay)
        gap_days = 0
        if today > base_start_date:
            gap_days = (today - base_start_date).days

        total_days_to_generate = days_ahead + gap_days

        forecast_dates_full = [base_start_date + datetime.timedelta(days=i) for i in range(total_days_to_generate)]
        forecast_results_full = pd.DataFrame({"date": pd.to_datetime(forecast_dates_full)})

        # gunakan hist hanya sampai last_valid_date untuk starting point
        df_hist_all = df_raw[df_raw["date"].dt.date <= last_valid_date].copy()

        for target_col in TARGET_COLUMNS:
            assets = self.model_assets.get(target_col)
            if not assets:
                continue

            model = assets["model"]
            scaler_X = assets["scaler_X"]
            scaler_y = assets["scaler_y"]
            model_type = assets.get("type", "absolute")

            print(f"⏳ Memulai Prediksi Rekursif untuk: {target_col.upper()} ({model_type})")

            if model_type == "delta":
                historic_abs, historic_delta = self._prepare_initial_data_delta(df_hist_all, target_col)
            else:
                historic_values = self._prepare_initial_data_absolute(df_hist_all, target_col)

            predicted_values_full = []

            for i in range(total_days_to_generate):
                current_date = forecast_dates_full[i]

                if model_type == "delta":
                    lag_7 = historic_delta[-7]
                    lag_14 = historic_delta[-14]
                    lag_30 = historic_delta[-30]
                else:
                    lag_7 = historic_values[-7]
                    lag_14 = historic_values[-14]
                    lag_30 = historic_values[-30]

                dayofyear = current_date.timetuple().tm_yday
                sin_day = sin(2 * pi * dayofyear / 365.25)
                cos_day = cos(2 * pi * dayofyear / 365.25)
                is_weekday = 1 if current_date.weekday() < 5 else 0

                X_input = np.array([[lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]])
                X_scaled = scaler_X.transform(X_input)

                y_scaled_pred = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))[0][0]
                y_pred = float(y_pred)

                if model_type == "delta":
                    # reconstruct (domain: abs_series yang dipakai training, biasanya smoothed/none sesuai config)
                    last_abs = float(historic_abs[-1])
                    new_abs = last_abs + y_pred
                    new_abs_rounded = round(new_abs, 2)

                    predicted_values_full.append(new_abs_rounded)

                    historic_abs.append(new_abs)
                    historic_delta.append(y_pred)
                else:
                    y_pred_rounded = round(y_pred, 2)
                    predicted_values_full.append(y_pred_rounded)
                    historic_values.append(y_pred)

            forecast_results_full[target_col] = predicted_values_full
            print(f"✅ Prediksi {total_days_to_generate}-hari untuk {target_col.upper()} selesai.")

        # slice output to always start from today
        today_ts = pd.Timestamp(today)
        end_ts = pd.Timestamp(end_date)

        sliced = forecast_results_full[
            (forecast_results_full["date"] >= today_ts) & (forecast_results_full["date"] <= end_ts)
        ].copy()

        if sliced.empty:
            sliced = forecast_results_full.head(days_ahead).copy()

        final_forecast_list: List[Dict[str, Any]] = []
        for _, row in sliced.iterrows():
            final_forecast_list.append(
                {
                    "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                    "temperature_c": row.get("temperature", None),
                    "humidity_percent": row.get("humidity", None),
                    "pressure_hpa": row.get("pressure", None),
                    "wind_speed_kmh": row.get("wind_speed", None),
                }
            )

        return final_forecast_list
