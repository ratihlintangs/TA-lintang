# backend/services/predict_service_v2.py

import pandas as pd
import numpy as np
import joblib
import datetime
from math import sin, cos, pi
import os
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.database import load_data_from_db  # dari database.py

# =========================
# KONFIGURASI MODEL
# =========================
TARGET_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed"]
FEATURES = ["lag_7", "lag_14", "lag_30", "sin_day", "cos_day", "is_weekday"]

# Lokasi file model berada di direktori induk (backend/)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")

# --- Smoothing (dipakai untuk DELTA model: temperature & humidity) ---
SMOOTHING_MODE = os.getenv("AGRI_SMOOTHING_MODE", "ewm").lower()  # default ewm
ROLLING_WINDOW = int(os.getenv("AGRI_ROLLING_WINDOW", "3"))
EWM_SPAN = int(os.getenv("AGRI_EWM_SPAN", "3"))

# Missing dates handling
FILL_MISSING_DATES = os.getenv("AGRI_FILL_MISSING_DATES", "true").lower() == "true"

# --- FILTER untuk ABS pipeline (pressure & wind_speed) ---
FILTER_PRESSURE_EMA_ALPHA = float(os.getenv("FILTER_PRESSURE_EMA_ALPHA", "0.2"))

FILTER_WIND_EMA_ENABLED = os.getenv("FILTER_WIND_EMA_ENABLED", "true").lower() == "true"
FILTER_WIND_EMA_ALPHA = float(os.getenv("FILTER_WIND_EMA_ALPHA", "0.3"))

FILTER_WIND_KALMAN_ENABLED = os.getenv("FILTER_WIND_KALMAN_ENABLED", "false").lower() == "true"
FILTER_WIND_KALMAN_Q = float(os.getenv("FILTER_WIND_KALMAN_Q", "1e-3"))
FILTER_WIND_KALMAN_R = float(os.getenv("FILTER_WIND_KALMAN_R", "1e-2"))


# =========================
# FILTER HELPERS (STATEFUL)
# =========================
class EMAState:
    def __init__(self, alpha: float):
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError(f"EMA alpha harus di (0,1], dapat: {alpha}")
        self.alpha = float(alpha)
        self.last: Optional[float] = None

    def seed(self, last_val: float):
        self.last = float(last_val)

    def update(self, x: float) -> float:
        x = float(x)
        if self.last is None:
            self.last = x
            return x
        y = self.alpha * x + (1.0 - self.alpha) * self.last
        self.last = y
        return y


class Kalman1DState:
    """
    Kalman 1D sederhana (state=value).
    Konsisten dengan training: q & r sama; causal; update incremental.
    """
    def __init__(self, q: float, r: float):
        if float(q) <= 0 or float(r) <= 0:
            raise ValueError(f"Kalman q dan r harus > 0. q={q}, r={r}")
        self.q = float(q)
        self.r = float(r)
        self.x: Optional[float] = None  # estimate
        self.p: float = 1.0            # covariance

    def seed(self, x0: float):
        self.x = float(x0)
        self.p = 1.0

    def update(self, z: float) -> float:
        z = float(z)
        if self.x is None:
            self.seed(z)
            return z

        # predict
        self.p = self.p + self.q

        # update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return float(self.x)


# =========================
# SERVICE (NAMA CLASS HARUS PredictService utk router kamu)
# =========================
class PredictService:
    """
    Layanan prediksi v2 hybrid:
      - temperature, humidity => model DELTA (suffix _v2)
      - pressure, wind_speed  => model ABSOLUTE v2 (suffix _v2)
         fitur dibentuk dari seri abs yang difilter (pressure EMA mandatory, wind EMA default + Kalman optional)
    """

    def __init__(self):
        self.model_assets: Dict[str, Dict[str, Any]] = {}
        self._load_all_model_assets()

    # -------------------------
    # MODEL LOADING
    # -------------------------
    def _load_model_assets(self, target_col: str) -> bool:
        try:
            # v2:
            # - temp/hum: delta model v2
            # - pressure/wind: absolute model v2 (yang baru kamu train)
            if target_col in ["temperature", "humidity"]:
                model_filename = f"{target_col.lower()}_mlp_model_v2.pkl"
                scaler_x_filename = f"{target_col.lower()}_scaler_X_v2.pkl"
                scaler_y_filename = f"{target_col.lower()}_scaler_y_v2.pkl"
                model_type = "delta"
            else:
                model_filename = f"{target_col.lower()}_mlp_model_v2.pkl"
                scaler_x_filename = f"{target_col.lower()}_scaler_X_v2.pkl"
                scaler_y_filename = f"{target_col.lower()}_scaler_y_v2.pkl"
                model_type = "absolute_v2_filtered_features"

            model_path = os.path.join(MODEL_DIR, model_filename)
            sx_path = os.path.join(MODEL_DIR, scaler_x_filename)
            sy_path = os.path.join(MODEL_DIR, scaler_y_filename)

            model = joblib.load(model_path)
            scaler_X = joblib.load(sx_path)
            scaler_y = joblib.load(sy_path)

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
                f"File model/scaler untuk '{target_col}' tidak ditemukan di: {MODEL_DIR}. "
                f"Pastikan .pkl v2 sudah ada."
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
    def _ensure_daily_series(self, df_raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df_raw.copy()
        if "date" not in df.columns:
            raise HTTPException(status_code=500, detail="Kolom 'date' tidak ditemukan pada data historis.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        if FILL_MISSING_DATES:
            df = df.set_index("date")
            df = df.asfreq("D")
            if target_col in df.columns:
                df[target_col] = df[target_col].interpolate(method="time", limit_direction="both")
                df[target_col] = df[target_col].ffill().bfill()
            df = df.reset_index()

        return df

    def _apply_smoothing(self, series: pd.Series) -> pd.Series:
        s = series.copy()

        if SMOOTHING_MODE == "none":
            return s

        if SMOOTHING_MODE == "rolling":
            return s.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

        return s.ewm(span=EWM_SPAN, adjust=False).mean()

    # -------------------------
    # INITIAL HISTORY BUILDERS
    # -------------------------
    def _prepare_initial_data_delta(self, df_raw: pd.DataFrame, target_col: str) -> Tuple[List[float], List[float]]:
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
            raise HTTPException(
                status_code=404,
                detail=f"Data historis tidak cukup untuk delta ({target_col}). abs={len(abs_series)}; minimal 31 dibutuhkan.",
            )
        if len(delta_series) < 30:
            raise HTTPException(
                status_code=404,
                detail=f"Data historis tidak cukup untuk delta lag ({target_col}). delta={len(delta_series)}; minimal 30 dibutuhkan.",
            )

        return abs_series.tail(30).tolist(), delta_series.tail(30).tolist()

    def _prepare_initial_data_abs_v2_filtered(
        self, df_raw: pd.DataFrame, target_col: str
    ) -> Tuple[List[float], Optional[EMAState], Optional[Kalman1DState]]:
        """
        Untuk pressure & wind_speed (ABS v2 dengan filtered features):
        - ambil minimal 30 nilai terakhir (abs raw)
        - bentuk 'used' series (filtered) untuk lag features
        - siapkan state filter untuk update incremental selama prediksi rekursif
        """
        df = self._ensure_daily_series(df_raw, target_col)

        if target_col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Kolom '{target_col}' tidak ditemukan pada data historis.")

        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Data historis kosong untuk {target_col}.")

        raw_vals = df[target_col].astype(float).tolist()
        if len(raw_vals) < 30:
            raise HTTPException(
                status_code=404,
                detail=f"Data historis tidak cukup ({len(raw_vals)} hari). Minimal 30 hari dibutuhkan untuk fitur lag.",
            )

        if target_col == "pressure":
            ema = EMAState(FILTER_PRESSURE_EMA_ALPHA)
            used_all = [ema.update(v) for v in raw_vals]
            return used_all[-30:], ema, None

        if target_col == "wind_speed":
            ema = EMAState(FILTER_WIND_EMA_ALPHA) if FILTER_WIND_EMA_ENABLED else None
            kal = Kalman1DState(q=FILTER_WIND_KALMAN_Q, r=FILTER_WIND_KALMAN_R) if FILTER_WIND_KALMAN_ENABLED else None

            used_all = []
            for v in raw_vals:
                x = float(v)
                if ema is not None:
                    x = ema.update(x)
                if kal is not None:
                    x = kal.update(x)
                used_all.append(x)

            return used_all[-30:], ema, kal

        # fallback
        return raw_vals[-30:], None, None

    # -------------------------
    # LAST VALID DATE
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

        df_hist_all = df_raw[df_raw["date"].dt.date <= last_valid_date].copy()

        for target_col in TARGET_COLUMNS:
            assets = self.model_assets.get(target_col)
            if not assets:
                continue

            model = assets["model"]
            scaler_X = assets["scaler_X"]
            scaler_y = assets["scaler_y"]
            model_type = assets.get("type", "delta")

            print(f"⏳ Memulai Prediksi Rekursif untuk: {target_col.upper()} ({model_type})")

            if model_type == "delta":
                historic_abs, historic_delta = self._prepare_initial_data_delta(df_hist_all, target_col)
            else:
                historic_used, ema_state, kalman_state = self._prepare_initial_data_abs_v2_filtered(df_hist_all, target_col)

            predicted_values_full: List[float] = []

            for i in range(total_days_to_generate):
                current_date = forecast_dates_full[i]

                if model_type == "delta":
                    lag_7, lag_14, lag_30 = historic_delta[-7], historic_delta[-14], historic_delta[-30]
                else:
                    lag_7, lag_14, lag_30 = historic_used[-7], historic_used[-14], historic_used[-30]

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
                    last_abs = float(historic_abs[-1])
                    new_abs = last_abs + y_pred
                    predicted_values_full.append(round(new_abs, 2))

                    historic_abs.append(new_abs)
                    historic_delta.append(y_pred)
                else:
                    # ABS v2 output
                    y_abs = y_pred
                    predicted_values_full.append(round(y_abs, 2))

                    # Update "used" for next lags (incremental, causal)
                    if target_col == "pressure":
                        # EMA mandatory
                        historic_used.append(ema_state.update(y_abs) if ema_state else y_abs)
                    elif target_col == "wind_speed":
                        x = y_abs
                        if ema_state is not None:
                            x = ema_state.update(x)
                        if kalman_state is not None:
                            x = kalman_state.update(x)
                        historic_used.append(x)
                    else:
                        historic_used.append(y_abs)

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
