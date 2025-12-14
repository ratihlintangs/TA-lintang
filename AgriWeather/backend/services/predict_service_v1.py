import pandas as pd
import numpy as np
import joblib
import datetime
from math import sin, cos, pi
import os
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.database import load_data_from_db  # dari database.py

# === KONFIGURASI MODEL ===
TARGET_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
]

# Lokasi file model berada di direktori induk (backend/)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")

class PredictService:
    """
    Layanan yang bertanggung jawab untuk memuat model, melakukan prediksi rekursif,
    dan mengelola data historis untuk fitur lag.
    """

    def __init__(self):
        self.model_assets = {}
        self._load_all_model_assets()

    def _load_model_assets(self, target_col: str):
        """Memuat model dan scalers untuk kolom target tertentu."""
        model_filename = f"{target_col.lower()}_mlp_model.pkl"
        scaler_x_filename = f"{target_col.lower()}_scaler_X.pkl"
        scaler_y_filename = f"{target_col.lower()}_scaler_y.pkl"

        try:
            model = joblib.load(os.path.join(MODEL_DIR, model_filename))
            scaler_X = joblib.load(os.path.join(MODEL_DIR, scaler_x_filename))
            scaler_y = joblib.load(os.path.join(MODEL_DIR, scaler_y_filename))

            self.model_assets[target_col] = {
                "model": model,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
            }
            print(f"✅ INFO: Aset model untuk {target_col.upper()} berhasil dimuat.")
            return True

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File model atau scaler untuk '{target_col}' tidak ditemukan di jalur: {MODEL_DIR}. Pastikan .pkl ada."
            )
        except Exception as e:
            print(f"❌ ERROR: Gagal memuat aset model untuk '{target_col}': {e}")
            return False

    def _load_all_model_assets(self):
        """Memuat semua 4 set model saat layanan diinisialisasi."""
        for target_col in TARGET_COLUMNS:
            try:
                self._load_model_assets(target_col)
            except FileNotFoundError as e:
                print(f"❌ KRITIS: Gagal inisialisasi layanan. {e}")
                self.model_assets = {}
                break

    def _prepare_initial_data(self, df_raw: pd.DataFrame, target_col: str) -> List[float]:
        """
        Pra-pemrosesan dan mendapatkan 30 nilai smoothed terakhir dari data historis
        untuk memulai fitur lag.
        """
        target_col_smooth = f"{target_col}_smooth"
        df = df_raw.copy()

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])

        # smoothing (window=3)
        df[target_col_smooth] = df[target_col].rolling(window=3, center=True).mean()
        df = df.dropna(subset=[target_col_smooth])

        # butuh minimal 30 hari untuk lag 7, 14, 30
        if len(df) < 30:
            raise HTTPException(
                status_code=404,
                detail=f"Data historis tidak cukup ({len(df)} hari). Minimal 30 hari dibutuhkan untuk fitur lag.",
            )

        return df[target_col_smooth].tail(30).tolist()

    def _get_last_valid_date_all_targets(self, df: pd.DataFrame) -> datetime.date:
        """
        Ambil tanggal terakhir yang valid (non-null) untuk SEMUA target.
        Ini menghindari 'tail null' (tanggal ada tapi data belum tersedia).
        """
        required_cols = ["date", "temperature", "humidity", "pressure", "wind_speed"]
        for c in required_cols:
            if c not in df.columns:
                raise HTTPException(status_code=500, detail=f"Kolom '{c}' tidak ditemukan pada data historis.")

        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        df2 = df2.dropna(subset=["date"])

        valid = df2.dropna(subset=["temperature", "humidity", "pressure", "wind_speed"])
        if valid.empty:
            # fallback: pakai tanggal terakhir yang ada
            return df2["date"].max().date()

        return valid["date"].max().date()

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

        # 1) Muat Data Historis
        df_raw = load_data_from_db(db)
        if df_raw.empty:
            raise HTTPException(status_code=404, detail="Data historis kosong dari database.")

        # pastikan date rapi & urut
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"]).sort_values("date")

        # 2) Tentukan last_valid_date (anti null tail)
        last_valid_date = self._get_last_valid_date_all_targets(df_raw)

        # 3) Base start date prediksi (t+1 dari observasi valid terakhir)
        base_start_date = last_valid_date + datetime.timedelta(days=1)

        # 4) Target output: selalu mulai hari ini
        today = datetime.date.today()
        end_date = today + datetime.timedelta(days=days_ahead - 1)

        # kalau today lebih maju dari base_start_date → ada gap karena NASA delay
        gap_days = 0
        if today > base_start_date:
            gap_days = (today - base_start_date).days

        total_days_to_generate = days_ahead + gap_days

        forecast_dates_full = [
            base_start_date + datetime.timedelta(days=i) for i in range(total_days_to_generate)
        ]

        forecast_results_full = pd.DataFrame({"date": pd.to_datetime(forecast_dates_full)})

        # 5) Prediksi rekursif tiap target
        for target_col in TARGET_COLUMNS:
            assets = self.model_assets.get(target_col)
            if not assets:
                continue

            model = assets["model"]
            scaler_X = assets["scaler_X"]
            scaler_y = assets["scaler_y"]

            print(f"⏳ Memulai Prediksi Rekursif untuk: {target_col.upper()}")

            # hanya pakai histori sampai last_valid_date (hindari tail null)
            df_hist = df_raw[df_raw["date"].dt.date <= last_valid_date].copy()

            try:
                historic_values = self._prepare_initial_data(df_hist, target_col)
            except HTTPException as e:
                raise e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Kesalahan pra-pemrosesan data untuk {target_col}: {e}")

            predicted_values_full = []

            for i in range(total_days_to_generate):
                current_date = forecast_dates_full[i]

                lag_7 = historic_values[-7]
                lag_14 = historic_values[-14]
                lag_30 = historic_values[-30]

                dayofyear = current_date.timetuple().tm_yday
                sin_day = sin(2 * pi * dayofyear / 365.25)
                cos_day = cos(2 * pi * dayofyear / 365.25)
                dayofweek = current_date.weekday()
                is_weekday = 1 if dayofweek < 5 else 0

                X_input = np.array([[lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]])

                X_scaled = scaler_X.transform(X_input)
                y_scaled_pred = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))[0][0]

                y_pred = float(y_pred)
                y_pred_rounded = round(y_pred, 2)

                predicted_values_full.append(y_pred_rounded)
                historic_values.append(y_pred)  # rekursif pakai nilai asli

            forecast_results_full[target_col] = predicted_values_full
            print(f"✅ Prediksi {total_days_to_generate}-hari untuk {target_col.upper()} selesai.")

        # 6) Slice output: today..end_date (fix Timestamp vs date)
        today_ts = pd.Timestamp(today)
        end_ts = pd.Timestamp(end_date)

        sliced = forecast_results_full[
            (forecast_results_full["date"] >= today_ts)
            & (forecast_results_full["date"] <= end_ts)
        ].copy()

        # fallback kalau sliced kosong (misalnya today < base_start_date)
        if sliced.empty:
            sliced = forecast_results_full.head(days_ahead).copy()

        # 7) Format output API
        final_forecast_list = []
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


