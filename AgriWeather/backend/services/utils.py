import os
import joblib
import pandas as pd
import numpy as np
import traceback

from pydantic import BaseModel
from typing import List, Tuple, Any
from datetime import datetime, timedelta
from math import sin, cos, pi

# ============================================================
# KONFIGURASI GLOBAL
# ============================================================

TARGET_COLUMNS = ['temperature', 'humidity', 'pressure', 'wind_speed']
DAYS_TO_FORECAST = 7


# ============================================================
# 1. SCHEMA PYDANTIC (UNTUK ROUTER & FRONTEND)
# ============================================================

class WeatherHistory(BaseModel):
    """Schema Pydantic untuk data historis cuaca."""
    id: int
    date: str
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float

    class Config:
        from_attributes = True  # WAJIB untuk SQLAlchemy


class ModelEvaluation(BaseModel):
    target: str
    rmse: float
    mae: float
    model_name: str


class ForecastData(BaseModel):
    date: str
    temperature_c: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_kmh: float


class ForecastResponse(BaseModel):
    predictions: List[ForecastData]
    evaluations: List[ModelEvaluation]


# ============================================================
# 2. MODEL & SCALER LOADER
# ============================================================

def _load_model_and_scalers(target_col: str) -> Tuple[Any, Any, Any]:
    """
    Memuat model MLP dan scaler X & y untuk satu parameter cuaca.
    """
    model_file = f"{target_col.lower()}_mlp_model.pkl"
    scaler_x_file = f"{target_col.lower()}_scaler_X.pkl"
    scaler_y_file = f"{target_col.lower()}_scaler_y.pkl"

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    model_path = os.path.join(base_dir, model_file)
    scaler_x_path = os.path.join(base_dir, scaler_x_file)
    scaler_y_path = os.path.join(base_dir, scaler_y_file)

    try:
        model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        print(f"✅ Model {target_col.upper()} berhasil dimuat")
        return model, scaler_X, scaler_y

    except Exception as e:
        print(f"❌ Gagal memuat model {target_col.upper()}")
        traceback.print_exc()
        raise e


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def _create_features(df: pd.DataFrame, target_col: str, forecast_date: datetime):
    """
    Membuat fitur lag + musiman untuk satu tanggal prediksi.
    """
    dayofyear = forecast_date.timetuple().tm_yday
    sin_day = sin(2 * pi * dayofyear / 365.25)
    cos_day = cos(2 * pi * dayofyear / 365.25)
    is_weekday = 1 if forecast_date.weekday() < 5 else 0

    target_smooth = f"{target_col}_smooth"

    if target_smooth not in df.columns:
        df[target_smooth] = (
            df[target_col]
            .rolling(window=3, center=True)
            .mean()
            .fillna(df[target_col].mean())
        )

    df = df.sort_values("tanggal").reset_index(drop=True)

    def lag_value(days):
        target_date = (forecast_date - timedelta(days=days)).strftime("%Y-%m-%d")
        match = df.loc[df["tanggal"].dt.strftime("%Y-%m-%d") == target_date, target_smooth]
        return match.iloc[0] if not match.empty else df[target_smooth].mean()

    lag_7 = lag_value(7)
    lag_14 = lag_value(14)
    lag_30 = lag_value(30)

    return np.array([
        lag_7, lag_14, lag_30,
        sin_day, cos_day,
        is_weekday
    ]).reshape(1, -1)


# ============================================================
# 4. PREDIKSI 1 PARAMETER (AUTO-REGRESSIVE)
# ============================================================

def _predict_single_variable(
    df_hist: pd.DataFrame,
    target_col: str
) -> List[Tuple[datetime, float]]:

    model, scaler_X, scaler_y = _load_model_and_scalers(target_col)

    df = df_hist[["tanggal", target_col]].copy()
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal").tail(30).reset_index(drop=True)

    last_date = df["tanggal"].iloc[-1]
    predictions = []

    for i in range(1, DAYS_TO_FORECAST + 1):
        forecast_date = last_date + timedelta(days=i)

        X = _create_features(df, target_col, forecast_date)
        X_scaled = scaler_X.transform(X)

        y_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        )[0][0]

        predictions.append((forecast_date, y_pred))

        df = pd.concat([
            df,
            pd.DataFrame([{
                "tanggal": forecast_date,
                target_col: y_pred,
                f"{target_col}_smooth": y_pred
            }])
        ], ignore_index=True)

    return predictions


# ============================================================
# 5. PIPELINE UTAMA (DIPANGGIL ROUTER PREDICT)
# ============================================================

def load_all_models_and_data(
    df_hist: pd.DataFrame
) -> Tuple[List[ForecastData], List[ModelEvaluation]]:

    if df_hist.empty:
        print("⚠️ Data historis kosong")
        return [], []

    results = {}

    for target in TARGET_COLUMNS:
        try:
            results[target] = _predict_single_variable(df_hist, target)
        except Exception:
            print(f"❌ Prediksi gagal untuk {target}")
            return [], []

    combined_forecast: List[ForecastData] = []

    for i in range(DAYS_TO_FORECAST):
        date_str = results["temperature"][i][0].strftime("%Y-%m-%d")
        combined_forecast.append(
            ForecastData(
                date=date_str,
                temperature_c=results["temperature"][i][1],
                humidity_percent=results["humidity"][i][1],
                pressure_hpa=results["pressure"][i][1],
                wind_speed_kmh=results["wind_speed"][i][1],
            )
        )

    # Evaluasi sementara (dummy → bisa diganti real evaluation)
    evaluations = [
        ModelEvaluation(target="Temperature", rmse=2.1, mae=1.5, model_name="MLP"),
        ModelEvaluation(target="Humidity", rmse=4.3, mae=3.0, model_name="MLP"),
        ModelEvaluation(target="Pressure", rmse=0.9, mae=0.6, model_name="MLP"),
        ModelEvaluation(target="Wind Speed", rmse=0.7, mae=0.5, model_name="MLP"),
    ]

    return combined_forecast, evaluations
