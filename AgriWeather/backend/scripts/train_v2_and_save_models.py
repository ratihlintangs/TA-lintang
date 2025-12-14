# backend/scripts/train_v2_and_save_models.py

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from math import pi
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================
# KONFIGURASI UMUM
# =========================

# v2 HYBRID:
# - temperature & humidity -> DELTA (stasioner/differencing)
# - pressure & wind_speed  -> ABSOLUTE (pakai filter untuk FEATURE input)

TARGET_COLUMNS_DELTA_V2 = ["temperature", "humidity"]
TARGET_COLUMNS_ABS_V2 = ["pressure", "wind_speed"]

FEATURES = ["lag_7", "lag_14", "lag_30", "sin_day", "cos_day", "is_weekday"]

# Simpan model v2 di folder backend/ (sama seperti v1)
BASE_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- Smoothing (untuk DELTA pipeline saja, harus konsisten dengan PredictService v2) ---
# "none" | "rolling" | "ewm"
SMOOTHING_MODE = os.getenv("AGRI_SMOOTHING_MODE", "ewm").lower()  # default: ewm
ROLLING_WINDOW = int(os.getenv("AGRI_ROLLING_WINDOW", "3"))
EWM_SPAN = int(os.getenv("AGRI_EWM_SPAN", "3"))

# Missing dates handling (harus konsisten dengan PredictService)
FILL_MISSING_DATES = os.getenv("AGRI_FILL_MISSING_DATES", "true").lower() == "true"

# --- FILTER untuk ABS pipeline (pressure & wind_speed) ---
# pressure: EMA mandatory
FILTER_PRESSURE_EMA_ALPHA = float(os.getenv("FILTER_PRESSURE_EMA_ALPHA", "0.2"))

# wind: EMA default ON
FILTER_WIND_EMA_ENABLED = os.getenv("FILTER_WIND_EMA_ENABLED", "true").lower() == "true"
FILTER_WIND_EMA_ALPHA = float(os.getenv("FILTER_WIND_EMA_ALPHA", "0.3"))

# wind: Kalman optional OFF by default
FILTER_WIND_KALMAN_ENABLED = os.getenv("FILTER_WIND_KALMAN_ENABLED", "false").lower() == "true"
FILTER_WIND_KALMAN_Q = float(os.getenv("FILTER_WIND_KALMAN_Q", "1e-3"))
FILTER_WIND_KALMAN_R = float(os.getenv("FILTER_WIND_KALMAN_R", "1e-2"))


# =========================
# HELPER: MODEL
# =========================
def create_model():
    return MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )


# =========================
# HELPER: LOAD DATA DB
# =========================
def load_weather_df_from_db():
    """
    Ambil data dari DB SQLite menggunakan SessionLocal dan WeatherHistoryModel.
    Return: DataFrame dengan kolom minimal: date, temperature, humidity, pressure, wind_speed
    """
    from backend.database import SessionLocal
    from backend.models import WeatherHistoryModel

    db = SessionLocal()
    try:
        rows = db.query(WeatherHistoryModel).order_by(WeatherHistoryModel.date.asc()).all()
        df = pd.DataFrame([vars(r) for r in rows])

        if "_sa_instance_state" in df.columns:
            df = df.drop(columns=["_sa_instance_state"])

        if "date" not in df.columns:
            raise RuntimeError("Kolom 'date' tidak ditemukan di DB. Cek models.py / skema tabel.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        return df
    finally:
        db.close()


# =========================
# HELPER: ENSURE DAILY SERIES
# =========================
def ensure_daily_series(df_raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Pastikan date valid, sorted, dan (opsional) dibuat harian kontinyu.
    Missing dates diisi dengan interpolasi time + ffill/bfill (agar lag = 7 hari beneran).
    """
    df = df_raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if target_col not in df.columns:
        return pd.DataFrame()

    # numeric coercion
    df[target_col] = df[target_col].replace("-", np.nan)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # keep only needed cols for stability
    df = df[["date", target_col]].copy()

    # optional daily fill
    if FILL_MISSING_DATES:
        df = df.set_index("date").asfreq("D")
        df[target_col] = df[target_col].interpolate(method="time", limit_direction="both")
        df[target_col] = df[target_col].ffill().bfill()
        df = df.reset_index()

    return df


# =========================
# HELPER: SMOOTHING (FOR DELTA ONLY, NO LEAKAGE)
# =========================
def apply_smoothing(series: pd.Series) -> pd.Series:
    """
    Smoothing causal (tanpa leakage):
    - none: no smoothing
    - rolling: rolling mean causal (tanpa center=True)
    - ewm: EWMA causal
    """
    s = series.copy()

    if SMOOTHING_MODE == "none":
        return s

    if SMOOTHING_MODE == "rolling":
        return s.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

    # default ewm
    return s.ewm(span=EWM_SPAN, adjust=False).mean()


# =========================
# HELPER: FILTER (FOR ABS ONLY)
# =========================
def ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    """
    EMA causal: y[t] = alpha*x[t] + (1-alpha)*y[t-1]
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x

    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def kalman_filter_1d(values: np.ndarray, q: float, r: float) -> np.ndarray:
    """
    Kalman 1D sederhana (state = value). Causal.
    q: process noise
    r: measurement noise
    """
    z = np.asarray(values, dtype=float)
    if z.size == 0:
        return z

    x = z[0]  # estimate
    p = 1.0   # covariance
    out = np.empty_like(z, dtype=float)
    out[0] = x

    for i in range(1, len(z)):
        # predict
        p = p + q

        # update
        k = p / (p + r)
        x = x + k * (z[i] - x)
        p = (1.0 - k) * p

        out[i] = x

    return out


def apply_abs_filters_for_feature(target_col: str, series_abs: pd.Series) -> pd.Series:
    """
    Return seri 'used' untuk dijadikan feature input.
    - pressure: EMA mandatory
    - wind_speed: EMA default, Kalman optional
    """
    x = series_abs.astype(float).values

    if target_col == "pressure":
        x = ema_filter(x, FILTER_PRESSURE_EMA_ALPHA)
        return pd.Series(x, index=series_abs.index)

    if target_col == "wind_speed":
        if FILTER_WIND_EMA_ENABLED:
            x = ema_filter(x, FILTER_WIND_EMA_ALPHA)
        if FILTER_WIND_KALMAN_ENABLED:
            x = kalman_filter_1d(x, q=FILTER_WIND_KALMAN_Q, r=FILTER_WIND_KALMAN_R)
        return pd.Series(x, index=series_abs.index)

    # default: no filter
    return series_abs


# =========================
# PREPROCESS v2 (DELTA) - temperature & humidity
# =========================
def preprocess_v2_delta(df_raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    v2: temperature & humidity → stasionerkan pakai differencing (delta).
    Pipeline:
    - ensure daily series (+ interpolate optional)
    - smoothing causal optional
    - delta = abs_used - abs_used.shift(1)
    - lag pada delta: 7, 14, 30
    - seasonality features
    - weekday
    """
    df = ensure_daily_series(df_raw, target_col)
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["date", target_col]).reset_index(drop=True)

    abs_series = apply_smoothing(df[target_col])
    df[f"{target_col}_abs_used"] = abs_series
    df = df.dropna(subset=[f"{target_col}_abs_used"]).reset_index(drop=True)

    delta_col = f"{target_col}_delta"
    df[delta_col] = df[f"{target_col}_abs_used"] - df[f"{target_col}_abs_used"].shift(1)

    # lag on delta
    df["lag_7"] = df[delta_col].shift(7)
    df["lag_14"] = df[delta_col].shift(14)
    df["lag_30"] = df[delta_col].shift(30)

    # seasonality
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_day"] = np.sin(2 * pi * df["dayofyear"] / 365.25)
    df["cos_day"] = np.cos(2 * pi * df["dayofyear"] / 365.25)

    # weekday
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekday"] = np.where(df["dayofweek"] < 5, 1, 0)

    df = df.dropna(subset=FEATURES + [delta_col]).reset_index(drop=True)
    return df


# =========================
# PREPROCESS v2 (ABS) - pressure & wind_speed
# =========================
def preprocess_v2_abs_filtered(df_raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    v2: pressure & wind_speed → ABSOLUTE target.
    Feature dibuat dari seri abs yang DIFILTER (causal), agar input lebih stabil.
    Pipeline:
    - ensure daily series (+ interpolate optional)
    - abs_used = apply_abs_filters_for_feature(target_col, abs_series)
    - lag pada abs_used: 7, 14, 30
    - seasonality features
    - weekday
    Output:
      - y (target) = abs original (target_col)
      - X (features) = lag dari abs_used + seasonality + weekday
    """
    df = ensure_daily_series(df_raw, target_col)
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["date", target_col]).reset_index(drop=True)

    # abs target (original)
    df[f"{target_col}_y"] = df[target_col].astype(float)

    # abs used for features (filtered)
    df[f"{target_col}_used"] = apply_abs_filters_for_feature(target_col, df[target_col])

    # lags built from used (filtered)
    df["lag_7"] = df[f"{target_col}_used"].shift(7)
    df["lag_14"] = df[f"{target_col}_used"].shift(14)
    df["lag_30"] = df[f"{target_col}_used"].shift(30)

    # seasonality
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_day"] = np.sin(2 * pi * df["dayofyear"] / 365.25)
    df["cos_day"] = np.cos(2 * pi * df["dayofyear"] / 365.25)

    # weekday
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekday"] = np.where(df["dayofweek"] < 5, 1, 0)

    df = df.dropna(subset=FEATURES + [f"{target_col}_y"]).reset_index(drop=True)
    return df


# =========================
# TRAIN + SAVE (DELTA)
# =========================
def train_and_save_single_delta_v2(df_raw: pd.DataFrame, target_col: str) -> bool:
    print("\n=======================================================")
    print(f"⏳ TRAIN v2 (DELTA) untuk TARGET: {target_col.upper()}")
    print("=======================================================")

    df = preprocess_v2_delta(df_raw, target_col)
    if df.empty:
        print(f"❌ Data kosong setelah preprocessing v2 untuk {target_col}.")
        return False

    delta_col = f"{target_col}_delta"

    # split train (80% awal)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()

    X_train = df_train[FEATURES].values
    y_train = df_train[delta_col].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    model = create_model()
    model.fit(X_train_scaled, y_train_scaled.ravel())

    # save (suffix _v2)
    model_filename = os.path.join(BASE_BACKEND_DIR, f"{target_col.lower()}_mlp_model_v2.pkl")
    scaler_x_filename = os.path.join(BASE_BACKEND_DIR, f"{target_col.lower()}_scaler_X_v2.pkl")
    scaler_y_filename = os.path.join(BASE_BACKEND_DIR, f"{target_col.lower()}_scaler_y_v2.pkl")

    joblib.dump(model, model_filename)
    joblib.dump(scaler_X, scaler_x_filename)
    joblib.dump(scaler_y, scaler_y_filename)

    print(f"✅ Saved: {os.path.basename(model_filename)}")
    print(f"✅ Saved: {os.path.basename(scaler_x_filename)}")
    print(f"✅ Saved: {os.path.basename(scaler_y_filename)}")

    print("Catatan konfigurasi training (DELTA):")
    print(f"- SMOOTHING_MODE={SMOOTHING_MODE} (rolling_window={ROLLING_WINDOW}, ewm_span={EWM_SPAN})")
    print(f"- FILL_MISSING_DATES={FILL_MISSING_DATES}")

    return True


# =========================
# TRAIN + SAVE (ABS FILTERED)
# =========================
def train_and_save_single_abs_v2(df_raw: pd.DataFrame, target_col: str) -> bool:
    print("\n=======================================================")
    print(f"⏳ TRAIN v2 (ABS + FILTERED FEATURES) untuk TARGET: {target_col.upper()}")
    print("=======================================================")

    df = preprocess_v2_abs_filtered(df_raw, target_col)
    if df.empty:
        print(f"❌ Data kosong setelah preprocessing ABS v2 untuk {target_col}.")
        return False

    y_col = f"{target_col}_y"

    # split train (80% awal)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()

    X_train = df_train[FEATURES].values
    y_train = df_train[y_col].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    model = create_model()
    model.fit(X_train_scaled, y_train_scaled.ravel())

    # save (suffix _v2)
    model_filename = os.path.join(BASE_BACKEND_DIR, f"{target_col.lower()}_mlp_model_v2.pkl")
    scaler_x_filename = os.path.join(BASE_BACKEND_DIR, f"{target_col.lower()}_scaler_X_v2.pkl")
    scaler_y_filename = os.path.join(BASE_BACKEND_DIR, f"{target_col.lower()}_scaler_y_v2.pkl")

    joblib.dump(model, model_filename)
    joblib.dump(scaler_X, scaler_x_filename)
    joblib.dump(scaler_y, scaler_y_filename)

    print(f"✅ Saved: {os.path.basename(model_filename)}")
    print(f"✅ Saved: {os.path.basename(scaler_x_filename)}")
    print(f"✅ Saved: {os.path.basename(scaler_y_filename)}")

    print("Catatan konfigurasi training (ABS FILTER):")
    print(f"- pressure EMA alpha={FILTER_PRESSURE_EMA_ALPHA} (mandatory)")
    print(f"- wind EMA enabled={FILTER_WIND_EMA_ENABLED}, alpha={FILTER_WIND_EMA_ALPHA}")
    print(f"- wind Kalman enabled={FILTER_WIND_KALMAN_ENABLED}, Q={FILTER_WIND_KALMAN_Q}, R={FILTER_WIND_KALMAN_R}")
    print(f"- FILL_MISSING_DATES={FILL_MISSING_DATES}")

    return True


def run_training_v2():
    print("⏳ Load data dari database (SQLite)...")
    df_raw = load_weather_df_from_db()

    if df_raw.empty:
        print("❌ Data kosong dari DB. Training dibatalkan.")
        return

    print(
        f"✅ Data loaded: {len(df_raw)} baris, range tanggal: "
        f"{df_raw['date'].min().date()} s/d {df_raw['date'].max().date()}"
    )

    ok_count = 0
    total = len(TARGET_COLUMNS_DELTA_V2) + len(TARGET_COLUMNS_ABS_V2)

    # 1) Train DELTA targets (temperature, humidity)
    for target in TARGET_COLUMNS_DELTA_V2:
        ok = train_and_save_single_delta_v2(df_raw, target)
        ok_count += 1 if ok else 0

    # 2) Train ABS targets with filtered features (pressure, wind_speed)
    for target in TARGET_COLUMNS_ABS_V2:
        ok = train_and_save_single_abs_v2(df_raw, target)
        ok_count += 1 if ok else 0

    print("\n=======================================================")
    print(f"✅ TRAINING v2 SELESAI. Sukses: {ok_count}/{total} model.")
    print("=======================================================")

    print("Catatan:")
    print("- v2 DELTA: memprediksi delta dari seri abs (optional smoothing). Reconstruct dilakukan di PredictService v2.")
    print("- v2 ABS (pressure & wind_speed): target tetap absolute, tapi fitur dibentuk dari seri abs yang difilter secara causal.")
    print("- Pastikan konfigurasi env sama antara training dan PredictService v2 (fill/filters).")


if __name__ == "__main__":
    run_training_v2()
