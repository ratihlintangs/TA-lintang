import pandas as pd
import numpy as np
from math import sin, cos, pi
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load data
df = pd.read_csv("kumpulan1.csv")
df['tanggal'] = pd.to_datetime(df['tanggal'], dayfirst=True)
df = df.sort_values('tanggal')

# =========================
# === Fungsi Umum Model ===
# =========================
def forecast_param(df, param, actual, label='Kombinasi D'):
    df = df.copy()
    df[param] = df[param].replace('-', np.nan)
    df[param] = pd.to_numeric(df[param], errors='coerce')
    df = df.dropna(subset=[param])

    df['lag_7'] = df[param].shift(7)
    df['lag_14'] = df[param].shift(14)
    df['lag_30'] = df[param].shift(30)
    df['dayofyear'] = df['tanggal'].dt.dayofyear
    df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365)
    df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365)
    df = df.dropna()

    df_copy = df.copy()
    pred_lin, pred_poly, pred_non, future_dates = [], [], [], []

    for i in range(1, 8):
        last_date = df_copy['tanggal'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        doy = next_date.dayofyear
        sin_day = sin(2 * pi * doy / 365)
        cos_day = cos(2 * pi * doy / 365)

        temp_df = df_copy.copy()
        temp_df['lag_7'] = temp_df[param].shift(7)
        temp_df['lag_14'] = temp_df[param].shift(14)
        temp_df['lag_30'] = temp_df[param].shift(30)
        temp_df['dayofyear'] = temp_df['tanggal'].dt.dayofyear
        temp_df['sin_day'] = np.sin(2 * pi * temp_df['dayofyear'] / 365)
        temp_df['cos_day'] = np.cos(2 * pi * temp_df['dayofyear'] / 365)
        temp_df = temp_df.dropna()

        X = temp_df[['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day']].values
        y = temp_df[param].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model_lin = LinearRegression().fit(X_scaled, y)
        model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_scaled, y)
        model_non = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42).fit(X_scaled, y)

        lag_7 = df_copy[param].iloc[-7]
        lag_14 = df_copy[param].iloc[-14]
        lag_30 = df_copy[param].iloc[-30]

        row = {
            'lag_7': lag_7,
            'lag_14': lag_14,
            'lag_30': lag_30,
            'sin_day': sin_day,
            'cos_day': cos_day
        }

        x_next = np.array([[row[f] for f in ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day']]])
        x_scaled = scaler.transform(x_next)

        y1 = model_lin.predict(x_scaled)[0]
        y2 = model_poly.predict(x_scaled)[0]
        y3 = model_non.predict(x_scaled)[0]

        pred_lin.append(round(y1, 2))
        pred_poly.append(round(y2, 2))
        pred_non.append(round(y3, 2))
        future_dates.append(next_date)

        df_copy = pd.concat([df_copy, pd.DataFrame({'tanggal': [next_date], param: [y1]})], ignore_index=True)

    return pd.DataFrame({
        'tanggal': future_dates,
        f'{param}_Aktual': actual,
        f'{param}_Pred_Linear': pred_lin,
        f'{param}_Pred_Nonlinear': pred_non,
        f'{param}_Pred_Polynomial': pred_poly
    })

# ========================
# Jalankan untuk T2M dan RH2M
# ========================

# Nilai aktual sesuai dari masing-masing file
actual_T2M = [25.3, 24.52, 25.26, 25.08, 24.46, 24.38, 24.32]
actual_RH2M = [88.76, 88.85, 85.82, 85.16, 86.43, 86.38, 87.96]

# Prediksi
pred_t2m = forecast_param(df.copy(), "T2M", actual_T2M)
pred_rh2m = forecast_param(df.copy(), "RH2M", actual_RH2M)

# Gabungkan berdasarkan tanggal
merged = pd.merge(pred_t2m, pred_rh2m, on='tanggal', how='inner')

# Simpan ke Excel
merged.to_excel("gabungan_prediksi_T2M_RH2M.xlsx", index=False)
print("\n✅ File 'gabungan_prediksi_T2M_RH2M.xlsx' berhasil disimpan.")
