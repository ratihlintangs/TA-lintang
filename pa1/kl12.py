import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sin, cos, pi
from scipy.stats import zscore

# === Load & Preprocessing ===
df = pd.read_csv("kumpulan12.csv")
df['tanggal'] = pd.to_datetime(df['tanggal'], dayfirst=True, errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['tanggal'])
df = df.sort_values('tanggal')

# Konversi & bersihkan
df['RH2M'] = df['RH2M'].replace('-', np.nan)
df['RH2M'] = pd.to_numeric(df['RH2M'], errors='coerce')
df = df.dropna(subset=['RH2M'])

# Smoothing
df['RH2M_smooth'] = df['RH2M'].rolling(window=3, center=True).mean()
df = df.dropna(subset=['RH2M_smooth'])

# Outlier removal
df['zscore'] = zscore(df['RH2M_smooth'])
df = df[df['zscore'].abs() < 3]

# Fitur lag & musiman
df['lag_7'] = df['RH2M_smooth'].shift(7)
df['lag_14'] = df['RH2M_smooth'].shift(14)
df['lag_30'] = df['RH2M_smooth'].shift(30)
df['dayofyear'] = df['tanggal'].dt.dayofyear
df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365)
df['dayofweek'] = df['tanggal'].dt.dayofweek
df = df.dropna()

# Nilai aktual 7 hari ke depan
actual = np.array([
85.05,
86.63,
88.02,
88.55,
91.8,
87.8,
86.25,





])
evaluasi_list = []

def evaluate_and_store(y_true, y_pred, name, kombinasi):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evaluasi_list.append({
        'Kombinasi': kombinasi,
        'Model': name,
        'MSE': round(mse, 3),
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'R2': round(r2, 3)
    })
    print(f"\n{name} ({kombinasi})")
    print(f"  MSE  : {mse:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  R²   : {r2:.3f}")

# === Fungsi Prediksi ===
def predict_model(df, feature_names, label):
    X = df[feature_names].values
    y = df['RH2M_smooth'].values
    scaler = MinMaxScaler()
    y_scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    model_lin = LinearRegression().fit(X_scaled, y)
    model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_scaled, y)
    model_non = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=3000, random_state=42).fit(X_scaled, y_scaled)

    last_date = df['tanggal'].iloc[-1]
    lag_values = list(df['RH2M_smooth'].values[-30:])
    pred_lin, pred_poly, pred_non, future_dates = [], [], [], []

    for i in range(1, 8):
        next_date = last_date + pd.Timedelta(days=i)
        doy = next_date.dayofyear
        dow = next_date.dayofweek
        sin_day = sin(2 * pi * doy / 365)
        cos_day = cos(2 * pi * doy / 365)

        row = {
            'lag_7': lag_values[-7],
            'lag_14': lag_values[-14],
            'lag_30': lag_values[-30],
            'dayofyear': doy,
            'sin_day': sin_day,
            'cos_day': cos_day,
            'dayofweek': dow
        }

        x_next = np.array([[row.get(f, 0) for f in feature_names]])
        x_scaled = scaler.transform(x_next)

        y1 = model_lin.predict(x_scaled)[0]
        y2 = model_poly.predict(x_scaled)[0]
        y3_scaled = model_non.predict(x_scaled)[0]
        y3 = y_scaler.inverse_transform([[y3_scaled]])[0][0]

        pred_lin.append(round(y1, 2))
        pred_poly.append(round(y2, 2))
        pred_non.append(round(y3, 2))

        lag_values.append(y1)
        future_dates.append(next_date)

    print(f"\n=== Evaluasi: {label} ===")
    evaluate_and_store(actual, pred_lin, "Linear Regression", label)
    evaluate_and_store(actual, pred_non, "Neural Network (Nonlinear)", label)
    evaluate_and_store(actual, pred_poly, "Polynomial Regression", label)

    return pd.DataFrame({
        'tanggal': future_dates,
        'Aktual': actual,
        'Pred_Linear': pred_lin,
        'Pred_Nonlinear': pred_non,
        'Pred_Polynomial': pred_poly
    })

# === Fungsi Rolling Update ===
def predict_with_rolling(df, feature_names, label):
    df_copy = df.copy()
    pred_lin, pred_poly, pred_non, future_dates = [], [], [], []

    for i in range(1, 8):
        last_date = df_copy['tanggal'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        doy = next_date.dayofyear
        dow = next_date.dayofweek
        sin_day = sin(2 * pi * doy / 365)
        cos_day = cos(2 * pi * doy / 365)

        temp_df = df_copy.copy()
        temp_df['lag_7'] = temp_df['RH2M_smooth'].shift(7)
        temp_df['lag_14'] = temp_df['RH2M_smooth'].shift(14)
        temp_df['lag_30'] = temp_df['RH2M_smooth'].shift(30)
        temp_df['dayofyear'] = temp_df['tanggal'].dt.dayofyear
        temp_df['sin_day'] = np.sin(2 * pi * temp_df['dayofyear'] / 365)
        temp_df['cos_day'] = np.cos(2 * pi * temp_df['dayofyear'] / 365)
        temp_df['dayofweek'] = temp_df['tanggal'].dt.dayofweek
        temp_df = temp_df.dropna()

        X = temp_df[feature_names].values
        y = temp_df['RH2M_smooth'].values

        scaler = MinMaxScaler()
        y_scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        model_lin = LinearRegression().fit(X_scaled, y)
        model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_scaled, y)
        model_non = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=3000, random_state=42).fit(X_scaled, y_scaled)

        row = {
            'lag_7': df_copy['RH2M_smooth'].iloc[-7],
            'lag_14': df_copy['RH2M_smooth'].iloc[-14],
            'lag_30': df_copy['RH2M_smooth'].iloc[-30],
            'sin_day': sin_day,
            'cos_day': cos_day,
            'dayofweek': dow
        }

        x_next = np.array([[row[f] for f in feature_names]])
        x_scaled = scaler.transform(x_next)

        y1 = model_lin.predict(x_scaled)[0]
        y2 = model_poly.predict(x_scaled)[0]
        y3_scaled = model_non.predict(x_scaled)[0]
        y3 = y_scaler.inverse_transform([[y3_scaled]])[0][0]

        pred_lin.append(round(y1, 2))
        pred_poly.append(round(y2, 2))
        pred_non.append(round(y3, 2))
        future_dates.append(next_date)

        df_copy = pd.concat([df_copy, pd.DataFrame({'tanggal': [next_date], 'RH2M_smooth': [y1]})], ignore_index=True)

    print(f"\n=== Evaluasi: {label} ===")
    evaluate_and_store(actual, pred_lin, "Linear Regression", label)
    evaluate_and_store(actual, pred_non, "Neural Network (Nonlinear)", label)
    evaluate_and_store(actual, pred_poly, "Polynomial Regression", label)

    return pd.DataFrame({
        'tanggal': future_dates,
        'Aktual': actual,
        'Pred_Linear': pred_lin,
        'Pred_Nonlinear': pred_non,
        'Pred_Polynomial': pred_poly
    })

# === Jalankan Kombinasi A–D ===
print("\n========== KOMBINASI A: Preprocessing ==========")
results_A = predict_model(df, ['dayofyear'], "Kombinasi A: Preprocessing Only")

print("\n========== KOMBINASI B: Lag ==========")
results_B = predict_model(df, ['lag_7', 'lag_14', 'lag_30'], "Kombinasi B: Lag Only")

print("\n========== KOMBINASI C: Lag + Musiman ==========")
results_C = predict_model(df, ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day'], "Kombinasi C: Lag + Musiman")

print("\n========== KOMBINASI D: Lag + Musiman + Rolling ==========")
results_D = predict_with_rolling(df, ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day', 'dayofweek'], "Kombinasi D: Lag + Musiman + Rolling")

# === Simpan Evaluasi ===
df_evaluasi = pd.DataFrame(evaluasi_list)
df_evaluasi.to_csv("evaluasi_per_minggu12.csv", index=False)
print("\n✅ File 'evaluasi_per_minggu12.csv' berhasil disimpan.")

# === Visualisasi ===
plt.figure(figsize=(10, 5))
plt.plot(results_D['tanggal'], results_D['Aktual'], label='Aktual', marker='o')
plt.plot(results_D['tanggal'], results_D['Pred_Linear'], label='Linear', marker='s')
plt.plot(results_D['tanggal'], results_D['Pred_Nonlinear'], label='Nonlinear', marker='^')
plt.plot(results_D['tanggal'], results_D['Pred_Polynomial'], label='Polynomial', marker='x')
plt.title("Prediksi RH2M 1–7 Juni 2025")
plt.xlabel("Tanggal")
plt.ylabel("RH2M (°C)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
