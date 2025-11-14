import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sin, cos, pi

# === Load & Preprocessing ===
df = pd.read_csv("kumpulan6.csv")

# Perbaikan parsing tanggal agar fleksibel
df['tanggal'] = pd.to_datetime(df['tanggal'], dayfirst=True, errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['tanggal'])

df = df.sort_values('tanggal')
df['T2M'] = df['T2M'].replace('-', np.nan)
df['T2M'] = pd.to_numeric(df['T2M'], errors='coerce')
df = df.dropna(subset=['T2M'])

# Tambah fitur lag dan musiman
df['lag_7'] = df['T2M'].shift(7)
df['lag_14'] = df['T2M'].shift(14)
df['lag_30'] = df['T2M'].shift(30)
df['dayofyear'] = df['tanggal'].dt.dayofyear
df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365)
df = df.dropna()

# Nilai aktual suhu 7 hari ke depan (1–7 Juni)
actual = np.array([ 24.57,
24.61,
24.48,
24.48,
24.64,
24.31,
24.51

 ])

# === List hasil evaluasi ===
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

# === Fungsi Prediksi Standar ===
def predict_model(df, feature_names, label):
    X = df[feature_names].values
    y = df['T2M'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model_lin = LinearRegression().fit(X_scaled, y)
    model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_scaled, y)
    model_non = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42).fit(X_scaled, y)

    last_date = df['tanggal'].iloc[-1]
    lag_values = list(df['T2M'].values[-30:])
    pred_lin, pred_poly, pred_non, future_dates = [], [], [], []

    for i in range(1, 8):
        next_date = last_date + pd.Timedelta(days=i)
        doy = next_date.dayofyear
        sin_day = sin(2 * pi * doy / 365)
        cos_day = cos(2 * pi * doy / 365)

        row = {}
        if 'lag_7' in feature_names: row['lag_7'] = lag_values[-7]
        if 'lag_14' in feature_names: row['lag_14'] = lag_values[-14]
        if 'lag_30' in feature_names: row['lag_30'] = lag_values[-30]
        if 'dayofyear' in feature_names: row['dayofyear'] = doy
        if 'sin_day' in feature_names: row['sin_day'] = sin_day
        if 'cos_day' in feature_names: row['cos_day'] = cos_day

        x_next = np.array([[row.get(f, 0) for f in feature_names]])
        x_scaled = scaler.transform(x_next)

        y1 = model_lin.predict(x_scaled)[0]
        y2 = model_poly.predict(x_scaled)[0]
        y3 = model_non.predict(x_scaled)[0]

        pred_lin.append(round(y1, 1))
        pred_poly.append(round(y2, 1))
        pred_non.append(round(y3, 1))

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

# === Fungsi Kombinasi D (Rolling Window Update) ===
def predict_with_rolling(df, feature_names, label):
    df_copy = df.copy()
    pred_lin, pred_poly, pred_non, future_dates = [], [], [], []

    for i in range(1, 8):
        last_date = df_copy['tanggal'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        doy = next_date.dayofyear
        sin_day = sin(2 * pi * doy / 365)
        cos_day = cos(2 * pi * doy / 365)

        temp_df = df_copy.copy()
        temp_df['lag_7'] = temp_df['T2M'].shift(7)
        temp_df['lag_14'] = temp_df['T2M'].shift(14)
        temp_df['lag_30'] = temp_df['T2M'].shift(30)
        temp_df['dayofyear'] = temp_df['tanggal'].dt.dayofyear
        temp_df['sin_day'] = np.sin(2 * pi * temp_df['dayofyear'] / 365)
        temp_df['cos_day'] = np.cos(2 * pi * temp_df['dayofyear'] / 365)
        temp_df = temp_df.dropna()

        X = temp_df[feature_names].values
        y = temp_df['T2M'].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model_lin = LinearRegression().fit(X_scaled, y)
        model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_scaled, y)
        model_non = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42).fit(X_scaled, y)

        lag_7 = df_copy['T2M'].iloc[-7]
        lag_14 = df_copy['T2M'].iloc[-14]
        lag_30 = df_copy['T2M'].iloc[-30]

        row = {
            'lag_7': lag_7,
            'lag_14': lag_14,
            'lag_30': lag_30,
            'sin_day': sin_day,
            'cos_day': cos_day
        }

        x_next = np.array([[row[f] for f in feature_names]])
        x_scaled = scaler.transform(x_next)

        y1 = model_lin.predict(x_scaled)[0]
        y2 = model_poly.predict(x_scaled)[0]
        y3 = model_non.predict(x_scaled)[0]

        pred_lin.append(round(y1, 2))
        pred_poly.append(round(y2, 2))
        pred_non.append(round(y3, 2))
        future_dates.append(next_date)

        df_copy = pd.concat([df_copy, pd.DataFrame({'tanggal': [next_date], 'T2M': [y1]})], ignore_index=True)

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
results_D = predict_with_rolling(df, ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day'], "Kombinasi D: Lag + Musiman + Rolling")

# === Simpan Evaluasi ke CSV ===
df_evaluasi = pd.DataFrame(evaluasi_list)
df_evaluasi.to_csv("evaluasi_per_minggu6.csv", index=False)
print("\n✅ File 'evaluasi_per_minggu6.csv' berhasil disimpan.")

# === Visualisasi Kombinasi D ===
plt.figure(figsize=(10, 5))
plt.plot(results_D['tanggal'], results_D['Aktual'], label='Aktual', marker='o')
plt.plot(results_D['tanggal'], results_D['Pred_Linear'], label='Linear', marker='s')
plt.plot(results_D['tanggal'], results_D['Pred_Nonlinear'], label='Nonlinear', marker='^')
plt.plot(results_D['tanggal'], results_D['Pred_Polynomial'], label='Polynomial', marker='x')
plt.title("Prediksi T2M 1–7 Juni 2025")
plt.xlabel("Tanggal")
plt.ylabel("T2M (°C)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
