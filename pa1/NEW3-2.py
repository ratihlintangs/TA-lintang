import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sin, cos, pi

# === 1. LOAD & PREPROCESSING ===
df = pd.read_excel("kumpulan2.xlsx")
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], dayfirst=True)
df = df.sort_values('TANGGAL')

df['Tavg'] = df['Tavg'].replace('-', np.nan).astype(float)
df = df.dropna(subset=['Tavg'])

# Tambah fitur lag & musiman
df['lag_7'] = df['Tavg'].shift(7)
df['lag_14'] = df['Tavg'].shift(14)
df['lag_30'] = df['Tavg'].shift(30)
df['dayofyear'] = df['TANGGAL'].dt.dayofyear
df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365)

df = df.dropna()

# === 2. FITUR & TARGET ===
features = ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day']
X_all = df[features].values
y_all = df['Tavg'].values

# Scaling fitur (TANPA scaling target)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_all)

# === 3. TRAINING ===
model_linear = LinearRegression()
model_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model_nonlinear = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)

model_linear.fit(X_scaled, y_all)
model_poly.fit(X_scaled, y_all)
model_nonlinear.fit(X_scaled, y_all)

# === 4. PREDIKSI REKURSIF 7 HARI ===
last_date = df['TANGGAL'].iloc[-1]
lag_values = list(df['Tavg'].values[-30:])  # ambil 30 hari terakhir
preds_linear, preds_poly, preds_nonlinear = [], [], []
future_dates = []

# Tentukan tanggal yang ingin diprediksi secara eksplisit
future_dates = pd.date_range(start="2025-06-08", end="2025-06-14")

for next_date in future_dates:
    doy = next_date.dayofyear
    sin_day = sin(2 * pi * doy / 365)
    cos_day = cos(2 * pi * doy / 365)

    lag_7 = lag_values[-7]
    lag_14 = lag_values[-14]
    lag_30 = lag_values[-30]

    X_next = np.array([[lag_7, lag_14, lag_30, sin_day, cos_day]])
    X_next_scaled = scaler_X.transform(X_next)

    pred_lin = model_linear.predict(X_next_scaled)[0]
    pred_pol = model_poly.predict(X_next_scaled)[0]
    pred_non = model_nonlinear.predict(X_next_scaled)[0]

    preds_linear.append(round(pred_lin, 1))
    preds_poly.append(round(pred_pol, 1))
    preds_nonlinear.append(round(pred_non, 1))

    lag_values.append(pred_lin)  # tetap update lag menggunakan model linear


# === 5. DATA AKTUAL ===
actual = np.array([29.2, 26.9, 28.4, 29.3, 28.7, 28.5, 28.7])


# === 6. HASIL TABEL PREDIKSI ===
results_df = pd.DataFrame({
    'Tanggal': future_dates,
    'Aktual': actual,
    'Pred_Linear': preds_linear,
    'Pred_Polynomial': preds_poly,
    'Pred_Nonlinear': preds_nonlinear
})
print("\n=== TABEL HASIL PREDIKSI 1–7 JUNI 2025 ===")
print(results_df)

# === 7. EVALUASI MASING-MASING MODEL ===
def evaluate_model(y_true, y_pred, name):
    print(f"\n=== Evaluasi Model: {name} ===")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")

evaluate_model(actual, preds_linear, "Linear Regression")
evaluate_model(actual, preds_nonlinear, "Neural Network (Nonlinear)")
evaluate_model(actual, preds_poly, "Polynomial Regression")

# === 8. GRAFIK PERBANDINGAN ===
plt.figure(figsize=(10, 5))
plt.plot(results_df['Tanggal'], results_df['Aktual'], label='Aktual', marker='o', linewidth=2)
plt.plot(results_df['Tanggal'], results_df['Pred_Linear'], label='Linear', marker='s')
plt.plot(results_df['Tanggal'], results_df['Pred_Nonlinear'], label='Nonlinear', marker='^')
plt.plot(results_df['Tanggal'], results_df['Pred_Polynomial'], label='Polinomial', marker='x')

plt.title("Perbandingan Suhu Aktual vs Prediksi (8–14 Juni 2025)")
plt.xlabel("Tanggal")
plt.ylabel("Suhu Rata-rata (°C)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
