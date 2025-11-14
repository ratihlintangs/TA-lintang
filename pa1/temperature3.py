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
df = pd.read_excel("kumpulan3.xlsx")
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

# Drop NA
df = df.dropna()

# === 2. FITUR & TARGET ===
features = ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day']
X_all = df[features].values
y_all = df['Tavg'].values

# Skaler hanya fitur
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_all)

# === 3. TRAINING DATA ===
X_train = X_scaled
y_train = y_all

# === 4. MODEL TRAINING ===
model_linear = LinearRegression()
model_nonlinear = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
model_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

model_linear.fit(X_train, y_train)
model_nonlinear.fit(X_train, y_train)
model_poly.fit(X_train, y_train)

# === 5. RECURSIVE FORECASTING UNTUK 7 HARI ===
last_date = df['TANGGAL'].iloc[-1]
lag_values = list(df['Tavg'].values[-30:])  # last 30 values
preds_linear, preds_nonlinear, preds_poly = [], [], []
future_dates = []

for i in range(1, 8):
    next_date = last_date + pd.Timedelta(days=i)
    doy = next_date.dayofyear
    sin_day = sin(2 * pi * doy / 365)
    cos_day = cos(2 * pi * doy / 365)

    lag_7 = lag_values[-7]
    lag_14 = lag_values[-14]
    lag_30 = lag_values[-30]

    X_next = np.array([[lag_7, lag_14, lag_30, sin_day, cos_day]])
    X_next_scaled = scaler_X.transform(X_next)

    pred_lin = model_linear.predict(X_next_scaled)[0]
    pred_non = model_nonlinear.predict(X_next_scaled)[0]
    pred_pol = model_poly.predict(X_next_scaled)[0]

    preds_linear.append(pred_lin)
    preds_nonlinear.append(pred_non)
    preds_poly.append(pred_pol)
    future_dates.append(next_date)

    # update lag untuk iterasi berikutnya
    lag_values.append(pred_lin)  # atau bisa pred_non/pred_pol, tapi di sini kita pakai linier sebagai baseline lag

# === 6. DATA AKTUAL ===
actual = np.array([29.6, 29.2, 29.8, 28.3, 27.8, 27.4, 28.3])



# === 7. TABEL HASIL ===
results_df = pd.DataFrame({
    'Tanggal': future_dates,
    'Aktual': actual,
    'Pred_Linear': preds_linear,
    'Pred_Nonlinear': preds_nonlinear,
    'Pred_Polynomial': preds_poly
})
print("\n=== TABEL HASIL PREDIKSI 15–21 JUNI 2025 ===")
print(results_df)

# === 8. EVALUASI MASING-MASING MODEL ===
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

# === 9. GRAFIK PERBANDINGAN ===
plt.figure(figsize=(10, 5))
plt.plot(results_df['Tanggal'], results_df['Aktual'], label='Aktual', marker='o', linewidth=2)
plt.plot(results_df['Tanggal'], results_df['Pred_Linear'], label='Linear', marker='s')
plt.plot(results_df['Tanggal'], results_df['Pred_Nonlinear'], label='Nonlinear', marker='^')
plt.plot(results_df['Tanggal'], results_df['Pred_Polynomial'], label='Polinomial', marker='x')

plt.title("Perbandingan Suhu Aktual vs Prediksi (15–21 Juni 2025)")
plt.xlabel("Tanggal")
plt.ylabel("Suhu Rata-rata (°C)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
