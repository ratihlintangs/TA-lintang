import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# ========== 1. LOAD DATA ========== #
file_path = "kumpulan1.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# ========== 2. PREPROCESSING ========== #
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y')
df.sort_values('TANGGAL', inplace=True)
df['Tavg'] = pd.to_numeric(df['Tavg'], errors='coerce')
df = df[['TANGGAL', 'Tavg']].dropna().reset_index(drop=True)

# ========== 3. BUAT FITUR LAG ========== #
n_lags = 7
for i in range(1, n_lags + 1):
    df[f'lag_{i}'] = df['Tavg'].shift(i)
df.dropna(inplace=True)

X = df[[f'lag_{i}' for i in range(1, n_lags + 1)]].values
y = df['Tavg'].values
last_lag_data = X[-1].reshape(1, -1)

n_forecast = 7

# ========== 4. MODEL REGRESI LINIER ========== #
model_lin = LinearRegression()
model_lin.fit(X, y)
preds_linear = []
lag_temp = last_lag_data.copy()
for _ in range(n_forecast):
    pred = model_lin.predict(lag_temp)[0]
    preds_linear.append(round(pred, 1))  # <= diubah di sini
    lag_temp = np.roll(lag_temp, -1)
    lag_temp[0, -1] = pred

# ========== 5. MODEL POLINOMIAL ========== #
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
preds_poly = []
lag_temp = last_lag_data.copy()
for _ in range(n_forecast):
    lag_poly = poly.transform(lag_temp)
    pred = model_poly.predict(lag_poly)[0]
    preds_poly.append(round(pred, 1))  # <= diubah di sini
    lag_temp = np.roll(lag_temp, -1)
    lag_temp[0, -1] = pred

# ========== 6. MODEL NONLINIER (MLP) ========== #
model_nonlin = MLPRegressor(hidden_layer_sizes=(10,), max_iter=5000, random_state=42)
model_nonlin.fit(X, y)
preds_nonlinear = []
lag_temp = last_lag_data.copy()
for _ in range(n_forecast):
    pred = model_nonlin.predict(lag_temp)[0]
    preds_nonlinear.append(round(pred, 1))  # <= diubah di sini
    lag_temp = np.roll(lag_temp, -1)
    lag_temp[0, -1] = pred

# ========== 7. DATA AKTUAL ========== #
actual = [27.9, 28.7, 29.4, 29.6, 30.1, 28.8, 29.0]
dates = pd.date_range(start="2025-06-01", periods=7)

results = pd.DataFrame({
    'Tanggal': dates,
    'Aktual': actual,
    'Linear': preds_linear,
    'Polynomial': preds_poly,
    'Nonlinear': preds_nonlinear
})

print("=== HASIL PREDIKSI ===")
print(results)

# ========== 8. EVALUASI MODEL ========== #
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

eval_linear = evaluate_model(actual, preds_linear)
eval_poly = evaluate_model(actual, preds_poly)
eval_nonlin = evaluate_model(actual, preds_nonlinear)

evaluation_df = pd.DataFrame({
    'Linear': eval_linear,
    'Polynomial': eval_poly,
    'Nonlinear': eval_nonlin
}).T.round(4)

print("\n=== HASIL EVALUASI MODEL ===")
print(evaluation_df)

# ========== 9. PLOTTING ========== #
plt.figure(figsize=(10, 6))
plt.plot(dates, actual, label='Aktual', marker='o')
plt.plot(dates, preds_linear, label='Linear', marker='o')
plt.plot(dates, preds_poly, label='Polynomial', marker='o')
plt.plot(dates, preds_nonlinear, label='Nonlinear', marker='o')
plt.title("Prediksi Tavg Autoregressive vs Aktual (1–7 Juni 2025)")
plt.xlabel("Tanggal")
plt.ylabel("Tavg (°C)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
