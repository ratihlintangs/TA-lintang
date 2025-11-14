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

# ========== 3. FITUR LAG ========== #
lag_list = [1, 2, 3, 4, 5, 6, 7, 14, 30]
for i in lag_list:
    df[f'lag_{i}'] = df['Tavg'].shift(i)

# ========== 4. FITUR MUSIMAN (CYCLICAL) ========== #
df['sin_doy'] = np.sin(2 * np.pi * df['TANGGAL'].dt.dayofyear / 365)
df['cos_doy'] = np.cos(2 * np.pi * df['TANGGAL'].dt.dayofyear / 365)

df.dropna(inplace=True)

# ========== 5. PERSIAPAN DATA TRAINING ========== #
X = df[[f'lag_{i}' for i in lag_list] + ['sin_doy', 'cos_doy']].values
y = df['Tavg'].values
last_row = df.iloc[-1]
last_lag_data = X[-1].reshape(1, -1)

n_forecast = 7
start_date = last_row['TANGGAL'] + pd.Timedelta(days=1)

# ========== 6. FUNGSI UNTUK FITUR MUSIMAN BARU ========== #
def get_seasonal_features(date):
    doy = date.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * doy / 365)
    cos_doy = np.cos(2 * np.pi * doy / 365)
    return sin_doy, cos_doy

# ========== 7. MODEL REGRESI LINIER ========== #
model_lin = LinearRegression()
model_lin.fit(X, y)
preds_linear = []
lag_temp = last_lag_data[0, :-2]  # exclude sin/cos
for i in range(n_forecast):
    date = start_date + pd.Timedelta(days=i)
    sin_doy, cos_doy = get_seasonal_features(date)
    input_data = np.append(lag_temp, [sin_doy, cos_doy]).reshape(1, -1)
    pred = model_lin.predict(input_data)[0]
    preds_linear.append(round(pred, 1))
    lag_temp = np.roll(lag_temp, -1)
    lag_temp[-1] = pred

# ========== 8. MODEL POLINOMIAL ========== #
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
preds_poly = []
lag_temp = last_lag_data[0, :-2]
for i in range(n_forecast):
    date = start_date + pd.Timedelta(days=i)
    sin_doy, cos_doy = get_seasonal_features(date)
    input_data = np.append(lag_temp, [sin_doy, cos_doy]).reshape(1, -1)
    pred = model_poly.predict(poly.transform(input_data))[0]
    preds_poly.append(round(pred, 1))
    lag_temp = np.roll(lag_temp, -1)
    lag_temp[-1] = pred

# ========== 9. MODEL NONLINIER (MLP) ========== #
model_nonlin = MLPRegressor(hidden_layer_sizes=(10,), max_iter=5000, random_state=42)
model_nonlin.fit(X, y)
preds_nonlinear = []
lag_temp = last_lag_data[0, :-2]
for i in range(n_forecast):
    date = start_date + pd.Timedelta(days=i)
    sin_doy, cos_doy = get_seasonal_features(date)
    input_data = np.append(lag_temp, [sin_doy, cos_doy]).reshape(1, -1)
    pred = model_nonlin.predict(input_data)[0]
    preds_nonlinear.append(round(pred, 1))
    lag_temp = np.roll(lag_temp, -1)
    lag_temp[-1] = pred

# ========== 10. HASIL & EVALUASI ========== #
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

# ========== 11. PLOTTING ========== #
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
