import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sin, cos, pi

df = pd.read_excel("kumpulan1.xlsx")
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], dayfirst=True)
df = df.sort_values('TANGGAL')

target_col = 'FF_AVG'
df[target_col] = df[target_col].replace('-', np.nan).astype(float)
df = df.dropna(subset=[target_col])

df['lag_7'] = df[target_col].shift(7)
df['lag_14'] = df[target_col].shift(14)
df['lag_30'] = df[target_col].shift(30)
df['dayofyear'] = df['TANGGAL'].dt.dayofyear
df['sin_day'] = np.sin(2 * pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * pi * df['dayofyear'] / 365)
df = df.dropna()

features = ['lag_7', 'lag_14', 'lag_30', 'sin_day', 'cos_day']
X_all = df[features].values
y_all = df[target_col].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_all)

model_linear = LinearRegression()
model_nonlinear = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
model_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

model_linear.fit(X_scaled, y_all)
model_nonlinear.fit(X_scaled, y_all)
model_poly.fit(X_scaled, y_all)

last_date = df['TANGGAL'].iloc[-1]
lag_vals = list(df[target_col].values[-30:])
pred_lin, pred_non, pred_poly, dates = [], [], [], []

for i in range(7):
    next_date = last_date + pd.Timedelta(days=i + 1)
    doy = next_date.dayofyear
    sin_day = sin(2 * pi * doy / 365)
    cos_day = cos(2 * pi * doy / 365)
    x_input = np.array([[lag_vals[-7], lag_vals[-14], lag_vals[-30], sin_day, cos_day]])
    x_scaled = scaler_X.transform(x_input)

    y_lin = model_linear.predict(x_scaled)[0]
    y_non = model_nonlinear.predict(x_scaled)[0]
    y_pol = model_poly.predict(x_scaled)[0]

    pred_lin.append(y_lin)
    pred_non.append(y_non)
    pred_poly.append(y_pol)
    dates.append(next_date)

    lag_vals.append(y_lin)

# Data aktual kecepatan angin (fff_avg) 1–7 Juni
aktual = [1, 1, 2, 2, 2, 1, 1]
results = pd.DataFrame({
    'Tanggal': dates,
    'Aktual': aktual,
    'Pred_Linear': pred_lin,
    'Pred_Nonlinear': pred_non,
    'Pred_Polynomial': pred_poly
})

print("=== TABEL PREDIKSI fff_avg ===")
print(results)

def evaluate(y_true, y_pred, name):
    print(f"\n-- Evaluasi Model: {name} --")
    print(f"MSE  : {mean_squared_error(y_true, y_pred):.3f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"R²   : {r2_score(y_true, y_pred):.3f}")

evaluate(aktual, pred_lin, "Linear")
evaluate(aktual, pred_non, "Neural Network")
evaluate(aktual, pred_poly, "Polynomial")

plt.figure(figsize=(10, 5))
plt.plot(dates, aktual, label='Aktual', marker='o', linewidth=2)
plt.plot(dates, pred_lin, label='Linear', marker='s')
plt.plot(dates, pred_non, label='Nonlinear', marker='^')
plt.plot(dates, pred_poly, label='Polynomial', marker='x')
plt.title("Prediksi fff_avg (1–7 Juni 2025)")
plt.xlabel("Tanggal")
plt.ylabel("Kecepatan Angin (m/s)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
