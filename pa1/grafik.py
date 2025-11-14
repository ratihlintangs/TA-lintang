import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data hasil evaluasi dari April hingga Juni
data = [
    # Format: [bulan, minggu, kombinasi, model, MSE, RMSE, MAE, R2]
    ['April', 1, 'A', 'Linear', 0.483, 0.695, 0.600, -0.322],
    ['April', 1, 'A', 'Neural', 0.483, 0.695, 0.600, -0.322],
    ['April', 1, 'A', 'Poly', 0.429, 0.655, 0.571, -0.173],
    ['April', 1, 'B', 'Linear', 0.684, 0.827, 0.643, -0.873],
    ['April', 1, 'B', 'Neural', 0.684, 0.827, 0.643, -0.873],
    ['April', 1, 'B', 'Poly', 0.890, 0.943, 0.700, -1.436],
    ['April', 1, 'C', 'Linear', 0.771, 0.878, 0.686, -1.112],
    ['April', 1, 'C', 'Neural', 0.753, 0.868, 0.671, -1.061],
    ['April', 1, 'C', 'Poly', 0.529, 0.727, 0.543, -0.447],
    ['April', 1, 'D', 'Linear', 0.771, 0.878, 0.686, -1.112],
    ['April', 1, 'D', 'Neural', 0.770, 0.877, 0.671, -1.108],
    ['April', 1, 'D', 'Poly', 0.490, 0.700, 0.529, -0.341],
    ['Mei', 1, 'A', 'Linear', 0.556, 0.745, 0.671, -1.613],
    ['Mei', 1, 'A', 'Neural', 0.556, 0.745, 0.671, -1.613],
    ['Mei', 1, 'A', 'Poly', 0.449, 0.670, 0.600, -1.109],
    ['Mei', 1, 'B', 'Linear', 0.389, 0.623, 0.571, -0.827],
    ['Mei', 1, 'B', 'Neural', 0.369, 0.607, 0.543, -0.733],
    ['Mei', 1, 'B', 'Poly', 0.414, 0.644, 0.571, -0.948],
    ['Mei', 1, 'C', 'Linear', 0.389, 0.623, 0.571, -0.827],
    ['Mei', 1, 'C', 'Neural', 0.389, 0.623, 0.571, -0.827],
    ['Mei', 1, 'C', 'Poly', 0.331, 0.576, 0.514, -0.559],
    ['Mei', 1, 'D', 'Linear', 0.389, 0.623, 0.571, -0.827],
    ['Mei', 1, 'D', 'Neural', 0.351, 0.593, 0.543, -0.653],
    ['Mei', 1, 'D', 'Poly', 0.314, 0.561, 0.514, -0.478],
    ['Juni', 4, 'A', 'Linear', 0.539, 0.734, 0.557, -0.010],
    ['Juni', 4, 'A', 'Neural', 0.539, 0.734, 0.557, -0.010],
    ['Juni', 4, 'A', 'Poly', 0.539, 0.734, 0.557, -0.010],
    ['Juni', 4, 'B', 'Linear', 0.636, 0.797, 0.614, -0.192],
    ['Juni', 4, 'B', 'Neural', 0.633, 0.796, 0.614, -0.186],
    ['Juni', 4, 'B', 'Poly', 0.593, 0.770, 0.586, -0.111],
    ['Juni', 4, 'C', 'Linear', 0.666, 0.816, 0.629, -0.248],
    ['Juni', 4, 'C', 'Neural', 0.619, 0.786, 0.614, -0.160],
    ['Juni', 4, 'C', 'Poly', 0.680, 0.825, 0.600, -0.275],
    ['Juni', 4, 'D', 'Linear', 0.651, 0.807, 0.629, -0.221],
    ['Juni', 4, 'D', 'Neural', 0.659, 0.812, 0.614, -0.235],
    ['Juni', 4, 'D', 'Poly', 0.676, 0.822, 0.586, -0.267],
]

columns = ['Bulan', 'Minggu', 'Kombinasi', 'Model', 'MSE', 'RMSE', 'MAE', 'R2']
df = pd.DataFrame(data, columns=columns)
df['Bulan_Minggu'] = df['Bulan'] + ' M' + df['Minggu'].astype(str)

import seaborn as sns
sns.set(style="whitegrid")

def plot_metric(metric):
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x='Bulan_Minggu',
        y=metric,
        hue='Model',
        style='Kombinasi',
        markers=True,
        dashes=False
    )
    plt.title(f'Perbandingan {metric} dari April hingga Juni')
    plt.xlabel('Waktu')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

for metric in ['MSE', 'RMSE', 'MAE', 'R2']:
    plot_metric(metric)
