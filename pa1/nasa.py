import requests
import json

lat, lon = -7.9007, 112.5978
start, end = "20200101", "20251231"
params = ",".join(["T2M","RH2M","WS2M","PS"])  # suhu, kelembaban, angin, tekanan

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters={params}&community=RE&latitude={lat}&longitude={lon}"
    f"&start={start}&end={end}&format=JSON"
)

res = requests.get(url, timeout=30)
data = res.json()

# Simpan sebagai CSV atau olah sendiri
with open("karangploso_power.json", "w") as f:
    json.dump(data, f)
print("Data berhasil diambil")
