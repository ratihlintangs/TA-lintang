import numpy as np
import pandas as pd
from math import sin, cos, pi
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from fastapi import HTTPException

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from backend.services.predict_service import PredictService
from backend.database import load_data_from_db



TARGET_COLUMNS = [
    'temperature',
    'humidity',
    'pressure',
    'wind_speed'
]


class EvaluationService:
    """
    Service untuk melakukan evaluasi model time series
    menggunakan rolling / walk-forward evaluation.
    """

    def __init__(self):
        self.predict_service = PredictService()

    def _create_features(
        self,
        historic_values: List[float],
        target_date
    ) -> np.ndarray:
        """
        Membuat satu baris fitur input sesuai predict_service.py
        """
        lag_7 = historic_values[-7]
        lag_14 = historic_values[-14]
        lag_30 = historic_values[-30]

        dayofyear = target_date.timetuple().tm_yday
        sin_day = sin(2 * pi * dayofyear / 365.25)
        cos_day = cos(2 * pi * dayofyear / 365.25)
        is_weekday = 1 if target_date.weekday() < 5 else 0

        return np.array([[lag_7, lag_14, lag_30, sin_day, cos_day, is_weekday]])

    def evaluate(
        self,
        db: Session,
        target_col: str,
        test_days: int = 7
    ) -> Dict[str, Any]:
        """
        Melakukan rolling evaluation untuk satu parameter cuaca.
        """

        assets = self.predict_service.model_assets.get(target_col)
        if not assets:
            raise HTTPException(
                status_code=500,
                detail=f"Aset model untuk {target_col} tidak tersedia."
            )

        model = assets['model']
        scaler_X = assets['scaler_X']
        scaler_y = assets['scaler_y']

        # === Load data historis ===
        df_raw = load_data_from_db(db)
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['date', target_col])
        df_raw = df_raw.sort_values('date')

        if len(df_raw) < 30 + test_days:
            raise HTTPException(
                status_code=404,
                detail=f"Data tidak cukup untuk evaluasi {target_col}"
            )

        # Split train & test (time-based)
        df_train = df_raw.iloc[:-test_days].copy()
        df_test = df_raw.iloc[-test_days:].copy()

        # Ambil 30 hari terakhir dari data train
        historic_values = df_train[target_col].tail(30).tolist()

        y_true = []
        y_pred = []

        # === Rolling prediction ===
        for idx, row in df_test.iterrows():
            target_date = row['date']

            X_input = self._create_features(historic_values, target_date)
            X_scaled = scaler_X.transform(X_input)

            y_scaled_pred = model.predict(X_scaled)
            y_hat = scaler_y.inverse_transform(
                y_scaled_pred.reshape(-1, 1)
            )[0][0]

            y_true.append(row[target_col])
            y_pred.append(y_hat)

            # rolling update
            historic_values.append(y_hat)

        # === Metrics ===
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "parameter": target_col,
            "periode_awal": df_test['date'].min().strftime('%Y-%m-%d'),
            "periode_akhir": df_test['date'].max().strftime('%Y-%m-%d'),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4)
        }

    def evaluate_all(
        self,
        db: Session,
        test_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Evaluasi seluruh parameter cuaca.
        """
        results = []

        for target_col in TARGET_COLUMNS:
            try:
                result = self.evaluate(db, target_col, test_days)
                results.append(result)
            except Exception as e:
                results.append({
                    "parameter": target_col,
                    "error": str(e)
                })

        return results
