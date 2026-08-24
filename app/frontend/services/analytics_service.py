"""Analytics data provider boundary with an explicit demo source."""

from enum import Enum

import numpy as np
import pandas as pd


class AnalyticsSource(str, Enum):
    REAL = "REAL"
    DEMO = "DEMO"
    UNAVAILABLE = "UNAVAILABLE"


class AnalyticsDataProvider:
    """Provide analytics data; real/history adapters can replace this implementation."""

    source = AnalyticsSource.DEMO

    def get_data(self, n_samples: int = 100) -> pd.DataFrame:
        random = np.random.RandomState(42)
        return pd.DataFrame({
            "Tenure": random.randint(0, 72, n_samples),
            "MonthlyCharges": random.uniform(20, 120, n_samples),
            "TotalCharges": random.uniform(0, 8000, n_samples),
            "Prediction": random.choice(["Stay", "Churn"], n_samples, p=[0.8, 0.2]),
            "Probability": random.uniform(0, 1, n_samples),
        })
