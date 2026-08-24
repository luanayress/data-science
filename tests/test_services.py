import io

import pandas as pd
import pytest

from app.core.exceptions import ModelUnavailableError
from app.schema import PredictionRequest
from app.services.churn_service import ChurnService
from app.services.model_service import ModelService
from app.services.monitoring_service import MonitoringService


PAYLOAD = {
    "SeniorCitizen": 0, "Age": 45, "NumOfProducts": 2, "Tenure": 24,
    "MonthlyCharges": 65.5, "TotalCharges": 1570.0,
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
}


class FakePipeline:
    version = "v2"
    metadata = {"version": "v2", "algorithm": "Fake", "model_features": ["x"], "metrics": {"accuracy": 0.8, "f1": 0.7}}

    def __init__(self):
        self.calls = 0

    def predict_with_confidence(self, frame):
        self.calls += 1
        assert isinstance(frame, pd.DataFrame)
        return {"predictions": [0] * len(frame), "probabilities": [0.2] * len(frame), "high_confidence": [True] * len(frame)}


def test_churn_service_delegates_without_loading_or_fitting():
    pipeline = FakePipeline()
    service = ChurnService(pipeline)
    request = PredictionRequest(**PAYLOAD)
    single = service.predict_one(request)
    batch = service.predict_batch([request, request])
    assert single.probability == 0.2
    assert batch.total_samples == 2
    assert pipeline.calls == 2


def test_model_service_health_and_metadata():
    service = ModelService(FakePipeline(), "v2")
    assert service.get_health_status().model_loaded is True
    assert service.get_model_info().feature_names == ["x"]
    with pytest.raises(ModelUnavailableError):
        ModelService(None, "v2").get_model_info()


def test_monitoring_service_coordinates_csvs():
    class Monitor:
        def detect_drift(self, reference, current, alpha):
            assert list(reference.columns) == ["x"]
            assert alpha == 0.05
            return {"drifted_features": [], "details": {}}

    csv_bytes = b"x\n1\n2\n"
    assert MonitoringService(Monitor()).create_report(csv_bytes, csv_bytes, 0.05)["drifted_features"] == []
