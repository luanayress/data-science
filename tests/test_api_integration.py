from fastapi.testclient import TestClient

from app.api import app


VALID_PAYLOAD = {
    "SeniorCitizen": 0, "Age": 45, "NumOfProducts": 2, "Tenure": 24,
    "MonthlyCharges": 65.5, "TotalCharges": 1570.0,
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
}

BANK_PAYLOAD = {
    "CreditScore": 650, "Geography": "France", "Gender": "Female",
    "Age": 45, "Tenure": 5, "Balance": 100000.0, "NumOfProducts": 2,
    "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 75000.0,
}


def test_real_api_routes():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["version"] == "v2"
        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["feature_names"] == [
            "NumOfProducts", "Age_Squared", "Age_Tenure_Interaction"
        ]
        prediction = client.post("/predict", json=VALID_PAYLOAD)
        assert prediction.status_code == 200
        assert 0 <= prediction.json()["probability"] <= 1
        batch = client.post("/predict-batch", json={"data": [VALID_PAYLOAD, VALID_PAYLOAD]})
        assert batch.status_code == 200
        assert batch.json()["total_samples"] == 2


def test_invalid_requests_are_rejected():
    with TestClient(app) as client:
        missing = dict(VALID_PAYLOAD)
        missing.pop("Age")
        assert client.post("/predict", json=missing).status_code == 422
        wrong_type = dict(VALID_PAYLOAD, Age="not-a-number")
        assert client.post("/predict", json=wrong_type).status_code == 422
        assert client.post("/predict-batch", json={"data": [missing]}).status_code == 422


def test_strict_v4_routes_require_complete_bank_contract_and_cache_model():
    with TestClient(app) as client:
        prediction = client.post("/v4/predict", json=BANK_PAYLOAD)
        assert prediction.status_code == 200
        assert 0 <= prediction.json()["probability"] <= 1
        loaded = app.state.v4_pipeline
        assert client.post("/v4/predict", json=BANK_PAYLOAD).status_code == 200
        assert app.state.v4_pipeline is loaded
        incomplete = dict(BANK_PAYLOAD)
        incomplete.pop("Balance")
        assert client.post("/v4/predict", json=incomplete).status_code == 422


def test_model_unavailable_returns_503():
    with TestClient(app) as client:
        pipeline = app.state.pipeline
        del app.state.pipeline
        try:
            assert client.get("/model-info").status_code == 503
            assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503
        finally:
            app.state.pipeline = pipeline
