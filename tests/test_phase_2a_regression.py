import pytest
from fastapi.testclient import TestClient

from app.api import app
from tests.test_api_integration import VALID_PAYLOAD


EXPECTED = [
    (30, 5, 2, 0, 0.03281707966926913, "high"),
    (60, 40, 2, 0, 0.16991580423470068, "high"),
    (45, 12, 1, 0, 0.4153779630366339, "medium"),
]


def test_predictions_match_pre_refactor_baseline():
    payloads = [dict(VALID_PAYLOAD, Age=age, Tenure=tenure, NumOfProducts=products) for age, tenure, products, _, _, _ in EXPECTED]
    with TestClient(app) as client:
        singles = [client.post("/predict", json=payload).json() for payload in payloads]
        batch = client.post("/predict-batch", json={"data": payloads}).json()["predictions"]
    for actual, batched, expected in zip(singles, batch, EXPECTED):
        assert actual["prediction"] == expected[3]
        assert actual["probability"] == pytest.approx(expected[4])
        assert actual["confidence"] == expected[5]
        assert batched == actual


def test_all_public_routes_are_registered():
    routes = {(method, route.path) for route in app.routes for method in getattr(route, "methods", set())}
    assert {("GET", "/health"), ("GET", "/model-info"), ("POST", "/predict"), ("POST", "/predict-batch"), ("POST", "/monitor/report")} <= routes


def test_request_id_and_monitoring_contract():
    files = {"reference": ("reference.csv", b"x\n1\n2\n", "text/csv"), "current": ("current.csv", b"x\n1\n2\n", "text/csv")}
    with TestClient(app) as client:
        response = client.post("/monitor/report", files=files, data={"alpha": "0.05"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json()["drifted_features"] == []
