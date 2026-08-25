"""Simple smoke test for FastAPI endpoints (health & predict).

Usage:
  .venv\Scripts\python.exe scripts\smoke_test_api.py
"""
import requests
import sys

BASE = "http://localhost:8000"


def check_health():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        r.raise_for_status()
        print("[OK] /health ->", r.json())
        return True
    except Exception as e:
        print("[FAIL] /health ->", e)
        return False


def check_model_info():
    try:
        r = requests.get(f"{BASE}/model-info", timeout=3)
        r.raise_for_status()
        print("[OK] /model-info ->", r.json())
        return True
    except Exception as e:
        print("[FAIL] /model-info ->", e)
        return False


def check_predict():
    payload = {
        "SeniorCitizen": 0,
        "Age": 45,
        "NumOfProducts": 2,
        "Tenure": 24,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1570.0,
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    }
    try:
        r = requests.post(f"{BASE}/predict", json=payload, timeout=5)
        r.raise_for_status()
        print("[OK] /predict ->", r.json())
        return True
    except Exception as e:
        print("[FAIL] /predict ->", e)
        return False


if __name__ == "__main__":
    ok = True
    ok &= check_health()
    ok &= check_model_info()
    ok &= check_predict()

    if not ok:
        print("Smoke test failed. Ensure FastAPI is running: `python -m uvicorn app.api:app --reload`")
        sys.exit(1)
    print("Smoke test passed!")
