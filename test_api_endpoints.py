"""
Test script for API endpoints and predictions
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("🧪 API ENDPOINT TESTING")
print("=" * 70)

# Test 1: Health check
print("\n1️⃣  Health Check")
try:
    resp = requests.get(f"{BASE_URL}/health")
    health = resp.json()
    print(f"   Status: {health['status']}")
    print(f"   Model Loaded: {health['model_loaded']}")
    print(f"   Version: {health['version']}")
    print(f"   ✅ PASSED")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 2: Model info
print("\n2️⃣  Model Information")
try:
    resp = requests.get(f"{BASE_URL}/model-info")
    info = resp.json()
    print(f"   Model Type: {info['model_type']}")
    print(f"   Features: {', '.join(info['features'])}")
    print(f"   ✅ PASSED")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 3: Single prediction
print("\n3️⃣  Single Prediction")
try:
    test_data = {
        "NumOfProducts": 2,
        "Age_Squared_StandardScaled": 0.5,
        "Age_Tenure_Interaction_MinMaxScaled": 0.3
    }
    resp = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    pred = resp.json()
    print(f"   Prediction: {pred['prediction']}")
    print(f"   Probability: {pred['probability']:.2%}")
    print(f"   ✅ PASSED")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 4: Batch prediction
print("\n4️⃣  Batch Prediction")
try:
    batch_data = {
        "data": [
            {"NumOfProducts": 1, "Age_Squared_StandardScaled": 0.2, "Age_Tenure_Interaction_MinMaxScaled": 0.1},
            {"NumOfProducts": 3, "Age_Squared_StandardScaled": 0.8, "Age_Tenure_Interaction_MinMaxScaled": 0.7},
        ]
    }
    resp = requests.post(
        f"{BASE_URL}/predict-batch",
        json=batch_data,
        headers={"Content-Type": "application/json"}
    )
    preds = resp.json()
    print(f"   Predictions made: {len(preds['predictions'])}")
    for i, p in enumerate(preds['predictions']):
        print(f"      Sample {i+1}: {p['prediction']} ({p['probability']:.2%})")
    print(f"   ✅ PASSED")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 5: API Documentation
print("\n5️⃣  API Documentation")
try:
    resp = requests.get(f"{BASE_URL}/docs")
    if resp.status_code == 200:
        print(f"   Swagger UI: http://localhost:8000/docs")
        print(f"   ReDoc: http://localhost:8000/redoc")
        print(f"   ✅ PASSED")
    else:
        print(f"   ❌ FAILED: Status {resp.status_code}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "=" * 70)
print("✅ API TESTING COMPLETE")
print("=" * 70)
