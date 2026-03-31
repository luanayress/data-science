"""Unit tests for API and application."""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.schema import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, HealthCheck
)


def test_prediction_request_schema():
    """Test prediction request schema validation."""
    
    valid_data = {
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
    
    request = PredictionRequest(**valid_data)
    assert request.SeniorCitizen == 0
    assert request.Tenure == 24
    print("✓ test_prediction_request_schema passed")


def test_prediction_response_schema():
    """Test prediction response schema."""
    
    response = PredictionResponse(
        prediction=0,
        probability=0.25,
        confidence="high"
    )
    
    assert response.prediction == 0
    assert response.probability == 0.25
    assert response.confidence == "high"
    print("✓ test_prediction_response_schema passed")


def test_batch_prediction_request_schema():
    """Test batch prediction request schema."""
    
    valid_data = {
        "data": [
            {
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
        ]
    }
    
    batch_request = BatchPredictionRequest(**valid_data)
    assert len(batch_request.data) == 1
    print("✓ test_batch_prediction_request_schema passed")


def test_batch_prediction_response_schema():
    """Test batch prediction response schema."""
    
    batch_response = BatchPredictionResponse(
        predictions=[
            PredictionResponse(prediction=0, probability=0.25, confidence="high"),
            PredictionResponse(prediction=1, probability=0.75, confidence="high")
        ],
        total_samples=2
    )
    
    assert batch_response.total_samples == 2
    assert len(batch_response.predictions) == 2
    print("✓ test_batch_prediction_response_schema passed")


def test_model_info_schema():
    """Test model info schema."""
    
    model_info = ModelInfo(
        model_type="GradientBoostingClassifier",
        version="v1",
        trained_at="2024-01-28T10:00:00",
        accuracy=0.85,
        f1_score=0.82,
        n_features=12,
        feature_names=["feature1", "feature2", "feature3"]
    )
    
    assert model_info.model_type == "GradientBoostingClassifier"
    assert model_info.n_features == 12
    print("✓ test_model_info_schema passed")


def test_health_check_schema():
    """Test health check schema."""
    
    health = HealthCheck(
        status="healthy",
        model_loaded=True,
        version="v1",
        timestamp="2024-01-28T10:00:00"
    )
    
    assert health.status == "healthy"
    assert health.model_loaded is True
    print("✓ test_health_check_schema passed")


def test_schema_serialization():
    """Test schema serialization to JSON."""
    
    response = PredictionResponse(
        prediction=0,
        probability=0.25,
        confidence="high"
    )
    
    json_str = response.model_dump_json()
    parsed = json.loads(json_str)
    
    assert parsed['prediction'] == 0
    assert parsed['probability'] == 0.25
    print("✓ test_schema_serialization passed")


if __name__ == "__main__":
    test_prediction_request_schema()
    test_prediction_response_schema()
    test_batch_prediction_request_schema()
    test_batch_prediction_response_schema()
    test_model_info_schema()
    test_health_check_schema()
    test_schema_serialization()
    print("\n✅ All API/schema tests passed!")
