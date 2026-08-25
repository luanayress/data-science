"""Pydantic schemas for data validation."""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ContractType(str, Enum):
    """Contract type options."""
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class InternetServiceType(str, Enum):
    """Internet service type."""
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class PredictionRequest(BaseModel):
    """Backward-compatible bank churn request.

    Age, Tenure and NumOfProducts remain required for v2/v3. Expanded bank
    fields are consumed by v4; legacy telecom fields remain optional while
    clients migrate.
    """
    
    Age: int = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=0)
    Tenure: int = Field(..., ge=0)
    CreditScore: Optional[int] = Field(None, ge=0, le=1000)
    Geography: Optional[str] = None
    Gender: Optional[str] = None
    Balance: Optional[float] = Field(None, ge=0)
    HasCrCard: Optional[int] = Field(None, ge=0, le=1)
    IsActiveMember: Optional[int] = Field(None, ge=0, le=1)
    EstimatedSalary: Optional[float] = Field(None, ge=0)
    SatisfactionScore: Optional[int] = Field(None, ge=1, le=5)
    CardType: Optional[str] = None
    PointEarned: Optional[int] = Field(None, ge=0)

    # Deprecated telecom fields retained for API compatibility with v2 clients.
    SeniorCitizen: Optional[int] = Field(None, ge=0, le=1)
    MonthlyCharges: Optional[float] = Field(None, gt=0)
    TotalCharges: Optional[float] = Field(None, ge=0)
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaymentMethod: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "CreditScore": 650, "Geography": "France", "Gender": "Female",
                "Age": 45, "Tenure": 5, "Balance": 100000.0,
                "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
                "EstimatedSalary": 75000.0, "SatisfactionScore": 3,
                "CardType": "GOLD", "PointEarned": 500
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    prediction: int = Field(..., ge=0, le=1, description="0=No Churn, 1=Churn")
    probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    confidence: str = Field(..., description="Confidence level: high, medium, low")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0,
                "probability": 0.25,
                "confidence": "high"
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    
    data: List[PredictionRequest]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    {
                        "CreditScore": 650, "Geography": "France", "Gender": "Female",
                        "Age": 45, "Tenure": 5, "Balance": 100000.0,
                        "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
                        "EstimatedSalary": 75000.0, "SatisfactionScore": 3,
                        "CardType": "GOLD", "PointEarned": 500
                    }
                ]
            }
        }
    }


class BankPredictionRequest(BaseModel):
    """Strict input contract for the selected v4 bank feature set."""

    CreditScore: int = Field(..., ge=0, le=1000)
    Age: int = Field(..., ge=18, le=120)
    Tenure: int = Field(..., ge=0)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=0)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)
    Geography: str = Field(..., min_length=1)
    Gender: str = Field(..., min_length=1)


class BankBatchPredictionRequest(BaseModel):
    data: List[BankPredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    
    predictions: List[PredictionResponse]
    total_samples: int
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "prediction": 0,
                        "probability": 0.25,
                        "confidence": "high"
                    }
                ],
                "total_samples": 1
            }
        }
    }


class ModelInfo(BaseModel):
    """Schema for model information."""
    model_type: str
    version: str
    trained_at: Optional[str]
    accuracy: Optional[float]
    f1_score: Optional[float]
    n_features: int
    feature_names: List[str]
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_type": "GradientBoostingClassifier",
                "version": "v1",
                "trained_at": "2026-01-28T00:00:00Z",
                "accuracy": 0.87,
                "f1_score": 0.81,
                "n_features": 18,
                "feature_names": ["age", "tenure", "...etc..."]
            }
        }
    }


class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    version: str
    timestamp: str
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "v1",
                "timestamp": "2026-02-02T12:00:00Z"
            }
        }
    }
