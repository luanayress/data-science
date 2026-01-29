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
    """Schema for prediction request."""
    
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaymentMethod: str
    
    class Config:
        schema_extra = {
            "example": {
                "SeniorCitizen": 0,
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
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    prediction: int = Field(..., ge=0, le=1, description="0=No Churn, 1=Churn")
    probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    confidence: str = Field(..., description="Confidence level: high, medium, low")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.25,
                "confidence": "high"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    
    data: List[PredictionRequest]
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "SeniorCitizen": 0,
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
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    
    predictions: List[PredictionResponse]
    total_samples: int
    
    class Config:
        schema_extra = {
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


class ModelInfo(BaseModel):
    """Schema for model information."""
    
    model_type: str
    version: str
    trained_at: Optional[str]
    accuracy: Optional[float]
    f1_score: Optional[float]
    n_features: int
    feature_names: List[str]


class HealthCheck(BaseModel):
    """Schema for health check response."""
    
    status: str
    model_loaded: bool
    version: str
    timestamp: str
