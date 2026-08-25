"""Root and health routes."""

from fastapi import APIRouter, Depends
from app.dependencies import get_model_service
from app.schema import HealthCheck
from app.services.model_service import ModelService

router = APIRouter()

@router.get("/health", response_model=HealthCheck)
async def health_check(service: ModelService = Depends(get_model_service)) -> HealthCheck:
    return service.get_health_status()

@router.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs", "health": "/health", "model_info": "/model-info", "endpoints": ["POST /predict", "POST /predict-batch", "POST /monitor/report"]}
