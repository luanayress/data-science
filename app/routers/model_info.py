"""Model metadata route."""

from fastapi import APIRouter, Depends
from app.dependencies import get_model_service
from app.schema import ModelInfo
from app.services.model_service import ModelService

router = APIRouter()

@router.get("/model-info", response_model=ModelInfo)
async def model_info(service: ModelService = Depends(get_model_service)) -> ModelInfo:
    return service.get_model_info()
