"""Churn prediction routes."""

from fastapi import APIRouter, Depends, Request
from app.dependencies import get_churn_service, get_v4_churn_service
from app.schema import (
    BankBatchPredictionRequest, BankPredictionRequest, BatchPredictionRequest,
    BatchPredictionResponse, PredictionRequest, PredictionResponse,
)
from app.services.churn_service import ChurnService

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(payload: PredictionRequest, request: Request, service: ChurnService = Depends(get_churn_service)) -> PredictionResponse:
    return service.predict_one(payload, request_id=request.state.request_id)

@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(payload: BatchPredictionRequest, request: Request, service: ChurnService = Depends(get_churn_service)) -> BatchPredictionResponse:
    return service.predict_batch(payload.data, request_id=request.state.request_id)


@router.post("/v4/predict", response_model=PredictionResponse)
async def predict_v4(payload: BankPredictionRequest, request: Request, service: ChurnService = Depends(get_v4_churn_service)) -> PredictionResponse:
    return service.predict_one(payload, request_id=request.state.request_id)


@router.post("/v4/predict-batch", response_model=BatchPredictionResponse)
async def predict_v4_batch(payload: BankBatchPredictionRequest, request: Request, service: ChurnService = Depends(get_v4_churn_service)) -> BatchPredictionResponse:
    return service.predict_batch(payload.data, request_id=request.state.request_id)
