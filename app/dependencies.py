"""FastAPI dependencies backed by application lifecycle state."""

from fastapi import Request

from app.core.exceptions import ModelUnavailableError, MonitoringError
from app.services.churn_service import ChurnService
from app.services.model_service import ModelService
from app.services.monitoring_service import MonitoringService
from app.services.shadow_service import ShadowPredictionService
from src.pipelines.inference_pipeline import InferencePipeline


def get_churn_service(request: Request) -> ChurnService:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise ModelUnavailableError("Model not loaded")
    shadow_pipeline = getattr(request.app.state, "shadow_pipeline", None)
    shadow_service = ShadowPredictionService(pipeline, shadow_pipeline) if shadow_pipeline is not None else None
    return ChurnService(pipeline, shadow_service)


def get_v4_churn_service(request: Request) -> ChurnService:
    pipeline = getattr(request.app.state, "v4_pipeline", None)
    if pipeline is None:
        try:
            pipeline = InferencePipeline(version="v4")
            request.app.state.v4_pipeline = pipeline
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise ModelUnavailableError("V4 model not loaded") from exc
    return ChurnService(pipeline)


def get_model_service(request: Request) -> ModelService:
    return ModelService(getattr(request.app.state, "pipeline", None), request.app.state.settings.model_version)


def get_monitoring_service(request: Request) -> MonitoringService:
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None:
        raise MonitoringError("Monitor not available")
    return MonitoringService(monitor)
