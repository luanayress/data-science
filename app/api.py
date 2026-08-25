"""Compatible FastAPI entrypoint and application bootstrap."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import churn_router, health_router, model_info_router, monitoring_router
from app.core.config import get_settings
from app.core.exceptions import ApplicationError, ModelUnavailableError
from app.core.logging import configure_logging
from src.monitoring.monitor import ModelMonitor
from src.pipelines.inference_pipeline import InferencePipeline

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting API environment=%s", settings.environment)
    application.state.settings = settings
    try:
        application.state.pipeline = InferencePipeline(
            version=settings.model_version,
            threshold_profile=settings.churn_threshold_profile,
        )
        application.state.shadow_pipeline = None
        if settings.shadow_model_version:
            application.state.shadow_pipeline = InferencePipeline(
                version=settings.shadow_model_version,
                threshold_profile=settings.churn_threshold_profile,
            )
        application.state.monitor = ModelMonitor()
        logger.info("Model loaded version=%s", settings.model_version)
        if application.state.shadow_pipeline is not None:
            logger.info("Shadow model loaded version=%s", settings.shadow_model_version)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.exception("Model load failure version=%s", settings.model_version)
        raise RuntimeError("Startup failed") from exc
    yield
    logger.info("Shutting down API")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Customer Churn Prediction API",
        description="Production-ready API for customer churn prediction",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.exception_handler(ApplicationError)
    async def application_error_handler(request: Request, exc: ApplicationError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    application.include_router(health_router)
    application.include_router(model_info_router)
    application.include_router(churn_router)
    application.include_router(monitoring_router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app", host=settings.api_host, port=settings.api_port,
        reload=True, log_level=settings.log_level.lower(),
    )
