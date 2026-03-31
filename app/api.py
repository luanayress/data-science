"""
FastAPI backend for model inference and monitoring (MLOps-ready).
"""

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
import pandas as pd
import tempfile
import os
import sys
import uuid
from typing import Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.inference_pipeline import InferencePipeline
from src.monitoring.monitor import ModelMonitor
from src.utils.logger import get_logger
from app.schema import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthCheck
)

logger = get_logger(__name__)


def _resolve_allowed_origins() -> list[str]:
    """Resolve CORS origins from env, defaulting to local development URLs."""
    origins_raw = os.getenv("ALLOWED_ORIGINS", "")
    if origins_raw.strip():
        return [origin.strip() for origin in origins_raw.split(",") if origin.strip()]

    # Safe defaults for local development. Override in production.
    return [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready API for customer churn prediction",
    version="1.0.0"
)

# CORS (Streamlit / frontend friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Load model artifacts and monitoring tools at startup.
    """
    logger.info("Starting FastAPI server...")
    try:
        app.state.pipeline = InferencePipeline(version="v1")
        app.state.monitor = ModelMonitor()
        logger.info("Model and monitor loaded successfully")
    except Exception as e:
        logger.exception("Failed to initialize application")
        raise RuntimeError("Startup failed") from e


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup resources on shutdown.
    """
    logger.info("Shutting down FastAPI server")


# ---------------------------------------------------------------------
# Middleware (request tracing)
# ---------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Attach a request_id to each request for traceability.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------
@app.get("/health", response_model=HealthCheck)
async def health_check(request: Request):
    """
    Health check endpoint.
    """
    pipeline_loaded = hasattr(request.app.state, "pipeline")

    return HealthCheck(
        status="healthy" if pipeline_loaded else "unhealthy",
        model_loaded=pipeline_loaded,
        version="v1",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model-info", response_model=ModelInfo)
async def model_info(request: Request):
    """
    Return metadata about the loaded model.
    """
    pipeline: Optional[InferencePipeline] = getattr(
        request.app.state, "pipeline", None
    )

    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = pipeline.metadata

    return ModelInfo(
        model_type=metadata.get("model_type", "Unknown"),
        version="v1",
        trained_at=metadata.get("saved_at"),
        accuracy=metadata.get("train_score"),
        f1_score=metadata.get("f1_score"),
        n_features=metadata.get("n_features", 0),
        feature_names=metadata.get("feature_names", [])
    )


# ---------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: Request, payload: PredictionRequest):
    """
    Single prediction endpoint.
    """
    pipeline: Optional[InferencePipeline] = getattr(
        request.app.state, "pipeline", None
    )

    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([payload.model_dump()])

        pred = pipeline.predict_with_confidence(df, threshold=0.5)

        response = PredictionResponse(
            prediction=int(pred["predictions"][0]),
            probability=float(pred["probabilities"][0]),
            confidence="high" if pred["high_confidence"][0] else "medium"
        )

        logger.info(
            "Single prediction executed",
            extra={
                "request_id": request.state.request_id,
                "probability": response.probability
            }
        )

        return response

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: Request, payload: BatchPredictionRequest):
    """
    Batch prediction endpoint (vectorized).
    """
    pipeline: Optional[InferencePipeline] = getattr(
        request.app.state, "pipeline", None
    )

    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([item.model_dump() for item in payload.data])

        pred = pipeline.predict_with_confidence(df, threshold=0.5)

        predictions = [
            PredictionResponse(
                prediction=int(p),
                probability=float(prob),
                confidence="high" if hc else "medium"
            )
            for p, prob, hc in zip(
                pred["predictions"],
                pred["probabilities"],
                pred["high_confidence"]
            )
        ]

        logger.info(
            "Batch prediction executed",
            extra={
                "request_id": request.state.request_id,
                "batch_size": len(predictions)
            }
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions)
        )

    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------
# Monitoring / Drift
# ---------------------------------------------------------------------
@app.post("/monitor/report")
async def monitor_report(
    request: Request,
    reference: UploadFile = File(...),
    current: UploadFile = File(...),
    alpha: float = Form(0.05)
):
    """
    Upload reference and current CSVs and return drift report.
    """
    monitor: Optional[ModelMonitor] = getattr(
        request.app.state, "monitor", None
    )

    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not available")

    ref_path, cur_path = None, None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as ref_tmp:
            ref_tmp.write(await reference.read())
            ref_path = ref_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as cur_tmp:
            cur_tmp.write(await current.read())
            cur_path = cur_tmp.name

        ref_df = pd.read_csv(ref_path)
        cur_df = pd.read_csv(cur_path)

        report = monitor.detect_drift(ref_df, cur_df, alpha=alpha)

        logger.info(
            "Drift report generated",
            extra={
                "request_id": request.state.request_id,
                "alpha": alpha,
                "drifted_features": report.get("drifted_features", [])
            }
        )

        return JSONResponse(content=report)

    except Exception as e:
        logger.exception("Drift detection failed")
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        if cur_path and os.path.exists(cur_path):
            os.remove(cur_path)


# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "endpoints": [
            "POST /predict",
            "POST /predict-batch",
            "POST /monitor/report"
        ]
    }


# ---------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
