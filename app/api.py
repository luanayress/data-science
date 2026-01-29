"""FastAPI backend for model inference."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.inference_pipeline import InferencePipeline
from src.utils.logger import get_logger
from app.schema import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, HealthCheck
)

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

# Add CORS middleware for Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global pipeline
    logger.info("Starting FastAPI server...")
    try:
        pipeline = InferencePipeline(version="v1")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down FastAPI server")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if pipeline else "unhealthy",
        model_loaded=pipeline is not None,
        version="v1",
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = pipeline.metadata
    return ModelInfo(
        model_type=metadata.get('model_type', 'Unknown'),
        version="v1",
        trained_at=metadata.get('saved_at', None),
        accuracy=metadata.get('train_score', None),
        f1_score=None,
        n_features=metadata.get('n_features', 0),
        feature_names=metadata.get('feature_names', [])
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Make a single prediction.
    
    Example:
    ```
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
    ```
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.model_dump()])
        
        # Make prediction with confidence
        pred_dict = pipeline.predict_with_confidence(df, threshold=0.5)
        
        prediction = int(pred_dict['predictions'][0])
        probability = float(pred_dict['probabilities'][0])
        confidence = 'high' if pred_dict['high_confidence'][0] else 'medium'
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Accepts a list of prediction requests and returns a list of predictions.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request list to DataFrame
        df = pd.DataFrame([item.model_dump() for item in request.data])
        
        # Make predictions
        predictions_list = []
        for idx in range(len(df)):
            df_row = df.iloc[[idx]]
            pred_dict = pipeline.predict_with_confidence(df_row, threshold=0.5)
            
            prediction = int(pred_dict['predictions'][0])
            probability = float(pred_dict['probabilities'][0])
            confidence = 'high' if pred_dict['high_confidence'][0] else 'medium'
            
            predictions_list.append(
                PredictionResponse(
                    prediction=prediction,
                    probability=probability,
                    confidence=confidence
                )
            )
        
        logger.info(f"Batch prediction complete: {len(predictions_list)} samples")
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            total_samples=len(predictions_list)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API documentation link."""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "endpoints": {
            "POST /predict": "Make a single prediction",
            "POST /predict-batch": "Make batch predictions"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
