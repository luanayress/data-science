"""Orchestration between HTTP schemas and the fitted inference pipeline."""

import logging
from typing import Iterable, Optional

import pandas as pd

from app.core.exceptions import PredictionError
from app.schema import BatchPredictionResponse, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class ChurnService:
    def __init__(self, pipeline, shadow_service=None):
        self.pipeline = pipeline
        self.shadow_service = shadow_service

    def predict_one(self, payload: PredictionRequest, request_id: Optional[str] = None) -> PredictionResponse:
        predictions = self._predict([payload], request_id=request_id)
        logger.info("Single prediction executed request_id=%s", request_id)
        return predictions[0]

    def predict_batch(self, payloads: Iterable[PredictionRequest], request_id: Optional[str] = None) -> BatchPredictionResponse:
        items = list(payloads)
        predictions = self._predict(items, request_id=request_id)
        logger.info("Batch prediction executed request_id=%s batch_size=%d", request_id, len(predictions))
        return BatchPredictionResponse(predictions=predictions, total_samples=len(predictions))

    def _predict(self, payloads, request_id=None):
        try:
            frame = pd.DataFrame([item.model_dump() for item in payloads])
            result = self.pipeline.predict_with_confidence(frame)
            if self.shadow_service is not None:
                self.shadow_service.compare(frame, result, request_id=request_id)
            return [
                PredictionResponse(
                    prediction=int(prediction), probability=float(probability),
                    confidence="high" if high_confidence else "medium",
                )
                for prediction, probability, high_confidence in zip(
                    result["predictions"], result["probabilities"], result["high_confidence"]
                )
            ]
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("Prediction failed")
            raise PredictionError(str(exc)) from exc
