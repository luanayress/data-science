"""Health and model metadata application service."""

from datetime import datetime

from app.core.exceptions import ModelUnavailableError
from app.schema import HealthCheck, ModelInfo


class ModelService:
    def __init__(self, pipeline, configured_version: str):
        self.pipeline = pipeline
        self.configured_version = configured_version

    def get_health_status(self) -> HealthCheck:
        loaded = self.pipeline is not None
        return HealthCheck(
            status="healthy" if loaded else "unhealthy", model_loaded=loaded,
            version=self.pipeline.version if loaded else self.configured_version,
            timestamp=datetime.utcnow().isoformat(),
        )

    def get_model_info(self) -> ModelInfo:
        if self.pipeline is None:
            raise ModelUnavailableError("Model not loaded")
        metadata = self.pipeline.metadata
        metrics = metadata.get("metrics", {})
        features = metadata.get("model_features", metadata.get("feature_names", []))
        return ModelInfo(
            model_type=metadata.get("algorithm", metadata.get("model_type", "Unknown")),
            version=metadata.get("version", self.pipeline.version),
            trained_at=metadata.get("trained_at", metadata.get("saved_at")),
            accuracy=metrics.get("accuracy", metadata.get("train_score")),
            f1_score=metrics.get("f1", metadata.get("f1_score")),
            n_features=len(features), feature_names=features,
        )
