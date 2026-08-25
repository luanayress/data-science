"""Inference over the persisted, fully fitted churn pipeline."""

import os
from typing import Dict, Optional

import pandas as pd

from ..features.feature_contract import MODEL_FEATURES
from ..models.registry import ModelRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


class InferencePipeline:
    """Load one versioned pipeline and use it unchanged for all predictions."""

    def __init__(self, version: str = "v2", threshold_profile: Optional[str] = None):
        self.version = version
        self.registry = ModelRegistry(version=version)
        self.pipeline = self.registry.load_pipeline()
        self.metadata = self.registry.load_metadata()
        if not self.metadata:
            raise ValueError("Missing pipeline metadata for version {}".format(version))
        self.feature_names = self.metadata.get("model_features", list(MODEL_FEATURES))
        self.required_input_features = self.metadata.get("raw_features", [])
        self.strict_input_contract = bool(self.metadata.get("strict_input_contract", False))
        self.threshold_profile = threshold_profile
        self.threshold = self._resolve_threshold(threshold_profile)
        if hasattr(self.pipeline, "named_steps"):
            self.model = self.pipeline.named_steps["model"]
            self.scaler = self.pipeline.named_steps["preprocessing"]
        else:
            # Calibrated challengers wrap the complete raw-input pipeline.
            self.model = self.pipeline
            self.scaler = None

    def _resolve_threshold(self, profile: Optional[str]) -> float:
        if profile is None:
            return float(self.metadata.get("selected_threshold", self.metadata["threshold"]))
        thresholds = self.metadata.get("thresholds", {})
        if not thresholds:
            return float(self.metadata["threshold"])
        if profile == "default":
            return float(thresholds.get("default", self.metadata["threshold"]))
        if profile not in thresholds:
            raise ValueError("Threshold profile {!r} is unavailable for model {}".format(profile, self.version))
        return float(thresholds[profile])

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data with already-fitted pipeline steps."""
        if not hasattr(self.pipeline, "named_steps"):
            raise NotImplementedError(
                "Preprocessing inspection is unavailable for wrapped calibrated pipelines"
            )
        transformed = self.pipeline[:-1].transform(df)
        return pd.DataFrame(transformed, columns=self.feature_names, index=df.index)

    def predict_with_confidence(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Dict:
        """Predict without fitting or mutating any transformer."""
        self.validate_input(df)
        decision_threshold = float(
            self.threshold if threshold is None else threshold
        )
        high_threshold = float(self.metadata.get("high_confidence_threshold", 0.8))
        probabilities = self.pipeline.predict_proba(df)[:, 1]
        predictions = (probabilities >= decision_threshold).astype(int)
        high_confidence = (
            (probabilities >= high_threshold)
            | (probabilities <= 1.0 - high_threshold)
        )
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "high_confidence": high_confidence.tolist(),
        }

    def validate_input(self, df: pd.DataFrame) -> None:
        """Reject incomplete payloads for models that declare a strict contract."""
        if not self.strict_input_contract:
            return
        missing = [name for name in self.required_input_features if name not in df.columns]
        empty = [
            name for name in self.required_input_features
            if name in df.columns and bool(df[name].isna().any())
        ]
        if missing or empty:
            raise ValueError(
                "Incomplete {} input; missing columns={}, null columns={}".format(
                    self.version, missing, empty,
                )
            )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.predict_with_confidence(df)
        out = df.copy()
        out["prediction"] = result["predictions"]
        out["probability"] = result["probabilities"]
        out["high_confidence"] = result["high_confidence"]
        return out
