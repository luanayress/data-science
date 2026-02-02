"""Inference pipeline - orchestrates end-to-end prediction workflow."""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.base import TransformerMixin
from ..features.build_features import build_features
from ..models.registry import ModelRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """Pipeline for making predictions on new data."""

    def __init__(self, version: str = "v1"):
        self.version = version
        self.registry = ModelRegistry(version=version)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model, scaler and metadata."""
        logger.info(f"Loading artifacts from version {self.version}")

        self.model = self.registry.load_model("model")
        self.scaler = self.registry.load_scaler("scaler")
        self.metadata = self.registry.load_metadata("model")

        self.feature_names = self.metadata["feature_names"]

        logger.info(
            f"Artifacts loaded | n_features={len(self.feature_names)} | "
            f"scaler_type={type(self.scaler)}"
        )

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features and prepare final matrix for inference.
        """
        logger.info("Preprocessing data for inference")

        # 1. Feature engineering (same as training)
        features_df = build_features(df)

        # 2. Ensure schema consistency
        missing = set(self.feature_names) - set(features_df.columns)
        if missing:
            raise ValueError(
                f"Missing required features for inference: {missing}"
            )

        X = features_df[self.feature_names]

        # 3. Apply scaler ONLY if it is a real transformer
        if self.scaler is not None:
            if isinstance(self.scaler, TransformerMixin):
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=self.feature_names,
                    index=X.index
                )
                logger.info("Scaler applied successfully")
            else:
                logger.warning(
                    "Scaler is not a transformer. Skipping scaling. "
                    f"Type={type(self.scaler)}"
                )

        return X

    def predict_with_confidence(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict:
        """
        Make predictions with confidence scores.
        """
        logger.info(f"Running inference on {len(df)} samples")

        X = self.preprocess_data(df)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        high_confidence = (probs >= 0.8) | (probs <= 0.2)

        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "high_confidence": high_confidence.tolist()
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict and return dataframe with predictions attached.
        """
        result = self.predict_with_confidence(df)

        out = df.copy()
        out["prediction"] = result["predictions"]
        out["probability"] = result["probabilities"]
        out["high_confidence"] = result["high_confidence"]

        return out
