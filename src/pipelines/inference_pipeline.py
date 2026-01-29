"""Inference pipeline - orchestrates end-to-end prediction workflow."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from ..features.build_features import build_features, get_features_for_modeling
from ..models.predict import make_predictions, predict_with_confidence, add_predictions_to_data
from ..models.registry import ModelRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """Pipeline for making predictions on new data."""
    
    def __init__(self, version: str = "v1"):
        """
        Initialize inference pipeline.
        
        Args:
            version: Model version to load.
        """
        self.version = version
        self.registry = ModelRegistry(version=version)
        self.model = None
        self.scaler = None
        self.metadata = None
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load model and scaler from registry."""
        logger.info(f"Loading artifacts from version {self.version}")
        try:
            self.model = self.registry.load_model("model")
            self.scaler = self.registry.load_scaler("scaler")
            self.metadata = self.registry.load_metadata()
            logger.info("Artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for inference.
        
        Args:
            df: Raw input DataFrame.
            
        Returns:
            Processed DataFrame ready for prediction.
        """
        logger.info("Preprocessing data...")
        
        # Build features
        df_features = build_features(df)
        
        # Keep original target column if present for later comparison
        target_col = 'Churn' if 'Churn' in df_features.columns else None
        
        # Get features
        try:
            X, y = get_features_for_modeling(df_features, drop_cols=['customerID'])
        except ValueError:
            # No target column, just drop ID
            X = df_features.drop(columns=[col for col in ['customerID', 'Churn'] if col in df_features.columns])
        
        # Scale features
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns
            )
            logger.info(f"Data scaled. Shape: {X_scaled.shape}")
            return X_scaled
        else:
            logger.warning("No scaler available, using raw features")
            return X
    
    def predict(
        self,
        df: pd.DataFrame,
        return_probabilities: bool = True,
        add_to_original: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions on new data.
        
        Args:
            df: Input DataFrame with raw features.
            return_probabilities: If True, return probability scores.
            add_to_original: If True, return DataFrame with predictions added.
            
        Returns:
            Predictions or DataFrame with predictions added.
        """
        logger.info(f"Making predictions on {len(df)} samples")
        
        # Preprocess
        X_processed = self.preprocess_data(df)
        
        # Predict
        if return_probabilities:
            predictions = make_predictions(self.model, X_processed, return_probabilities=True)
        else:
            predictions = make_predictions(self.model, X_processed, return_probabilities=False)
        
        if add_to_original:
            df_with_pred = add_predictions_to_data(
                df, self.model, X_processed,
                pred_col='prediction',
                prob_col='probability'
            )
            return df_with_pred
        else:
            return predictions
    
    def predict_with_confidence(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict:
        """
        Make predictions with confidence scores.
        
        Args:
            df: Input DataFrame.
            threshold: Decision threshold.
            
        Returns:
            Dictionary with predictions and confidence.
        """
        logger.info(f"Making predictions with confidence threshold {threshold}")
        
        X_processed = self.preprocess_data(df)
        predictions_dict = predict_with_confidence(self.model, X_processed, threshold)
        
        return predictions_dict
    
    def batch_predict(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Make predictions on large dataset in batches.
        
        Args:
            df: Input DataFrame.
            batch_size: Batch size for processing.
            
        Returns:
            DataFrame with predictions.
        """
        logger.info(f"Making batch predictions (batch_size={batch_size})")
        
        all_predictions = []
        n_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            
            batch_df = df.iloc[start_idx:end_idx]
            batch_pred = self.predict(batch_df, add_to_original=False)
            all_predictions.extend(batch_pred)
            
            logger.info(f"Processed batch {i+1}/{n_batches}")
        
        df_with_pred = df.copy()
        df_with_pred['prediction'] = all_predictions
        
        return df_with_pred


def run_inference(
    df: pd.DataFrame,
    version: str = "v1",
    return_probabilities: bool = True
) -> pd.DataFrame:
    """
    Convenience function to run inference pipeline.
    
    Args:
        df: Input DataFrame.
        version: Model version.
        return_probabilities: Whether to return probability scores.
        
    Returns:
        DataFrame with predictions.
    """
    pipeline = InferencePipeline(version=version)
    return pipeline.predict(df, return_probabilities=return_probabilities, add_to_original=True)
