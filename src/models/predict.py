"""Model prediction utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


def make_predictions(
    model,
    X: pd.DataFrame,
    return_probabilities: bool = False
) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        model: Trained model.
        X: Features for prediction.
        return_probabilities: If True, return probability predictions.
        
    Returns:
        Predictions (classes or probabilities).
    """
    if return_probabilities:
        predictions = model.predict_proba(X)[:, 1]
        logger.info(f"Made {len(predictions)} probability predictions")
    else:
        predictions = model.predict(X)
        logger.info(f"Made {len(predictions)} class predictions")
    
    return predictions


def batch_predict(
    model,
    X: pd.DataFrame,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Make predictions in batches (useful for large datasets).
    
    Args:
        model: Trained model.
        X: Features for prediction.
        batch_size: Number of samples per batch.
        
    Returns:
        Predictions.
    """
    predictions = []
    n_batches = (len(X) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        
        batch_pred = model.predict(X.iloc[start_idx:end_idx])
        predictions.extend(batch_pred)
    
    logger.info(f"Made {len(predictions)} predictions in {n_batches} batches")
    
    return np.array(predictions)


def predict_with_confidence(
    model,
    X: pd.DataFrame,
    threshold: float = 0.5
) -> Dict:
    """
    Make predictions with confidence scores.
    
    Args:
        model: Trained model.
        X: Features for prediction.
        threshold: Decision threshold for binary classification.
        
    Returns:
        Dictionary with predictions and confidence.
    """
    probabilities = model.predict_proba(X)
    predictions = (probabilities[:, 1] >= threshold).astype(int)
    confidence = np.max(probabilities, axis=1)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities[:, 1],
        'confidence': confidence,
        'high_confidence': confidence > 0.8
    }


def add_predictions_to_data(
    df: pd.DataFrame,
    model,
    X: pd.DataFrame,
    pred_col: str = 'prediction',
    prob_col: str = 'probability'
) -> pd.DataFrame:
    """
    Add predictions to original dataframe.
    
    Args:
        df: Original DataFrame.
        model: Trained model.
        X: Features used for prediction.
        pred_col: Name of column for predictions.
        prob_col: Name of column for probabilities.
        
    Returns:
        DataFrame with predictions added.
    """
    df_with_pred = df.copy()
    df_with_pred[pred_col] = model.predict(X)
    df_with_pred[prob_col] = model.predict_proba(X)[:, 1]
    
    logger.info(f"Added predictions to DataFrame: {pred_col}, {prob_col}")
    
    return df_with_pred
