"""Model evaluation utilities."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model with predict method.
        X_test: Test features.
        y_test: Test target.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"Model evaluation:\n{format_metrics(metrics)}")
    
    return metrics


def evaluate_on_train_test(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on both train and test sets.
    
    Args:
        model: Trained model.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        
    Returns:
        Dictionary with 'train' and 'test' metrics.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_pred_train),
        'f1': f1_score(y_train, y_pred_train, zero_division=0)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test, zero_division=0)
    }
    
    return {'train': train_metrics, 'test': test_metrics}


def get_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dict: bool = False
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_dict: If True, return dict instead of string.
        
    Returns:
        Classification report (string or dict).
    """
    return classification_report(y_true, y_pred, output_dict=output_dict)


def get_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Generate confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Confusion matrix as numpy array.
    """
    return confusion_matrix(y_true, y_pred)


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary as string.
    
    Args:
        metrics: Dictionary of metrics.
        
    Returns:
        Formatted string.
    """
    lines = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    return "\n".join(lines)


def compare_models(
    models_metrics: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Compare metrics across multiple models.
    
    Args:
        models_metrics: Dict of {model_name: metrics_dict}.
        
    Returns:
        DataFrame with comparison.
    """
    df = pd.DataFrame(models_metrics).T
    return df.sort_values('f1', ascending=False)
