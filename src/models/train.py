"""Model training utilities."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
    """
    Train a Gradient Boosting Classifier.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        **kwargs: Additional parameters for GradientBoostingClassifier.
        
    Returns:
        Tuple of (model, metadata).
    """
    params = {
        'n_estimators': kwargs.get('n_estimators', 100),
        'learning_rate': kwargs.get('learning_rate', 0.1),
        'max_depth': kwargs.get('max_depth', 5),
        'min_samples_split': kwargs.get('min_samples_split', 5),
        'min_samples_leaf': kwargs.get('min_samples_leaf', 2),
        'random_state': kwargs.get('random_state', 42)
    }
    
    logger.info(f"Training Gradient Boosting with params: {params}")
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_score = model.score(X_train, y_train)
    
    metadata = {
        'model_type': 'GradientBoostingClassifier',
        'train_score': float(train_score),
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns),
        'params': params
    }
    
    logger.info(f"Model trained. Training score: {train_score:.4f}")
    
    return model, metadata


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Train a Logistic Regression model.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        **kwargs: Additional parameters.
        
    Returns:
        Tuple of (model, metadata).
    """
    params = {
        'max_iter': kwargs.get('max_iter', 1000),
        'random_state': kwargs.get('random_state', 42),
        'solver': kwargs.get('solver', 'lbfgs')
    }
    
    logger.info(f"Training Logistic Regression with params: {params}")
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    
    metadata = {
        'model_type': 'LogisticRegression',
        'train_score': float(train_score),
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns),
        'params': params
    }
    
    logger.info(f"Model trained. Training score: {train_score:.4f}")
    
    return model, metadata


def create_preprocessing_pipeline(X_train: pd.DataFrame) -> StandardScaler:
    """
    Create and fit a preprocessing scaler.
    
    Args:
        X_train: Training features.
        
    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info("Preprocessing pipeline created and fitted")
    return scaler
