"""Feature engineering pipeline."""

import pandas as pd
import numpy as np
from typing import List, Tuple
from ..utils.logger import get_logger
from .transformers import (
    CategoricalEncoder,
    NumericalScaler,
    FeatureSelector
)

logger = get_logger(__name__)


def get_feature_config() -> dict:
    """Get feature engineering configuration."""
    return {
        'categorical_features': [
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaymentMethod'
        ],
        'numerical_features': [
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
            'Tenure', 'PhoneService', 'MultipleLines'
        ],
        'binary_features': [
            'Churn', 'gender', 'Partner', 'Dependents', 'PhoneService',
            'InternetService', 'PaperlessBilling'
        ]
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from raw data.
    
    Args:
        df: Raw DataFrame.
        
    Returns:
        DataFrame with engineered features.
    """
    df_processed = df.copy()
    
    logger.info("Starting feature engineering")
    
    # Handle missing values
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
    
    # Create derived features
    if 'Tenure' in df_processed.columns and 'MonthlyCharges' in df_processed.columns:
        df_processed['LifetimeValue'] = df_processed['Tenure'] * df_processed['MonthlyCharges']
        logger.info("Created LifetimeValue feature")
    
    if 'MonthlyCharges' in df_processed.columns and 'TotalCharges' in df_processed.columns:
        df_processed['MonthlyChargeToTotal'] = df_processed['MonthlyCharges'] / (df_processed['TotalCharges'] + 1)
        logger.info("Created MonthlyChargeToTotal feature")
    
    # Tenure groups
    if 'Tenure' in df_processed.columns:
        df_processed['TenureGroup'] = pd.cut(
            df_processed['Tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12', '13-24', '25-48', '49+']
        )
        logger.info("Created TenureGroup feature")
    
    logger.info(f"Feature engineering complete. Shape: {df_processed.shape}")
    
    return df_processed


def get_features_for_modeling(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    drop_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for modeling.
    
    Args:
        df: Feature-engineered DataFrame.
        target_col: Name of target column.
        drop_cols: Columns to drop.
        
    Returns:
        Tuple of (X, y).
    """
    if drop_cols is None:
        drop_cols = ['customerID'] if 'customerID' in df.columns else []
    
    # Drop unnecessary columns
    X = df.drop(columns=[col for col in drop_cols + [target_col] if col in df.columns])
    
    # Extract target
    if target_col in df.columns:
        y = df[target_col]
        # Convert to binary if string
        if y.dtype == 'object':
            y = (y == 'Yes').astype(int)
    else:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
    
    return X, y
