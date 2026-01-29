"""Feature transformation classes."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical variables."""
    
    def __init__(self, categorical_columns: List[str] = None):
        """
        Initialize encoder.
        
        Args:
            categorical_columns: List of categorical column names.
        """
        self.categorical_columns = categorical_columns or []
        self.encoders = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders."""
        for col in self.categorical_columns:
            if col in X.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(X[col].astype(str))
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns."""
        X_transformed = X.copy()
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = encoder.transform(X_transformed[col].astype(str))
        return X_transformed


class NumericalScaler(BaseEstimator, TransformerMixin):
    """Scale numerical features."""
    
    def __init__(self, numerical_columns: List[str] = None):
        """
        Initialize scaler.
        
        Args:
            numerical_columns: List of numerical column names.
        """
        self.numerical_columns = numerical_columns or []
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler."""
        if self.numerical_columns and any(col in X.columns for col in self.numerical_columns):
            cols_to_scale = [col for col in self.numerical_columns if col in X.columns]
            self.scaler.fit(X[cols_to_scale])
            self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical columns."""
        X_transformed = X.copy()
        if self.is_fitted:
            cols_to_scale = [col for col in self.numerical_columns if col in X_transformed.columns]
            X_transformed[cols_to_scale] = self.scaler.transform(X_transformed[cols_to_scale])
        return X_transformed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features."""
    
    def __init__(self, features: List[str] = None):
        """
        Initialize selector.
        
        Args:
            features: List of feature names to keep.
        """
        self.features = features or []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit selector."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features."""
        if not self.features:
            return X
        available_features = [f for f in self.features if f in X.columns]
        return X[available_features]


class CombinedTransformer(BaseEstimator, TransformerMixin):
    """Combine multiple transformations."""
    
    def __init__(
        self,
        categorical_cols: List[str] = None,
        numerical_cols: List[str] = None
    ):
        """Initialize combined transformer."""
        self.categorical_transformer = CategoricalEncoder(categorical_cols)
        self.numerical_transformer = NumericalScaler(numerical_cols)
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit all transformers."""
        self.categorical_transformer.fit(X, y)
        self.numerical_transformer.fit(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations."""
        X_transformed = X.copy()
        X_transformed = self.categorical_transformer.transform(X_transformed)
        X_transformed = self.numerical_transformer.transform(X_transformed)
        return X_transformed
