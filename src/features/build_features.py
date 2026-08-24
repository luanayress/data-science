"""Deterministic churn feature engineering shared by training and inference."""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .feature_contract import MODEL_FEATURES, RAW_FEATURES, TARGET_COLUMN


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stateless model features from raw customer columns."""

    def fit(self, X: pd.DataFrame, y=None):
        self._validate(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        out = X.loc[:, list(RAW_FEATURES)].copy()
        out["Age_Squared"] = out["Age"] ** 2
        out["Age_Tenure_Interaction"] = out["Age"] * out["Tenure"]
        return out.loc[:, list(MODEL_FEATURES)]

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = set(RAW_FEATURES) - set(df.columns)
        if missing:
            raise ValueError("Missing raw input columns: {}".format(sorted(missing)))
        non_numeric = [
            name for name in RAW_FEATURES
            if not pd.api.types.is_numeric_dtype(df[name])
        ]
        if non_numeric:
            raise ValueError("Raw input columns must be numeric: {}".format(non_numeric))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible stateless feature builder; never fits a scaler."""
    return ChurnFeatureEngineer().fit_transform(df)


def get_features_for_modeling(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError("Target column not found: {}".format(target_col))
    missing = set(RAW_FEATURES) - set(df.columns)
    if missing:
        raise ValueError("Missing raw input columns: {}".format(sorted(missing)))
    drop_cols = drop_cols or []
    X = df.loc[:, list(RAW_FEATURES)].drop(columns=drop_cols, errors="ignore")
    return X, df[target_col]
