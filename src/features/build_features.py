"""
Feature engineering shared between training and inference.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = {"Age", "Tenure", "NumOfProducts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing raw input columns: {missing}")

    df["Age_Squared"] = df["Age"] ** 2
    df["Age_Tenure_Interaction"] = df["Age"] * df["Tenure"]

    age_sq_scaler = StandardScaler()
    interaction_scaler = MinMaxScaler()

    df["Age_Squared_StandardScaled"] = age_sq_scaler.fit_transform(
        df[["Age_Squared"]]
    )

    df["Age_Tenure_Interaction_MinMaxScaled"] = interaction_scaler.fit_transform(
        df[["Age_Tenure_Interaction"]]
    )

    return df[
        [
            "NumOfProducts",
            "Age_Squared_StandardScaled",
            "Age_Tenure_Interaction_MinMaxScaled",
        ]
    ]


def get_features_for_modeling(
    df: pd.DataFrame,
    target_col: str = "Churn",
    drop_cols: Optional[List[str]] = None,
):
    drop_cols = drop_cols or []

    if target_col not in df.columns:
        raise ValueError("Target column not found")

    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")

    return X, y
