"""Single source of truth for the deployed churn model contract."""

from typing import Set, Tuple

TARGET_COLUMN = "Exited"
RAW_FEATURES: Tuple[str, ...] = ("Age", "Tenure", "NumOfProducts")
MODEL_FEATURES: Tuple[str, ...] = (
    "NumOfProducts",
    "Age_Squared",
    "Age_Tenure_Interaction",
)
HTTP_FEATURES: Tuple[str, ...] = (
    "SeniorCitizen", "Age", "NumOfProducts", "Tenure", "MonthlyCharges",
    "TotalCharges", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
)
IGNORED_HTTP_FEATURES: Tuple[str, ...] = tuple(
    feature for feature in HTTP_FEATURES if feature not in RAW_FEATURES
)


def required_raw_features() -> Set[str]:
    return set(RAW_FEATURES)


def engineered_features() -> Set[str]:
    return set(MODEL_FEATURES)
