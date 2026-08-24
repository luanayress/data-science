"""Leakage-safe feature contract for the expanded bank churn challenger."""

from typing import Dict, Tuple

TARGET_COLUMN = "Exited"

BASELINE_FEATURES: Tuple[str, ...] = ("Age", "Tenure", "NumOfProducts")
FINANCIAL_FEATURES: Tuple[str, ...] = (
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
)
DEMOGRAPHIC_FEATURES: Tuple[str, ...] = FINANCIAL_FEATURES + ("Geography", "Gender")
EXPANDED_FEATURES: Tuple[str, ...] = DEMOGRAPHIC_FEATURES + (
    "SatisfactionScore", "CardType", "PointEarned",
)

NUMERIC_FEATURES: Tuple[str, ...] = tuple(
    name for name in EXPANDED_FEATURES if name not in ("Geography", "Gender", "CardType")
)
CATEGORICAL_FEATURES: Tuple[str, ...] = ("Geography", "Gender", "CardType")

FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "baseline_raw": BASELINE_FEATURES,
    "financial_activity": FINANCIAL_FEATURES,
    "plus_demographics": DEMOGRAPHIC_FEATURES,
    "expanded_bank": EXPANDED_FEATURES,
}

SOURCE_RENAMES = {
    "Satisfaction Score": "SatisfactionScore",
    "Card Type": "CardType",
    "Point Earned": "PointEarned",
}

EXCLUDED_FEATURES: Tuple[str, ...] = (
    "RowNumber", "CustomerId", "Surname", "Exited", "Complain",
    "Complain_With_Low_Satisfaction",
)
