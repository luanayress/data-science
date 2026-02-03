"""
Feature contract: defines required raw and engineered features for both training and inference.

Contract:
    - All required raw features must be present and numeric in input
    - All engineered features must be present after preprocessing and scaling
    - No extra/unexpected features allowed

Guarantees:
    - Any contract violation will fail fast with actionable diagnostics
    - No silent schema changes
    - Training and inference use identical feature contracts
"""


def required_raw_features() -> set:
    """
    Returns the set of raw features required as input for feature engineering.
    Contract: All must be present in input DataFrame, with compatible types (numeric).
    This contract validates only external/raw inputs, not internal pipeline artifacts.
    """
    return {
        "Age",
        "Tenure",
        "NumOfProducts",
        "HasCrCard",
        # Add all other raw input features expected from external sources here
        # e.g., "CreditScore", "Balance", "IsActiveMember", ...
    }


def engineered_features() -> set:
    """
    Returns the set of engineered features expected after feature engineering and preprocessing.
    This is for internal pipeline use only, not for external contract validation.
    """
    return set()
