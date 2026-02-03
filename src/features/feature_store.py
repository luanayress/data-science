"""
FeatureStore abstraction for feature engineering and validation.
"""

import pandas as pd
from src.features.feature_contract import required_raw_features, engineered_features

class FeatureStore:
    def transform_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw input. No scaling or renaming.
        Raises ValueError with actionable diagnostics if contract is violated.
        """
        out = df.copy()
        diagnostics = self.validate(df, diagnostics=True)
        if diagnostics:
            msg = "Feature contract violation:\n"
            if 'missing' in diagnostics:
                msg += f"- Missing required features: {diagnostics['missing']}\n  Suggestion: Add these columns to your input.\n"
            if 'type_mismatches' in diagnostics:
                msg += f"- Type mismatches: {diagnostics['type_mismatches']}\n  Suggestion: Ensure these columns are numeric.\n"
            if 'unexpected' in diagnostics:
                msg += f"- Unexpected features: {diagnostics['unexpected']}\n  Suggestion: Remove or ignore these columns.\n"
            raise ValueError(msg)
        out["Age_Squared"] = out["Age"] ** 2
        out["Age_Tenure_Interaction"] = out["Age"] * out["Tenure"]
        return out[["NumOfProducts", "Age_Squared", "Age_Tenure_Interaction"]]

    def validate(self, df: pd.DataFrame, diagnostics: bool = False) -> dict:
        """
        Validate that all required raw features are present in the input DataFrame.
        If diagnostics=True, returns a dict of issues (missing, type mismatches, unexpected) instead of raising.
        Returns actionable suggestions in error messages.
        """
        required = required_raw_features()
        missing = required - set(df.columns)
        type_mismatches = [col for col in required if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        extra = set(df.columns) - required
        issues = {}
        if missing:
            issues['missing'] = list(missing)
        if type_mismatches:
            issues['type_mismatches'] = type_mismatches
        if extra:
            issues['unexpected'] = list(extra)
        if diagnostics:
            return issues
        if issues:
            msg = "Feature contract violation:\n"
            if 'missing' in issues:
                msg += f"- Missing required features: {issues['missing']}\n  Suggestion: Add these columns to your input.\n"
            if 'type_mismatches' in issues:
                msg += f"- Type mismatches: {issues['type_mismatches']}\n  Suggestion: Ensure these columns are numeric.\n"
            if 'unexpected' in issues:
                msg += f"- Unexpected features: {issues['unexpected']}\n  Suggestion: Remove or ignore these columns.\n"
            raise ValueError(msg)
