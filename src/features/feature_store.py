"""
FeatureStore abstraction for feature engineering and validation.
"""

import pandas as pd
from src.features.build_features import build_features
from src.features.feature_contract import required_raw_features

class FeatureStore:
    def transform_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw input. No scaling or renaming.
        Raises ValueError with actionable diagnostics if contract is violated.
        """
        diagnostics = self.validate(df, diagnostics=True)
        if diagnostics:
            msg = "Feature contract violation:\n"
            if 'missing' in diagnostics:
                msg += f"- Missing required features: {diagnostics['missing']}\n  Suggestion: Add these columns to your input.\n"
            if 'type_mismatches' in diagnostics:
                msg += f"- Type mismatches: {diagnostics['type_mismatches']}\n  Suggestion: Ensure these columns are numeric.\n"
            raise ValueError(msg)
        return build_features(df)

    def validate(self, df: pd.DataFrame, diagnostics: bool = False) -> dict:
        """
        Validate that all required raw features are present in the input DataFrame.
        If diagnostics=True, returns a dict of issues (missing, type mismatches, unexpected) instead of raising.
        Returns actionable suggestions in error messages.
        """
        required = required_raw_features()
        missing = required - set(df.columns)
        type_mismatches = [col for col in required if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        issues = {}
        if missing:
            issues['missing'] = list(missing)
        if type_mismatches:
            issues['type_mismatches'] = type_mismatches
        if diagnostics:
            return issues
        if issues:
            msg = "Feature contract violation:\n"
            if 'missing' in issues:
                msg += f"- Missing required features: {issues['missing']}\n  Suggestion: Add these columns to your input.\n"
            if 'type_mismatches' in issues:
                msg += f"- Type mismatches: {issues['type_mismatches']}\n  Suggestion: Ensure these columns are numeric.\n"
            raise ValueError(msg)
