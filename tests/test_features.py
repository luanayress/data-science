"""Unit tests for feature engineering."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.build_features import build_features, get_features_for_modeling
from src.features.transformers import (
    CategoricalEncoder, NumericalScaler, FeatureSelector
)


def test_build_features():
    """Test feature building."""
    # Create sample data
    df = pd.DataFrame({
        'Age': [30, 40, 50, 60],
        'NumOfProducts': [1, 2, 3, 4],
        'Tenure': [10, 20, 30, 40],
    })
    
    result = build_features(df)
    
    # Check if new features are created
    assert 'NumOfProducts' in result.columns
    assert 'Age_Squared_StandardScaled' in result.columns
    assert 'Age_Tenure_Interaction_MinMaxScaled' in result.columns
    assert result.shape[0] == df.shape[0]
    print("✓ test_build_features passed")


def test_categorical_encoder():
    """Test categorical encoder."""
    df = pd.DataFrame({
        'color': ['red', 'blue', 'red', 'green'],
        'size': ['small', 'large', 'large', 'small']
    })
    
    encoder = CategoricalEncoder(['color', 'size'])
    encoder.fit(df)
    result = encoder.transform(df)
    
    # Check that values are encoded
    assert pd.api.types.is_numeric_dtype(result['color'])
    assert pd.api.types.is_numeric_dtype(result['size'])
    print("✓ test_categorical_encoder passed")


def test_numerical_scaler():
    """Test numerical scaler."""
    df = pd.DataFrame({
        'value1': [10, 20, 30, 40],
        'value2': [100, 200, 300, 400]
    })
    
    scaler = NumericalScaler(['value1', 'value2'])
    scaler.fit(df)
    result = scaler.transform(df)
    
    # Check that scaling happened
    assert result['value1'].mean() < 0.1  # Should be close to 0
    print("✓ test_numerical_scaler passed")


def test_feature_selector():
    """Test feature selector."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    
    selector = FeatureSelector(['A', 'C'])
    selector.fit(df)
    result = selector.transform(df)
    
    assert list(result.columns) == ['A', 'C']
    print("✓ test_feature_selector passed")


def test_get_features_for_modeling():
    """Test feature preparation for modeling."""
    df = pd.DataFrame({
        'customerID': [1, 2, 3],
        'feature1': [10, 20, 30],
        'feature2': [40, 50, 60],
        'Churn': ['No', 'Yes', 'No']
    })
    
    X, y = get_features_for_modeling(df, drop_cols=['customerID'])
    
    assert 'customerID' not in X.columns
    assert 'Churn' not in X.columns
    assert len(y) == 3
    print("✓ test_get_features_for_modeling passed")


if __name__ == "__main__":
    test_build_features()
    test_categorical_encoder()
    test_numerical_scaler()
    test_feature_selector()
    test_get_features_for_modeling()
    print("\n✅ All feature tests passed!")
