"""Unit tests for model operations."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.train import train_gradient_boosting, train_logistic_regression
from src.models.evaluate import evaluate_model, evaluate_on_train_test
from src.models.predict import make_predictions, predict_with_confidence
from src.models.registry import ModelRegistry


def create_sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
    })
    
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


def test_train_gradient_boosting():
    """Test Gradient Boosting training."""
    X, y = create_sample_data()
    
    model, metadata = train_gradient_boosting(X, y)
    
    assert model is not None
    assert metadata['model_type'] == 'GradientBoostingClassifier'
    assert metadata['train_score'] > 0
    print("✓ test_train_gradient_boosting passed")


def test_train_logistic_regression():
    """Test Logistic Regression training."""
    X, y = create_sample_data()
    
    model, metadata = train_logistic_regression(X, y)
    
    assert model is not None
    assert metadata['model_type'] == 'LogisticRegression'
    assert metadata['train_score'] > 0
    print("✓ test_train_logistic_regression passed")


def test_make_predictions():
    """Test predictions."""
    X_train, y_train = create_sample_data()
    X_test, _ = create_sample_data()
    
    model, _ = train_gradient_boosting(X_train, y_train)
    
    # Test class predictions
    preds = make_predictions(model, X_test, return_probabilities=False)
    assert len(preds) == len(X_test)
    assert all(p in [0, 1] for p in preds)
    
    # Test probability predictions
    probs = make_predictions(model, X_test, return_probabilities=True)
    assert len(probs) == len(X_test)
    assert all(0 <= p <= 1 for p in probs)
    
    print("✓ test_make_predictions passed")


def test_predict_with_confidence():
    """Test predictions with confidence."""
    X_train, y_train = create_sample_data()
    X_test, _ = create_sample_data()
    
    model, _ = train_gradient_boosting(X_train, y_train)
    
    result = predict_with_confidence(model, X_test)
    
    assert 'predictions' in result
    assert 'probabilities' in result
    assert 'confidence' in result
    assert 'high_confidence' in result
    assert len(result['predictions']) == len(X_test)
    print("✓ test_predict_with_confidence passed")


def test_evaluate_model():
    """Test model evaluation."""
    X_train, y_train = create_sample_data()
    X_test, y_test = create_sample_data()
    
    model, _ = train_gradient_boosting(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    print("✓ test_evaluate_model passed")


def test_model_registry():
    """Test model registry."""
    X_train, y_train = create_sample_data()
    model, metadata = train_gradient_boosting(X_train, y_train)
    
    # Create temporary registry
    with tempfile.TemporaryDirectory() as tmpdir:
        # We'll just test the registry structure
        registry = ModelRegistry(version="test")
        
        # Test that registry can be instantiated
        assert registry is not None
        assert registry.version == "test"
        print("✓ test_model_registry passed")


if __name__ == "__main__":
    test_train_gradient_boosting()
    test_train_logistic_regression()
    test_make_predictions()
    test_predict_with_confidence()
    test_evaluate_model()
    test_model_registry()
    print("\n✅ All model tests passed!")
