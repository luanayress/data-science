"""Training pipeline - orchestrates end-to-end training workflow."""

import pandas as pd
from typing import Dict, Tuple, Any
from ..data.load_data import load_customer_churn_data
from ..data.validation import validate_data_shape, get_data_summary
from ..data.split import split_train_test
from ..features.build_features import build_features, get_features_for_modeling
from ..models.train import train_gradient_boosting, create_preprocessing_pipeline
from ..models.evaluate import evaluate_model, evaluate_on_train_test
from ..models.registry import ModelRegistry
from ..utils.logger import get_logger
from ..utils.config import load_training_config

logger = get_logger(__name__)


def run_training_pipeline(
    config_file: str = None,
    save_model: bool = True,
    version: str = "v1"
) -> Dict[str, Any]:
    """
    Run complete training pipeline.
    
    Args:
        config_file: Path to config file (optional).
        save_model: Whether to save trained model.
        version: Model version directory.
        
    Returns:
        Dictionary with results including model, metrics, and metadata.
    """
    logger.info("=" * 50)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 50)
    
    # Load configuration
    try:
        config = load_training_config()
        logger.info(f"Configuration loaded: {config}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        config = {}
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    try:
        df_raw = load_customer_churn_data()
        validate_data_shape(df_raw, min_rows=1, min_cols=1)
        logger.info(f"Data shape: {df_raw.shape}")
        logger.info(f"Data summary:\n{get_data_summary(df_raw)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Step 2: Feature engineering
    logger.info("\nStep 2: Building features...")
    try:
        df_features = build_features(df_raw)
        logger.info(f"Features shape: {df_features.shape}")
    except Exception as e:
        logger.error(f"Failed to build features: {e}")
        raise
    
    # Step 3: Prepare X and y
    logger.info("\nStep 3: Preparing features and target...")
    try:
        X, y = get_features_for_modeling(df_features)
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Class distribution:\n{y.value_counts()}")
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise
    
    # Step 4: Train-test split
    logger.info("\nStep 4: Splitting data...")
    try:
        test_size = config.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=test_size, stratify=True
        )
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise
    
    # Step 5: Preprocess
    logger.info("\nStep 5: Creating preprocessing pipeline...")
    try:
        scaler = create_preprocessing_pipeline(X_train)
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        logger.info("Preprocessing complete")
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}. Continuing without scaling.")
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
    
    # Step 6: Train model
    logger.info("\nStep 6: Training model...")
    try:
        model_params = config.get('model_params', {})
        model, model_metadata = train_gradient_boosting(X_train_scaled, y_train, **model_params)
        logger.info(f"Model trained successfully")
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise
    
    # Step 7: Evaluate
    logger.info("\nStep 7: Evaluating model...")
    try:
        test_metrics = evaluate_model(model, X_test_scaled, y_test)
        train_test_metrics = evaluate_on_train_test(model, X_train_scaled, X_test_scaled, y_train, y_test)
        logger.info(f"Evaluation complete")
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        test_metrics = {}
        train_test_metrics = {}
    
    # Step 8: Save model
    results = {
        'model': model,
        'scaler': scaler,
        'metadata': model_metadata,
        'test_metrics': test_metrics,
        'train_test_metrics': train_test_metrics,
        'data_info': {
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns)
        }
    }
    
    if save_model:
        logger.info("\nStep 8: Saving model and artifacts...")
        try:
            registry = ModelRegistry(version=version)
            registry.save_model(model, "model", model_metadata)
            if scaler:
                registry.save_scaler(scaler, "scaler")
            logger.info("Model and scaler saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 50)
    
    return results
