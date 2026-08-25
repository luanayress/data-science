"""Official leakage-safe churn training pipeline."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.load_data import load_customer_churn_data
from ..data.split import split_train_test
from ..data.validation import validate_data_shape, validate_required_columns
from ..features.build_features import ChurnFeatureEngineer, get_features_for_modeling
from ..features.feature_contract import (
    IGNORED_HTTP_FEATURES, MODEL_FEATURES, RAW_FEATURES, TARGET_COLUMN,
)
from ..models.registry import ModelRegistry
from ..utils.config import load_config, load_training_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_churn_pipeline(model_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """Build an unfitted pipeline that accepts raw churn columns."""
    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessing", StandardScaler()),
        ("model", GradientBoostingClassifier(**(model_params or {}))),
    ])


def _evaluate(pipeline: Pipeline, X_test, y_test, threshold: float) -> Dict[str, float]:
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }


def run_training_pipeline(
    config_file: Optional[str] = None,
    save_model: bool = True,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Split raw data, fit only on train, evaluate on test and persist v2."""
    config = load_config(config_file) if config_file else load_training_config()
    training = config["training"]
    model_config = config["model"]
    prediction = config["prediction"]
    selected_version = version or training["version"]
    if model_config["type"] != "GradientBoostingClassifier":
        raise ValueError("Unsupported model type: {}".format(model_config["type"]))

    df = load_customer_churn_data()
    validate_data_shape(df, min_rows=1, min_cols=1)
    validate_required_columns(df, list(RAW_FEATURES) + [TARGET_COLUMN])
    X, y = get_features_for_modeling(df)

    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=training["test_size"],
        random_state=training["random_state"],
        stratify=True,
    )
    params = dict(model_config["params"])
    params["random_state"] = training["random_state"]
    pipeline = create_churn_pipeline(params)
    pipeline.fit(X_train, y_train)

    threshold = float(prediction["threshold"])
    metrics = _evaluate(pipeline, X_test, y_test, threshold)
    metadata = {
        "name": "churn",
        "version": selected_version,
        "algorithm": type(pipeline.named_steps["model"]).__name__,
        "target": TARGET_COLUMN,
        "raw_features": list(RAW_FEATURES),
        "model_features": list(MODEL_FEATURES),
        "ignored_http_features": list(IGNORED_HTTP_FEATURES),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": int(training["random_state"]),
        "threshold": threshold,
        "high_confidence_threshold": float(prediction["high_confidence_threshold"]),
        "metrics": metrics,
    }
    if save_model:
        ModelRegistry(selected_version).save_pipeline(pipeline, metadata)
    logger.info("Training complete | version=%s metrics=%s", selected_version, metrics)
    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
