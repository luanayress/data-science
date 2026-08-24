"""Cross-validation and holdout evaluation utilities."""

import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
    make_scorer,
)
from sklearn.model_selection import cross_validate


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": "accuracy",
}


def cross_validation_summary(estimator, X, y, cv, n_jobs: int = 1) -> Tuple[Dict[str, float], pd.DataFrame]:
    scores = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs, return_train_score=False)
    summary = {}
    folds = []
    for metric in SCORING:
        values = scores["test_" + metric]
        summary.update({
            "cv_{}_mean".format(metric): float(np.mean(values)),
            "cv_{}_std".format(metric): float(np.std(values)),
            "cv_{}_min".format(metric): float(np.min(values)),
            "cv_{}_max".format(metric): float(np.max(values)),
        })
        for fold, value in enumerate(values, 1):
            folds.append({"fold": fold, "metric": metric, "value": float(value)})
    summary["fit_time"] = float(np.mean(scores["fit_time"]))
    summary["score_time"] = float(np.mean(scores["score_time"]))
    return summary, pd.DataFrame(folds)


def probability_metrics(y_true, probabilities, threshold: float = 0.5) -> Dict[str, float]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(np.mean((np.asarray(probabilities) - np.asarray(y_true)) ** 2)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "specificity": float(specificity),
        "fpr": float(1.0 - specificity),
        "fnr": float(fn / (fn + tp) if fn + tp else 0.0),
    }


def evaluate_holdout(estimator, X_test, y_test, threshold: float = 0.5):
    start = time.perf_counter()
    probabilities = estimator.predict_proba(X_test)[:, 1]
    predict_time = time.perf_counter() - start
    metrics = probability_metrics(y_test, probabilities, threshold)
    metrics["predict_time"] = float(predict_time)
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(y_test, predictions, target_names=["Stay", "Churn"], output_dict=True, zero_division=0)
    fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
    roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    pr_data = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": pr_thresholds})
    return metrics, report, roc_data, pr_data, probabilities
