"""Calibration assessment without learning from the holdout."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict


def make_calibrated(estimator, method: str, cv) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv, n_jobs=1)


def calibration_metrics(y_true, probabilities, bins: int = 10) -> Tuple[Dict[str, float], pd.DataFrame]:
    probabilities = np.asarray(probabilities)
    y_array = np.asarray(y_true)
    fraction, mean_probability = calibration_curve(y_array, probabilities, n_bins=bins, strategy="quantile")
    categories = pd.qcut(probabilities, q=bins, duplicates="drop")
    counts = pd.Series(categories).value_counts(sort=False).to_numpy()
    curve = pd.DataFrame({
        "mean_predicted_probability": mean_probability,
        "fraction_of_positives": fraction,
        "count": counts[:len(fraction)],
    })
    return {
        "brier_score": float(np.mean((probabilities - y_array) ** 2)),
        "log_loss": float(log_loss(y_array, probabilities, labels=[0, 1])),
    }, curve


def calibrated_oof_probabilities(estimator, X, y, method: str, outer_cv, inner_cv):
    calibrated = make_calibrated(estimator, method, inner_cv)
    return cross_val_predict(calibrated, X, y, cv=outer_cv, method="predict_proba", n_jobs=1)[:, 1]
