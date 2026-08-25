"""OOF threshold and campaign-cost analysis."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def analyze_thresholds(y_true, probabilities, cost_false_negative: float = 5.0, cost_false_positive: float = 1.0) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "accuracy": float((tp + tn) / len(y_true)),
            "specificity": float(specificity), "fpr": float(1 - specificity),
            "fnr": float(fn / (fn + tp) if fn + tp else 0.0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "total_cost": float(fn * cost_false_negative + fp * cost_false_positive),
        })
    return pd.DataFrame(rows)


def select_thresholds(analysis: pd.DataFrame) -> Dict[str, float]:
    balanced = analysis.loc[analysis["f1"].idxmax()]
    precision_floor = max(0.35, float(balanced["precision"]) * 0.75)
    recall_candidates = analysis[analysis["precision"] >= precision_floor]
    high_recall = recall_candidates.sort_values(["recall", "precision"], ascending=False).iloc[0]
    recall_floor = max(0.30, float(balanced["recall"]) * 0.75)
    precision_candidates = analysis[analysis["recall"] >= recall_floor]
    high_precision = precision_candidates.sort_values(["precision", "recall"], ascending=False).iloc[0]
    lowest_cost = analysis.loc[analysis["total_cost"].idxmin()]
    return {
        "default": 0.5,
        "balanced": float(balanced["threshold"]),
        "high_recall": float(high_recall["threshold"]),
        "high_precision": float(high_precision["threshold"]),
        "lowest_cost": float(lowest_cost["threshold"]),
        "high_recall_precision_floor": float(precision_floor),
        "high_precision_recall_floor": float(recall_floor),
    }
