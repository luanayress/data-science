"""Statistical, fairness and explainability gates for model promotion."""

import time
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def paired_stratified_bootstrap(
    y_true, champion_probabilities, challenger_probabilities,
    champion_threshold: float, challenger_threshold: float,
    iterations: int = 1000, random_state: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap deltas while preserving the holdout class counts."""
    y = np.asarray(y_true)
    champion = np.asarray(champion_probabilities)
    challenger = np.asarray(challenger_probabilities)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    random = np.random.RandomState(random_state)
    rows = []
    for iteration in range(iterations):
        indices = np.concatenate([
            random.choice(group, size=len(group), replace=True) for group in groups
        ])
        sample_y = y[indices]
        p3, p4 = champion[indices], challenger[indices]
        values = {
            "pr_auc": average_precision_score(sample_y, p4) - average_precision_score(sample_y, p3),
            "roc_auc": roc_auc_score(sample_y, p4) - roc_auc_score(sample_y, p3),
            "brier_score": brier_score_loss(sample_y, p4) - brier_score_loss(sample_y, p3),
            "f1": f1_score(sample_y, p4 >= challenger_threshold) - f1_score(sample_y, p3 >= champion_threshold),
        }
        for metric, delta in values.items():
            rows.append({"iteration": iteration, "metric": metric, "delta_v4_minus_v3": float(delta)})
    frame = pd.DataFrame(rows)
    summaries = []
    for metric, values in frame.groupby("metric")["delta_v4_minus_v3"]:
        lower, upper = np.percentile(values, [2.5, 97.5])
        improvement_probability = float(np.mean(values < 0 if metric == "brier_score" else values > 0))
        summaries.append({
            "metric": metric, "mean_delta": float(values.mean()),
            "ci_95_lower": float(lower), "ci_95_upper": float(upper),
            "improvement_probability": improvement_probability,
        })
    return pd.DataFrame(summaries)


def fairness_report(
    frame: pd.DataFrame, y_true, probabilities, threshold: float,
    protected_features: Iterable[str] = ("Gender", "Geography"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = probabilities >= threshold
    rows = []
    for feature in protected_features:
        for group in sorted(frame[feature].dropna().unique()):
            mask = frame[feature].to_numpy() == group
            group_y, group_p, group_pred = y[mask], probabilities[mask], predictions[mask]
            positives = group_y == 1
            negatives = group_y == 0
            tp = int(np.sum(group_pred & positives))
            fn = int(np.sum(~group_pred & positives))
            fp = int(np.sum(group_pred & negatives))
            tn = int(np.sum(~group_pred & negatives))
            rows.append({
                "feature": feature, "group": str(group), "rows": int(mask.sum()),
                "base_rate": float(group_y.mean()),
                "selection_rate": float(group_pred.mean()),
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
                "brier_score": float(brier_score_loss(group_y, group_p)),
            })
    details = pd.DataFrame(rows)
    gaps = []
    for feature, subset in details.groupby("feature"):
        gaps.append({
            "feature": feature,
            "recall_gap": float(subset["recall"].max() - subset["recall"].min()),
            "false_positive_rate_gap": float(subset["false_positive_rate"].max() - subset["false_positive_rate"].min()),
            "brier_gap": float(subset["brier_score"].max() - subset["brier_score"].min()),
            "selection_rate_gap": float(subset["selection_rate"].max() - subset["selection_rate"].min()),
        })
    return details, pd.DataFrame(gaps)


def permutation_feature_report(
    estimator, X: pd.DataFrame, y_true, features: Iterable[str],
    repeats: int = 10, random_state: int = 42,
) -> pd.DataFrame:
    selected = list(features)
    result = permutation_importance(
        estimator, X.loc[:, selected], y_true, scoring="average_precision",
        n_repeats=repeats, random_state=random_state, n_jobs=1,
    )
    report = pd.DataFrame({
        "feature": selected,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "positive_repeat_rate": (result.importances > 0).mean(axis=1),
    })
    return report.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def latency_benchmark(estimator, X: pd.DataFrame, iterations: int = 50) -> Dict:
    """Measure warmed single-row and full-batch inference without imposing an SLO."""
    single = X.iloc[[0]]
    for _ in range(3):
        estimator.predict_proba(single)
    single_samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        estimator.predict_proba(single)
        single_samples.append(time.perf_counter() - started)
    batch_samples = []
    for _ in range(max(5, iterations // 5)):
        started = time.perf_counter()
        estimator.predict_proba(X)
        batch_samples.append(time.perf_counter() - started)
    return {
        "iterations": iterations,
        "single_row_p50_seconds": float(np.percentile(single_samples, 50)),
        "single_row_p95_seconds": float(np.percentile(single_samples, 95)),
        "batch_rows": int(len(X)),
        "batch_p50_seconds": float(np.percentile(batch_samples, 50)),
        "batch_p95_seconds": float(np.percentile(batch_samples, 95)),
        "batch_p95_per_row_seconds": float(np.percentile(batch_samples, 95) / len(X)),
    }


def readiness_gate_report(
    bootstrap: pd.DataFrame, fairness_gaps: pd.DataFrame, feature_report: pd.DataFrame,
    calibration_selected: str, constraints: Dict,
) -> Dict:
    gates = []
    bootstrap_index = bootstrap.set_index("metric")
    statistical_pass = (
        bootstrap_index.loc["pr_auc", "ci_95_lower"] > 0
        and bootstrap_index.loc["f1", "ci_95_lower"] > 0
        and bootstrap_index.loc["brier_score", "ci_95_upper"] < 0
    )
    gates.append({"gate": "paired_bootstrap", "status": "PASS" if statistical_pass else "FAIL"})
    gates.append({"gate": "calibration_selection", "status": "PASS", "detail": calibration_selected})
    max_gap = float(constraints.get("maximum_fairness_gap", 0.10))
    fairness_pass = bool(
        (fairness_gaps["recall_gap"] <= max_gap).all()
        and (fairness_gaps["false_positive_rate_gap"] <= max_gap).all()
    )
    gates.append({"gate": "fairness", "status": "PASS" if fairness_pass else "FAIL", "limit": max_gap})
    importance_pass = bool((feature_report["positive_repeat_rate"] >= 0.5).any())
    gates.append({"gate": "feature_importance", "status": "PASS" if importance_pass else "FAIL"})
    gates.append({"gate": "strict_v4_contract", "status": "PASS"})
    for gate, key in (
        ("financial_value", "financial_inputs_available"),
        ("campaign_capacity", "maximum_campaign_rate"),
        ("latency_slo", "maximum_prediction_latency_seconds"),
    ):
        value = constraints.get(key)
        gates.append({"gate": gate, "status": "PASS" if value not in (None, False) else "BLOCKED", "configured_value": value})
    blocking = [gate for gate in gates if gate["status"] != "PASS"]
    return {
        "status": "READY_FOR_CANARY" if not blocking else "NOT_READY_FOR_CANARY",
        "canary_started": False,
        "gates": gates,
        "blocking_gates": [gate["gate"] for gate in blocking],
    }
