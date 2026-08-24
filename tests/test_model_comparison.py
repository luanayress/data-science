import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.calibration import calibration_metrics
from src.models.candidates import candidate_registry, make_pipeline
from src.models.evaluation import cross_validation_summary, probability_metrics
from src.models.selection import promotion_decision, select_challenger
from src.models.threshold import analyze_thresholds, select_thresholds


def sample_data(rows=60):
    random = np.random.RandomState(42)
    X = pd.DataFrame({
        "Age": random.randint(18, 80, rows),
        "Tenure": random.randint(0, 72, rows),
        "NumOfProducts": random.randint(1, 5, rows),
    })
    y = pd.Series(([0] * 40) + ([1] * 20))
    return X, y


def test_candidate_registry_contains_required_models():
    assert {"DummyPrior", "LogisticRegression", "RandomForest", "GradientBoosting", "HistGradientBoosting"} <= set(candidate_registry())


def test_cv_execution_and_metrics():
    X, y = sample_data()
    estimator = make_pipeline(candidate_registry()["LogisticRegression"].estimator)
    summary, folds = cross_validation_summary(estimator, X, y, StratifiedKFold(3, shuffle=True, random_state=42))
    assert 0 <= summary["cv_pr_auc_mean"] <= 1
    assert set(folds["metric"]) >= {"roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"}
    assert probability_metrics(y, np.repeat(0.4, len(y)))["tn"] == 40


def test_threshold_and_cost_calculation():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.4, 0.9])
    analysis = analyze_thresholds(y, probabilities, cost_false_negative=5, cost_false_positive=1)
    row = analysis.loc[analysis["threshold"] == 0.5].iloc[0]
    assert row["total_cost"] == row["fn"] * 5 + row["fp"]
    selected = select_thresholds(analysis)
    assert {"balanced", "high_recall", "high_precision", "lowest_cost"} <= set(selected)


def test_calibration_metrics_and_reliability_bins():
    metrics, curve = calibration_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), bins=2)
    assert metrics["brier_score"] < 0.1
    assert curve["count"].sum() == 4


def test_model_selection_policy():
    cv = {
        "ProductionV2": {"cv_pr_auc_mean": 0.55, "cv_pr_auc_std": 0.01},
        "DummyPrior": {"cv_pr_auc_mean": 0.2, "cv_pr_auc_std": 0.0},
        "Candidate": {"cv_pr_auc_mean": 0.58, "cv_pr_auc_std": 0.01},
    }
    assert select_challenger(cv) == "Candidate"
    decision, _ = promotion_decision(cv["ProductionV2"], cv["Candidate"], {"pr_auc": 0.56, "roc_auc": 0.8}, {"pr_auc": 0.59, "roc_auc": 0.81})
    assert decision.startswith("C")
