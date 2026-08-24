"""Candidate registry for leakage-safe churn experiments."""

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import ChurnFeatureEngineer


@dataclass
class Candidate:
    name: str
    estimator: Any
    search_space: Dict[str, List[Any]]
    search_iterations: int = 0


@dataclass
class OptionalCandidateStatus:
    library: str
    status: str
    reason: str


def make_pipeline(estimator: Any) -> Pipeline:
    """Give every candidate identical feature engineering and scaling."""
    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessing", StandardScaler()),
        ("model", estimator),
    ])


def candidate_registry(random_state: int = 42) -> Dict[str, Candidate]:
    return {
        "DummyPrior": Candidate(
            "DummyPrior", DummyClassifier(strategy="prior"), {}, 0
        ),
        "LogisticRegression": Candidate(
            "LogisticRegression",
            LogisticRegression(max_iter=2000, random_state=random_state),
            {
                "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            8,
        ),
        "RandomForest": Candidate(
            "RandomForest",
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 4, 7, 10],
                "model__min_samples_leaf": [1, 3, 8, 15],
                "model__max_features": ["sqrt", 0.7],
                "model__class_weight": [None, "balanced"],
            },
            10,
        ),
        "GradientBoosting": Candidate(
            "GradientBoosting",
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [75, 100, 150, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [2, 5, 10, 20],
                "model__subsample": [0.8, 1.0],
            },
            10,
        ),
        "HistGradientBoosting": Candidate(
            "HistGradientBoosting",
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [100, 150, 250],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__min_samples_leaf": [10, 20, 40],
            },
            10,
        ),
    }


def optional_candidate_statuses() -> List[OptionalCandidateStatus]:
    reasons = {
        "xgboost": "not installed; optional dependency was not added automatically",
        "lightgbm": "not installed; optional dependency was not added automatically",
        "catboost": "not installed; optional dependency was not added automatically",
    }
    return [
        OptionalCandidateStatus(
            library=name,
            status="AVAILABLE" if importlib.util.find_spec(name) else "SKIPPED",
            reason="installed" if importlib.util.find_spec(name) else reason,
        )
        for name, reason in reasons.items()
    ]
