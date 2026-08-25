from dataclasses import replace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import app.api as api_module
from app.schema import PredictionRequest
from src.business.churn_promotion import (
    best_under_capacity, cost_sensitivity, evaluate_promotion,
    financial_value, scenario_metrics,
)
from src.pipelines.inference_pipeline import InferencePipeline
from tests.test_api_integration import VALID_PAYLOAD


def test_relative_cost_campaign_rate_and_nnc():
    metrics = scenario_metrics([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], 0.5, 5, 1)
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 1
    assert metrics["relative_cost"] == 6
    assert metrics["campaign_rate"] == 0.5
    assert metrics["number_needed_to_contact"] == 2
    assert metrics["churn_coverage"] == 0.5


def test_financial_value_requires_explicit_inputs():
    metrics = {"tp": 10, "fp": 5, "fn": 2}
    assert financial_value(metrics, {"average_customer_value": None}) is None
    result = financial_value(metrics, {
        "average_customer_value": 100, "retention_campaign_cost": 10,
        "retention_success_rate": 0.5, "false_positive_contact_cost": 2,
    })
    assert result["expected_retention_value"] == 500
    assert result["expected_net_value"] == 140


def test_cost_sensitivity_and_capacity():
    analysis = pd.DataFrame([
        {"threshold": 0.2, "precision": 0.4, "recall": 0.8, "f1": 0.53, "tn": 50, "fp": 30, "fn": 4, "tp": 16},
        {"threshold": 0.5, "precision": 0.7, "recall": 0.5, "f1": 0.58, "tn": 75, "fp": 5, "fn": 10, "tp": 10},
    ])
    sensitivity = cost_sensitivity(analysis, [2, 10])
    assert list(sensitivity["best_threshold"]) == [0.5, 0.2]
    assert best_under_capacity(analysis) is None
    constrained = best_under_capacity(analysis, max_campaign_rate=0.2)
    assert constrained["threshold"] == 0.5


def test_promotion_policy_is_technical_without_financial_inputs():
    champion = {"pr_auc": 0.59, "roc_auc": 0.82, "brier_score": 0.12, "log_loss": 0.38, "recall": 0.43, "precision": 0.62, "campaign_rate": 0.14, "predicted_positive_count": 286, "threshold": 0.5}
    challenger = {"pr_auc": 0.62, "roc_auc": 0.83, "brier_score": 0.116, "log_loss": 0.37, "recall": 0.56, "precision": 0.59, "campaign_rate": 0.195, "predicted_positive_count": 390, "threshold": 0.34}
    constraints = {"pr_auc_tolerance": 0, "roc_auc_tolerance": 0.005, "brier_tolerance": 0, "log_loss_tolerance": 0, "minimum_recall_gain": 0.05, "maximum_precision_drop": 0.1, "maximum_campaign_rate": None, "max_customers_contacted": None}
    decision = evaluate_promotion(champion, challenger, constraints)
    assert decision.status == "PROMOTION_RECOMMENDED"
    assert decision.classification == "PROMOTE_V3_TECHNICALLY"
    assert decision.approved is False


def test_threshold_profiles_are_model_specific():
    v2 = InferencePipeline("v2", threshold_profile="balanced")
    v3 = InferencePipeline("v3", threshold_profile="high_recall")
    assert v2.threshold == 0.5
    assert v3.threshold == 0.21


def test_shadow_loads_once_and_preserves_champion_response(monkeypatch):
    loads = []

    class FakeInference:
        def __init__(self, version, threshold_profile=None):
            loads.append(version)
            self.version = version
            self.metadata = {"version": version, "algorithm": "Fake", "model_features": ["x"], "metrics": {}}

        def predict_with_confidence(self, frame):
            probability = 0.2 if self.version == "v2" else 0.8
            return {"predictions": [0 if self.version == "v2" else 1] * len(frame), "probabilities": [probability] * len(frame), "high_confidence": [True] * len(frame)}

    monkeypatch.setattr(api_module, "InferencePipeline", FakeInference)
    monkeypatch.setattr(api_module, "settings", replace(api_module.settings, model_version="v2", shadow_model_version="v3", churn_threshold_profile=None))
    application = api_module.create_app()
    with TestClient(application) as client:
        first = client.post("/predict", json=VALID_PAYLOAD).json()
        second = client.post("/predict", json=VALID_PAYLOAD).json()
    assert first == second == {"prediction": 0, "probability": 0.2, "confidence": "high"}
    assert loads == ["v2", "v3"]
