"""Evaluate v2/v3 promotion using persisted artifacts and OOF results only."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src.business.churn_promotion import (
    best_under_capacity, cost_sensitivity, evaluate_promotion,
    financial_value, scenario_metrics,
)
from src.data.load_data import load_customer_churn_data
from src.data.split import split_train_test
from src.features.build_features import get_features_for_modeling
from src.models.registry import ModelRegistry
from src.utils.config import load_config

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _probability_quality(y_true, probabilities) -> Dict[str, float]:
    values = np.asarray(probabilities)
    target = np.asarray(y_true)
    return {
        "roc_auc": float(roc_auc_score(target, values)),
        "pr_auc": float(average_precision_score(target, values)),
        "brier_score": float(np.mean((values - target) ** 2)),
        "log_loss": float(log_loss(target, values, labels=[0, 1])),
    }


def run_promotion_evaluation(config_file: str = "configs/churn_business.yaml") -> Dict[str, Any]:
    config = load_config(config_file)
    promotion = config["promotion"]
    reports = config["reports"]
    output_dir = Path(reports["promotion_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_customer_churn_data()
    X, y = get_features_for_modeling(data)
    _, X_holdout, _, y_holdout = split_train_test(X, y, test_size=0.2, random_state=42, stratify=True)
    champion_registry = ModelRegistry(promotion["champion_version"])
    challenger_registry = ModelRegistry(promotion["challenger_version"])
    champion_pipeline = champion_registry.load_pipeline()
    challenger_pipeline = challenger_registry.load_pipeline()
    champion_metadata = champion_registry.load_metadata()
    challenger_metadata = challenger_registry.load_metadata()
    champion_probabilities = champion_pipeline.predict_proba(X_holdout)[:, 1]
    challenger_probabilities = challenger_pipeline.predict_proba(X_holdout)[:, 1]

    fn_cost = float(config["cost"]["false_negative_cost"])
    fp_cost = float(config["cost"]["false_positive_cost"])
    thresholds = challenger_metadata["thresholds"]
    scenario_definitions = [
        ("v2_current", "v2", float(champion_metadata["threshold"]), champion_probabilities),
        ("v3_balanced", "v3", float(thresholds["balanced"]), challenger_probabilities),
        ("v3_high_recall", "v3", float(thresholds["high_recall"]), challenger_probabilities),
        ("v3_high_precision", "v3", float(thresholds["high_precision"]), challenger_probabilities),
    ]
    scenario_rows = []
    holdout_results = pd.read_csv(Path(reports["model_comparison_dir"]) / "holdout_results.csv")
    latency_by_version = {
        "v2": float(holdout_results.loc[holdout_results["model"] == "ProductionV2", "predict_time"].iloc[0]),
        "v3": float(holdout_results.loc[holdout_results["model"] == "HistGradientBoosting+sigmoid", "predict_time"].iloc[0]),
    }
    for scenario, version, threshold, probabilities in scenario_definitions:
        row = scenario_metrics(y_holdout, probabilities, threshold, fn_cost, fp_cost)
        row.update(_probability_quality(y_holdout, probabilities))
        row["predict_time"] = latency_by_version[version]
        row.update({"model_version": version, "scenario": scenario, "cost_mode": config["cost"]["mode"]})
        financial = financial_value(row, config["financial"])
        if financial:
            row.update(financial)
        scenario_rows.append(row)
    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(output_dir / "promotion_comparison.csv", index=False)
    scenarios.to_csv(output_dir / "campaign_analysis.csv", index=False)

    champion = scenario_rows[0]
    challenger = scenario_rows[1]
    uplift = {
        "additional_customers_contacted": challenger["predicted_positive_count"] - champion["predicted_positive_count"],
        "additional_churners_identified": challenger["tp"] - champion["tp"],
        "additional_false_positives": challenger["fp"] - champion["fp"],
        "false_negatives_avoided": champion["fn"] - challenger["fn"],
        "campaign_volume_change_pct": float((challenger["predicted_positive_count"] / champion["predicted_positive_count"] - 1) * 100),
    }

    oof = pd.read_csv(Path(reports["model_comparison_dir"]) / "threshold_analysis.csv")
    sensitivity = cost_sensitivity(oof, config["cost"]["sensitivity_fn_fp_ratios"])
    sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    capacity = best_under_capacity(
        oof, promotion.get("maximum_campaign_rate"), promotion.get("max_customers_contacted")
    )

    champion_financial = financial_value(champion, config["financial"])
    challenger_financial = financial_value(challenger, config["financial"])
    decision = evaluate_promotion(
        champion, challenger, promotion, champion_financial, challenger_financial
    )

    champion_predictions = (champion_probabilities >= champion["threshold"]).astype(int)
    challenger_predictions = (challenger_probabilities >= challenger["threshold"]).astype(int)
    differences = challenger_probabilities - champion_probabilities
    categories = np.select(
        [
            (champion_predictions == 0) & (challenger_predictions == 1),
            (champion_predictions == 1) & (challenger_predictions == 0),
            (champion_predictions == 1) & (challenger_predictions == 1),
        ],
        ["v2_stay_v3_churn", "v2_churn_v3_stay", "both_churn"],
        default="both_stay",
    )
    disagreements = pd.DataFrame({
        "holdout_row": X_holdout.index,
        "v2_probability": champion_probabilities,
        "v3_probability": challenger_probabilities,
        "probability_delta": differences,
        "v2_prediction": champion_predictions,
        "v3_prediction": challenger_predictions,
        "agreement": champion_predictions == challenger_predictions,
        "category": categories,
    })
    disagreements.to_csv(output_dir / "model_disagreements.csv", index=False)
    agreement = {
        "agreement_rate": float(np.mean(champion_predictions == challenger_predictions)),
        "mean_probability_delta": float(np.mean(differences)),
        "median_probability_delta": float(np.median(differences)),
        "p95_probability_delta": float(np.percentile(np.abs(differences), 95)),
        "categories": {str(key): int(value) for key, value in pd.Series(categories).value_counts().items()},
    }
    payload = decision.to_dict()
    payload.update({
        "cost_mode": config["cost"]["mode"],
        "financial_inputs_available": champion_financial is not None,
        "financial_champion": champion_financial,
        "financial_challenger": challenger_financial,
        "uplift": uplift, "agreement": agreement,
        "capacity_constraint": capacity,
        "rollback": "MODEL_VERSION=v2",
        "shadow_example": "MODEL_VERSION=v2; SHADOW_MODEL_VERSION=v3",
    })
    (output_dir / "promotion_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"decision": payload, "scenarios": scenario_rows, "sensitivity": sensitivity.to_dict("records"), "agreement": agreement, "uplift": uplift, "output_dir": str(output_dir)}
