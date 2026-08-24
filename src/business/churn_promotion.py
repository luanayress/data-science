"""Configurable cost, campaign, and promotion policy calculations."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


PROMOTION_STATUSES = {
    "KEEP_CHAMPION", "PROMOTION_RECOMMENDED", "PROMOTION_APPROVED",
    "PROMOTION_REJECTED", "NEEDS_BUSINESS_INPUT",
}


@dataclass
class PromotionDecision:
    approved: bool
    status: str
    classification: str
    reasons: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    threshold: float
    business_scenario: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def scenario_metrics(y_true, probabilities, threshold: float, fn_cost: float, fp_cost: float) -> Dict[str, float]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    total = len(y_true)
    contacts = int(tp + fp)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_positive_count": contacts,
        "predicted_positive_rate": float(contacts / total),
        "campaign_rate": float(contacts / total),
        "churn_coverage": float(tp / (tp + fn) if tp + fn else 0.0),
        "number_needed_to_contact": float(contacts / tp if tp else float("inf")),
        "relative_cost": float(fn * fn_cost + fp * fp_cost),
    }


def financial_value(metrics: Dict[str, float], parameters: Dict[str, Optional[float]]) -> Optional[Dict[str, float]]:
    required = ("average_customer_value", "retention_campaign_cost", "retention_success_rate", "false_positive_contact_cost")
    if any(parameters.get(name) is None for name in required):
        return None
    value = float(parameters["average_customer_value"])
    campaign = float(parameters["retention_campaign_cost"])
    success = float(parameters["retention_success_rate"])
    contact = float(parameters["false_positive_contact_cost"])
    saved = metrics["tp"] * value * success
    campaign_cost = (metrics["tp"] + metrics["fp"]) * campaign
    false_positive_cost = metrics["fp"] * contact
    missed = metrics["fn"] * value
    return {
        "expected_retention_value": float(saved),
        "expected_campaign_cost": float(campaign_cost + false_positive_cost),
        "missed_churn_cost": float(missed),
        "expected_net_value": float(saved - campaign_cost - false_positive_cost - missed),
    }


def cost_sensitivity(oof_analysis: pd.DataFrame, ratios: List[float]) -> pd.DataFrame:
    rows = []
    for ratio in ratios:
        frame = oof_analysis.copy()
        frame["cost"] = frame["fn"].astype(float) * float(ratio) + frame["fp"].astype(float)
        best = frame.loc[frame["cost"].idxmin()]
        rows.append({
            "fn_fp_ratio": float(ratio), "best_threshold": float(best["threshold"]),
            "precision": float(best["precision"]), "recall": float(best["recall"]),
            "f1": float(best["f1"]), "tn": int(best["tn"]), "fp": int(best["fp"]),
            "fn": int(best["fn"]), "tp": int(best["tp"]), "cost": float(best["cost"]),
            "predicted_positive_count": int(best["tp"] + best["fp"]),
            "predicted_positive_rate": float((best["tp"] + best["fp"]) / (best["tn"] + best["fp"] + best["fn"] + best["tp"])),
        })
    return pd.DataFrame(rows)


def best_under_capacity(oof_analysis: pd.DataFrame, max_campaign_rate=None, max_customers_contacted=None) -> Optional[Dict[str, float]]:
    if max_campaign_rate is None and max_customers_contacted is None:
        return None
    frame = oof_analysis.copy()
    total = frame[["tn", "fp", "fn", "tp"]].iloc[0].sum()
    frame["campaign_rate"] = (frame["tp"] + frame["fp"]) / total
    if max_campaign_rate is not None:
        frame = frame[frame["campaign_rate"] <= float(max_campaign_rate)]
    if max_customers_contacted is not None:
        frame = frame[(frame["tp"] + frame["fp"]) <= int(max_customers_contacted)]
    if frame.empty:
        return None
    best = frame.sort_values(["recall", "precision"], ascending=False).iloc[0]
    return {"threshold": float(best["threshold"]), "precision": float(best["precision"]), "recall": float(best["recall"]), "campaign_rate": float(best["campaign_rate"]), "customers_contacted": int(best["tp"] + best["fp"])}


def evaluate_promotion(champion: Dict[str, float], challenger: Dict[str, float], constraints: Dict[str, Any], financial_champion=None, financial_challenger=None) -> PromotionDecision:
    reasons, warnings = [], []
    checks = {
        "PR-AUC improved": challenger["pr_auc"] >= champion["pr_auc"] - float(constraints["pr_auc_tolerance"]),
        "ROC-AUC preserved": challenger["roc_auc"] >= champion["roc_auc"] - float(constraints["roc_auc_tolerance"]),
        "Brier improved": challenger["brier_score"] <= champion["brier_score"] + float(constraints["brier_tolerance"]),
        "Log Loss improved": challenger["log_loss"] <= champion["log_loss"] + float(constraints["log_loss_tolerance"]),
        "Recall gain acceptable": challenger["recall"] - champion["recall"] >= float(constraints["minimum_recall_gain"]),
        "Precision drop acceptable": champion["precision"] - challenger["precision"] <= float(constraints["maximum_precision_drop"]),
    }
    max_rate = constraints.get("maximum_campaign_rate")
    if max_rate is not None:
        checks["Campaign capacity respected"] = challenger["campaign_rate"] <= float(max_rate)
    max_contacts = constraints.get("max_customers_contacted")
    if max_contacts is not None:
        checks["Contact capacity respected"] = challenger["predicted_positive_count"] <= int(max_contacts)
    max_latency = constraints.get("maximum_prediction_latency_seconds")
    if max_latency is not None:
        checks["Latency SLO respected"] = challenger["predict_time"] <= float(max_latency)
    for label, passed in checks.items():
        (reasons if passed else warnings).append(label)
    technical = all(checks.values())
    if not technical:
        status, classification = "PROMOTION_REJECTED", "KEEP_V2"
    elif financial_champion is None or financial_challenger is None:
        status, classification = "PROMOTION_RECOMMENDED", "PROMOTE_V3_TECHNICALLY"
        warnings.append("Real financial inputs are unavailable; business validation is pending")
    elif financial_challenger["expected_net_value"] > financial_champion["expected_net_value"]:
        status, classification = "PROMOTION_APPROVED", "PROMOTE_V3_BUSINESS_VALIDATED"
        reasons.append("Expected net value improves")
    else:
        status, classification = "NEEDS_BUSINESS_INPUT", "NEEDS_BUSINESS_INPUT"
        warnings.append("Configured financial scenario does not confirm higher net value")
    if max_latency is None:
        warnings.append("No production latency SLO is configured")
    return PromotionDecision(
        approved=status == "PROMOTION_APPROVED", status=status, classification=classification,
        reasons=reasons, warnings=warnings,
        metrics={"champion": champion, "challenger": challenger},
        threshold=float(challenger["threshold"]), business_scenario="v3_balanced",
    )
