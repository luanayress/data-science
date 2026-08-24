"""Explicit champion/challenger selection policy."""

from typing import Dict, Tuple


def select_challenger(cv_results: Dict[str, Dict[str, float]], production_name: str = "ProductionV2") -> str:
    eligible = {name: values for name, values in cv_results.items() if name not in {"DummyPrior", production_name}}
    return max(eligible, key=lambda name: eligible[name]["cv_pr_auc_mean"])


def promotion_decision(production: Dict[str, float], challenger: Dict[str, float], production_holdout: Dict[str, float], challenger_holdout: Dict[str, float]) -> Tuple[str, str]:
    cv_gain = challenger["cv_pr_auc_mean"] - production["cv_pr_auc_mean"]
    stable_gain = cv_gain >= max(0.01, production["cv_pr_auc_std"])
    holdout_confirmed = challenger_holdout["pr_auc"] >= production_holdout["pr_auc"] and challenger_holdout["roc_auc"] >= production_holdout["roc_auc"] - 0.005
    if stable_gain and holdout_confirmed:
        return "C — CREATE V3 AND RECOMMEND PROMOTION", "PR-AUC CV gain is material and holdout confirms it"
    if cv_gain > 0 and holdout_confirmed:
        return "B — CREATE V3, DO NOT PROMOTE", "challenger improves CV and holdout, but the gain is below the promotion margin"
    return "A — KEEP V2", "no challenger provides sufficiently strong and confirmed evidence"
