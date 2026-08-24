"""Train and evaluate the leakage-safe expanded bank churn challenger."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_customer_churn_data
from src.data.split import split_train_test
from src.features.bank_feature_contract import (
    CATEGORICAL_FEATURES, EXCLUDED_FEATURES, EXPANDED_FEATURES, FEATURE_GROUPS,
    NUMERIC_FEATURES, SOURCE_RENAMES, TARGET_COLUMN,
)
from src.models.calibration import calibration_metrics, make_calibrated
from src.models.evaluation import cross_validation_summary, evaluate_holdout
from src.models.readiness import (
    fairness_report, latency_benchmark, paired_stratified_bootstrap, permutation_feature_report,
    readiness_gate_report,
)
from src.models.registry import ModelRegistry
from src.models.threshold import analyze_thresholds, select_thresholds
from src.utils.config import load_config


def prepare_bank_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize safe source names without creating target-derived features."""
    return data.rename(columns=SOURCE_RENAMES).copy()


def create_expanded_pipeline(features: Iterable[str], random_state: int = 42) -> Pipeline:
    selected = tuple(features)
    numeric = [name for name in selected if name in NUMERIC_FEATURES]
    categorical = [name for name in selected if name in CATEGORICAL_FEATURES]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=None)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return Pipeline([
        ("preprocessing", ColumnTransformer(transformers, remainder="drop")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=0.1, random_state=random_state,
        )),
    ])


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _add_operational_metrics(metrics: Dict[str, Any], rows: int, fn_cost: float, fp_cost: float) -> Dict[str, Any]:
    enriched = dict(metrics)
    contacts = int(metrics["tp"] + metrics["fp"])
    enriched.update({
        "predicted_positive_count": contacts,
        "campaign_rate": float(contacts / rows),
        "churn_coverage": float(metrics["recall"]),
        "number_needed_to_contact": float(contacts / metrics["tp"]) if metrics["tp"] else None,
        "relative_cost": float(metrics["fn"] * fn_cost + metrics["fp"] * fp_cost),
    })
    return enriched


def run_feature_expansion(
    config_file: str = "configs/feature_expansion.yaml",
    save_challenger: bool = True,
) -> Dict[str, Any]:
    config = load_config(config_file)["experiment"]
    random_state = int(config["random_state"])
    folds = int(config["cv_folds"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_bank_data(load_customer_churn_data())
    required = set(EXPANDED_FEATURES) | {TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Bank dataset is missing columns: {}".format(sorted(missing)))
    X = data.loc[:, list(EXPANDED_FEATURES)]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=float(config["test_size"]), random_state=random_state, stratify=True,
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    ablation_rows = []
    fold_frames = []
    for group_name, features in FEATURE_GROUPS.items():
        estimator = create_expanded_pipeline(features, random_state)
        summary, fold_frame = cross_validation_summary(estimator, X_train, y_train, cv, n_jobs=int(config["n_jobs"]))
        ablation_rows.append({"feature_group": group_name, "feature_count": len(features), **summary})
        fold_frame.insert(0, "feature_group", group_name)
        fold_frames.append(fold_frame)

    ablation = pd.DataFrame(ablation_rows).sort_values("cv_pr_auc_mean", ascending=False)
    selected_group = str(ablation.iloc[0]["feature_group"])
    selected_features = FEATURE_GROUPS[selected_group]
    base_pipeline = create_expanded_pipeline(selected_features, random_state)

    calibration_candidates = {
        "uncalibrated": (
            cross_val_predict(base_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=int(config["n_jobs"]))[:, 1],
            base_pipeline,
        )
    }
    for method in ("sigmoid", "isotonic"):
        estimator = make_calibrated(base_pipeline, method, inner_cv)
        probabilities = cross_val_predict(
            estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=int(config["n_jobs"]),
        )[:, 1]
        calibration_candidates[method] = (probabilities, estimator)
    calibration_rows = []
    for method, (probabilities, _) in calibration_candidates.items():
        calibration_result, _ = calibration_metrics(y_train, probabilities, bins=10)
        calibration_rows.append({"calibration": method, **calibration_result})
    selected_calibration = min(calibration_rows, key=lambda row: row["brier_score"])["calibration"]
    oof_probabilities, calibrated = calibration_candidates[selected_calibration]
    threshold_analysis = analyze_thresholds(
        y_train, oof_probabilities,
        cost_false_negative=float(config["cost_false_negative"]),
        cost_false_positive=float(config["cost_false_positive"]),
    )
    thresholds = select_thresholds(threshold_analysis)
    selected_threshold = thresholds["balanced"]
    calibrated.fit(X_train, y_train)
    metrics, _, roc_curve, pr_curve, probabilities = evaluate_holdout(
        calibrated, X_test, y_test, selected_threshold,
    )

    v3 = ModelRegistry("v3").load_pipeline()
    v3_metadata = ModelRegistry("v3").load_metadata()
    v3_metrics, _, _, _, v3_probabilities = evaluate_holdout(
        v3, X_test, y_test, float(v3_metadata.get("selected_threshold", 0.34)),
    )
    fn_cost = float(config["cost_false_negative"])
    fp_cost = float(config["cost_false_positive"])
    metrics = _add_operational_metrics(metrics, len(X_test), fn_cost, fp_cost)
    v3_metrics = _add_operational_metrics(v3_metrics, len(X_test), fn_cost, fp_cost)
    decision = "PROMOTION_RECOMMENDED" if (
        metrics["pr_auc"] > v3_metrics["pr_auc"]
        and metrics["brier_score"] <= v3_metrics["brier_score"]
        and metrics["relative_cost"] < v3_metrics["relative_cost"]
    ) else "KEEP_V3_CHALLENGER"

    bootstrap = paired_stratified_bootstrap(
        y_test, v3_probabilities, probabilities,
        float(v3_metadata.get("selected_threshold", 0.34)), selected_threshold,
        iterations=int(config["bootstrap_iterations"]), random_state=random_state,
    )
    fairness_details, fairness_gaps = fairness_report(
        X_test, y_test, probabilities, selected_threshold,
    )
    feature_importance = permutation_feature_report(
        calibrated, X_test, y_test, selected_features,
        repeats=int(config["permutation_repeats"]), random_state=random_state,
    )
    latency = latency_benchmark(calibrated, X_test.loc[:, list(selected_features)])
    readiness = readiness_gate_report(
        bootstrap, fairness_gaps, feature_importance, selected_calibration, config,
    )

    metadata = {
        "name": "churn", "version": config["challenger_version"],
        "algorithm": type(calibrated).__name__, "base_algorithm": "HistGradientBoostingClassifier",
        "target": TARGET_COLUMN, "domain": "bank_churn",
        "raw_features": list(selected_features), "model_features": list(selected_features),
        "strict_input_contract": True,
        "feature_group": selected_group, "excluded_features": list(EXCLUDED_FEATURES),
        "trained_at": datetime.now(timezone.utc).isoformat(), "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)), "random_state": random_state,
        "cv": {"folds": folds, "primary_metric": "average_precision",
               "results": _safe(ablation.iloc[0].to_dict())},
        "metrics": metrics, "v3_holdout_metrics": v3_metrics,
        "cost_assumptions": {"false_negative": fn_cost, "false_positive": fp_cost},
        "calibration": {"method": selected_calibration, "oof_results": calibration_rows}, "thresholds": thresholds,
        "threshold": selected_threshold, "selected_threshold": selected_threshold,
        "high_confidence_threshold": 0.8, "decision": decision,
        "readiness_status": readiness["status"],
    }
    if save_challenger:
        ModelRegistry(config["challenger_version"]).save_pipeline(calibrated, _safe(metadata))

    ablation.to_csv(output_dir / "feature_ablation.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(output_dir / "cross_validation_results.csv", index=False)
    threshold_analysis.to_csv(output_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_results.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)
    fairness_details.to_csv(output_dir / "fairness_by_group.csv", index=False)
    fairness_gaps.to_csv(output_dir / "fairness_gaps.csv", index=False)
    feature_importance.to_csv(output_dir / "permutation_feature_importance.csv", index=False)
    (output_dir / "latency_benchmark.json").write_text(
        json.dumps(_safe(latency), indent=2), encoding="utf-8",
    )
    pd.DataFrame([
        {"model": "v3", "threshold": v3_metadata.get("selected_threshold", 0.34), **v3_metrics},
        {"model": config["challenger_version"], "threshold": selected_threshold, **metrics},
    ]).to_csv(output_dir / "holdout_comparison.csv", index=False)
    roc_curve.to_csv(output_dir / "v4_roc_curve.csv", index=False)
    pr_curve.to_csv(output_dir / "v4_precision_recall_curve.csv", index=False)
    (output_dir / "experiment.json").write_text(json.dumps(_safe({
        "selected_group": selected_group, "selected_features": selected_features,
        "excluded_features": EXCLUDED_FEATURES, "decision": decision,
        "thresholds": thresholds, "v3_metrics": v3_metrics, "v4_metrics": metrics,
        "holdout_probability_rows": len(probabilities),
        "calibration_results": calibration_rows, "latency_benchmark": latency,
        "readiness": readiness,
    }), indent=2), encoding="utf-8")
    (output_dir / "readiness_decision.json").write_text(
        json.dumps(_safe(readiness), indent=2), encoding="utf-8",
    )
    return {"pipeline": calibrated, "metadata": metadata, "ablation": ablation,
            "metrics": metrics, "v3_metrics": v3_metrics, "decision": decision}
