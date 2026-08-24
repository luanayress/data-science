"""Reproducible champion/challenger experiment with an untouched holdout."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from src.data.load_data import load_customer_churn_data
from src.data.split import split_train_test
from src.features.build_features import get_features_for_modeling
from src.features.feature_contract import IGNORED_HTTP_FEATURES, MODEL_FEATURES, RAW_FEATURES, TARGET_COLUMN
from src.models.calibration import calibrated_oof_probabilities, calibration_metrics, make_calibrated
from src.models.candidates import candidate_registry, make_pipeline, optional_candidate_statuses
from src.models.evaluation import cross_validation_summary, evaluate_holdout
from src.models.registry import ModelRegistry
from src.models.selection import promotion_decision, select_challenger
from src.models.threshold import analyze_thresholds, select_thresholds
from src.utils.config import load_config

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def run_model_comparison(config_file: str = "configs/model_comparison.yaml", save_challenger: bool = True) -> Dict[str, Any]:
    config = load_config(config_file)["experiment"]
    random_state = int(config["random_state"])
    folds = int(config["cv_folds"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_customer_churn_data()
    X, y = get_features_for_modeling(data)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=float(config["test_size"]), random_state=random_state, stratify=True
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    production_pipeline = ModelRegistry("v2").load_pipeline()
    fitted_estimators = {"ProductionV2": production_pipeline}
    best_parameters = {"ProductionV2": production_pipeline.get_params(deep=False)}
    search_rows = []

    for name, candidate in candidate_registry(random_state).items():
        pipeline = make_pipeline(candidate.estimator)
        if candidate.search_iterations:
            search = RandomizedSearchCV(
                pipeline, candidate.search_space, n_iter=candidate.search_iterations,
                scoring=config["primary_metric"], cv=cv, random_state=random_state,
                n_jobs=int(config["n_jobs"]), refit=True, return_train_score=False,
            )
            started = time.perf_counter()
            search.fit(X_train, y_train)
            fitted_estimators[name] = search.best_estimator_
            best_parameters[name] = search.best_params_
            search_rows.append({
                "model": name, "best_score": float(search.best_score_),
                "search_iterations": candidate.search_iterations,
                "search_time": float(time.perf_counter() - started),
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            })
        else:
            fitted_estimators[name] = pipeline.fit(X_train, y_train)
            best_parameters[name] = {}
            search_rows.append({"model": name, "best_score": None, "search_iterations": 0, "search_time": 0.0, "best_parameters": "{}"})

    cv_results = {}
    cv_frames = []
    comparison_rows = []
    for name, estimator in fitted_estimators.items():
        summary, folds_frame = cross_validation_summary(clone(estimator), X_train, y_train, cv, n_jobs=int(config["n_jobs"]))
        cv_results[name] = summary
        folds_frame.insert(0, "model", name)
        cv_frames.append(folds_frame)
        comparison_rows.append(dict(model=name, **summary))

    challenger_name = select_challenger(cv_results)
    challenger = fitted_estimators[challenger_name]
    base_oof = cross_val_predict(clone(challenger), X_train, y_train, cv=cv, method="predict_proba", n_jobs=int(config["n_jobs"]))[:, 1]
    calibration_candidates = {"uncalibrated": (base_oof, clone(challenger))}
    for method in ("sigmoid", "isotonic"):
        probabilities = calibrated_oof_probabilities(clone(challenger), X_train, y_train, method, cv, inner_cv)
        calibration_candidates[method] = (probabilities, make_calibrated(clone(challenger), method, inner_cv))

    calibration_rows = []
    reliability_frames = []
    for method, (probabilities, _) in calibration_candidates.items():
        metrics, curve = calibration_metrics(y_train, probabilities, int(config["calibration_bins"]))
        calibration_rows.append(dict(model=challenger_name, calibration=method, **metrics))
        curve.insert(0, "calibration", method)
        curve.insert(0, "model", challenger_name)
        reliability_frames.append(curve)
    best_calibration = min(calibration_rows, key=lambda row: row["brier_score"])["calibration"]
    selected_oof, final_estimator = calibration_candidates[best_calibration]

    threshold_frame = analyze_thresholds(
        y_train, selected_oof,
        cost_false_negative=float(config["cost_false_negative"]),
        cost_false_positive=float(config["cost_false_positive"]),
    )
    thresholds = select_thresholds(threshold_frame)
    selected_threshold = thresholds["balanced"]

    holdout_results = {}
    holdout_rows = []
    roc_frames = []
    pr_frames = []
    classification_rows = []
    for name, estimator in fitted_estimators.items():
        if name != "ProductionV2":
            estimator.fit(X_train, y_train)
        metrics, report, roc_data, pr_data, _ = evaluate_holdout(estimator, X_test, y_test, 0.5)
        holdout_results[name] = metrics
        holdout_rows.append(dict(model=name, threshold=0.5, **metrics))
        roc_data.insert(0, "model", name)
        pr_data.insert(0, "model", name)
        roc_frames.append(roc_data)
        pr_frames.append(pr_data)
        for label in ("Stay", "Churn"):
            classification_rows.append(dict(model=name, label=label, **report[label]))

    final_estimator.fit(X_train, y_train)
    final_metrics, final_report, final_roc, final_pr, final_probabilities = evaluate_holdout(final_estimator, X_test, y_test, selected_threshold)
    final_name = challenger_name + ("+" + best_calibration if best_calibration != "uncalibrated" else "")
    holdout_results[final_name] = final_metrics
    holdout_rows.append(dict(model=final_name, threshold=selected_threshold, **final_metrics))
    final_roc.insert(0, "model", final_name)
    final_pr.insert(0, "model", final_name)
    roc_frames.append(final_roc)
    pr_frames.append(final_pr)
    for label in ("Stay", "Churn"):
        classification_rows.append(dict(model=final_name, label=label, **final_report[label]))

    decision, reason = promotion_decision(
        cv_results["ProductionV2"], cv_results[challenger_name],
        holdout_results["ProductionV2"], final_metrics,
    )
    version = config["challenger_version"] if decision.startswith(("B", "C")) else None
    metadata = None
    if version and save_challenger:
        metadata = {
            "name": "churn", "version": version,
            "algorithm": type(final_estimator).__name__, "base_algorithm": challenger_name,
            "target": TARGET_COLUMN, "raw_features": list(RAW_FEATURES),
            "model_features": list(MODEL_FEATURES), "ignored_http_features": list(IGNORED_HTTP_FEATURES),
            "trained_at": datetime.now(timezone.utc).isoformat(), "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)), "random_state": random_state,
            "cv": {"folds": folds, "primary_metric": config["primary_metric"], "results": cv_results[challenger_name]},
            "metrics": final_metrics, "calibration": {"method": best_calibration, "oof_results": calibration_rows},
            "thresholds": thresholds, "threshold": selected_threshold,
            "selected_threshold": selected_threshold, "high_confidence_threshold": 0.8,
            "hyperparameters": best_parameters[challenger_name], "decision": decision,
        }
        ModelRegistry(version).save_pipeline(final_estimator, _json_safe(metadata))

    comparison = pd.DataFrame(comparison_rows)
    holdout = pd.DataFrame(holdout_rows)
    comparison = comparison.merge(holdout.add_prefix("holdout_").rename(columns={"holdout_model": "model"}), on="model", how="left")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    pd.concat(cv_frames, ignore_index=True).to_csv(output_dir / "cross_validation_results.csv", index=False)
    holdout.to_csv(output_dir / "holdout_results.csv", index=False)
    threshold_frame.assign(model=final_name, calibration=best_calibration).to_csv(output_dir / "threshold_analysis.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_results.csv", index=False)
    pd.concat(reliability_frames, ignore_index=True).to_csv(output_dir / "reliability_curves.csv", index=False)
    pd.concat(roc_frames, ignore_index=True).to_csv(output_dir / "roc_curves.csv", index=False)
    pd.concat(pr_frames, ignore_index=True).to_csv(output_dir / "precision_recall_curves.csv", index=False)
    pd.DataFrame(classification_rows).to_csv(output_dir / "classification_reports.csv", index=False)
    pd.DataFrame(search_rows).to_csv(output_dir / "hyperparameter_search.csv", index=False)
    experiment = {
        "dataset": "data/raw/Customer-Churn-Records.csv", "dataset_rows": len(data),
        "train_rows": len(X_train), "holdout_rows": len(X_test), "random_state": random_state,
        "test_size": config["test_size"], "cv_folds": folds, "primary_metric": config["primary_metric"],
        "optional_candidates": [item.__dict__ for item in optional_candidate_statuses()],
        "best_parameters": best_parameters, "challenger": challenger_name,
        "calibration": best_calibration, "thresholds": thresholds,
        "decision": decision, "decision_reason": reason, "v3_created": bool(version and save_challenger),
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(_json_safe(experiment), indent=2), encoding="utf-8")
    return {"decision": decision, "reason": reason, "challenger": challenger_name, "final_name": final_name, "thresholds": thresholds, "cv_results": cv_results, "holdout_results": holdout_results, "calibration_results": calibration_rows, "metadata": metadata, "output_dir": str(output_dir)}
