import json

import pandas as pd
import pytest

from app.frontend.services.feature_expansion_service import (
    FeatureExpansionDataProvider, FeatureExpansionReportError,
)
from src.data.load_data import load_customer_churn_data
from src.features.bank_feature_contract import EXCLUDED_FEATURES, EXPANDED_FEATURES
from src.pipelines.feature_expansion_pipeline import create_expanded_pipeline, prepare_bank_data
from src.pipelines.inference_pipeline import InferencePipeline
from src.models.readiness import fairness_report, paired_stratified_bootstrap
from app.services.shadow_service import ShadowPredictionService


def test_expanded_contract_excludes_identifiers_target_and_complaint():
    assert not set(EXCLUDED_FEATURES).intersection(EXPANDED_FEATURES)
    assert {"CreditScore", "Balance", "Geography", "Gender"}.issubset(EXPANDED_FEATURES)


def test_expanded_pipeline_handles_unknown_categories():
    data = prepare_bank_data(load_customer_churn_data()).head(200)
    pipeline = create_expanded_pipeline(EXPANDED_FEATURES)
    pipeline.fit(data[list(EXPANDED_FEATURES)], data["Exited"])
    sample = data.loc[[data.index[0]], list(EXPANDED_FEATURES)].copy()
    sample.loc[:, "Geography"] = "New market"
    assert 0 <= pipeline.predict_proba(sample)[0, 1] <= 1


def test_persisted_v4_accepts_bank_payload():
    pipeline = InferencePipeline("v4")
    frame = pd.DataFrame([{
        "CreditScore": 650, "Age": 45, "Tenure": 5, "Balance": 100000.0,
        "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
        "EstimatedSalary": 75000.0, "Geography": "France", "Gender": "Female",
    }])
    result = pipeline.predict_with_confidence(frame)
    assert len(result["probabilities"]) == 1
    assert 0 <= result["probabilities"][0] <= 1
    assert pipeline.threshold == pytest.approx(0.33)


def test_persisted_v4_rejects_incomplete_input_and_shadow_skips_it():
    challenger = InferencePipeline("v4")
    incomplete = pd.DataFrame([{"Age": 45, "Tenure": 5, "NumOfProducts": 2}])
    with pytest.raises(ValueError, match="Incomplete v4 input"):
        challenger.predict_with_confidence(incomplete)

    class Champion:
        version = "v2"

    result = ShadowPredictionService(Champion(), challenger).compare(
        incomplete,
        {"predictions": [0], "probabilities": [0.2], "high_confidence": [True]},
        request_id="test",
    )
    assert result["status"] == "skipped"


def test_feature_expansion_report_provider(tmp_path):
    pd.DataFrame([{"feature_group": "g", "feature_count": 2, "cv_pr_auc_mean": 0.7}]).to_csv(
        tmp_path / "feature_ablation.csv", index=False
    )
    pd.DataFrame([{"model": "v4", "pr_auc": 0.7, "roc_auc": 0.8, "recall": 0.6,
                   "f1": 0.6, "brier_score": 0.1}]).to_csv(tmp_path / "holdout_comparison.csv", index=False)
    pd.DataFrame([{"metric": "pr_auc", "mean_delta": 0.1}]).to_csv(
        tmp_path / "bootstrap_confidence_intervals.csv", index=False
    )
    pd.DataFrame([{"feature": "Gender", "recall_gap": 0.1}]).to_csv(
        tmp_path / "fairness_gaps.csv", index=False
    )
    pd.DataFrame([{"feature": "Age", "importance_mean": 0.2}]).to_csv(
        tmp_path / "permutation_feature_importance.csv", index=False
    )
    (tmp_path / "experiment.json").write_text(json.dumps({"decision": "TEST"}), encoding="utf-8")
    assert FeatureExpansionDataProvider(tmp_path).load().experiment["decision"] == "TEST"


def test_feature_expansion_report_provider_rejects_missing_files(tmp_path):
    with pytest.raises(FeatureExpansionReportError):
        FeatureExpansionDataProvider(tmp_path).load()


def test_paired_bootstrap_finds_clear_improvement():
    y = [0, 0, 0, 1, 1, 1]
    weak = [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]
    strong = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    report = paired_stratified_bootstrap(y, weak, strong, 0.5, 0.5, iterations=50)
    assert report.set_index("metric").loc["pr_auc", "ci_95_lower"] > 0


def test_fairness_report_exposes_group_gaps():
    frame = pd.DataFrame({"Gender": ["F", "F", "M", "M"], "Geography": ["A", "A", "B", "B"]})
    details, gaps = fairness_report(frame, [0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert len(details) == 4
    assert gaps["recall_gap"].max() == 1.0
