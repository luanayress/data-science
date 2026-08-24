import json

import pandas as pd
import pytest

from app.frontend.services.model_comparison_service import (
    ComparisonReportError,
    ModelComparisonDataProvider,
)


def write_valid_report(path):
    scenarios = pd.DataFrame([{
        "model_version": "v3", "scenario": "v3_balanced", "threshold": 0.34,
        "precision": 0.58, "recall": 0.56, "f1": 0.57, "roc_auc": 0.83,
        "pr_auc": 0.62, "brier_score": 0.116, "log_loss": 0.37,
        "tn": 1431, "fp": 161, "fn": 179, "tp": 229, "campaign_rate": 0.195,
        "churn_coverage": 0.56, "relative_cost": 1056, "predict_time": 0.09,
    }])
    scenarios.to_csv(path / "promotion_comparison.csv", index=False)
    pd.DataFrame([{
        "fn_fp_ratio": 5, "best_threshold": 0.17, "precision": 0.41,
        "recall": 0.74, "cost": 3820, "predicted_positive_rate": 0.36,
    }]).to_csv(path / "cost_sensitivity.csv", index=False)
    (path / "promotion_decision.json").write_text(json.dumps({
        "metrics": {"champion": {"model_version": "v2"}, "challenger": {"model_version": "v3"}}
    }), encoding="utf-8")


def test_comparison_provider_loads_persisted_reports(tmp_path):
    write_valid_report(tmp_path)
    report = ModelComparisonDataProvider(tmp_path).load()
    assert report.scenarios.iloc[0]["scenario"] == "v3_balanced"
    assert report.cost_sensitivity.iloc[0]["best_threshold"] == pytest.approx(0.17)
    assert report.decision["metrics"]["challenger"]["model_version"] == "v3"


def test_comparison_provider_rejects_missing_report(tmp_path):
    with pytest.raises(ComparisonReportError, match="Could not load"):
        ModelComparisonDataProvider(tmp_path).load()


def test_comparison_provider_rejects_invalid_schema(tmp_path):
    write_valid_report(tmp_path)
    pd.DataFrame([{"scenario": "v3_balanced"}]).to_csv(tmp_path / "promotion_comparison.csv", index=False)
    with pytest.raises(ComparisonReportError, match="missing columns"):
        ModelComparisonDataProvider(tmp_path).load()
