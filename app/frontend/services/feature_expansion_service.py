"""Read-only provider for the persisted v4 feature expansion experiment."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class FeatureExpansionReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureExpansionReport:
    ablation: pd.DataFrame
    holdout: pd.DataFrame
    bootstrap: pd.DataFrame
    fairness_gaps: pd.DataFrame
    feature_importance: pd.DataFrame
    experiment: Dict[str, Any]


class FeatureExpansionDataProvider:
    def __init__(self, report_dir=None):
        project_root = Path(__file__).resolve().parents[3]
        self.report_dir = Path(report_dir) if report_dir else project_root / "reports" / "feature-expansion"

    def load(self) -> FeatureExpansionReport:
        try:
            ablation = pd.read_csv(self.report_dir / "feature_ablation.csv")
            holdout = pd.read_csv(self.report_dir / "holdout_comparison.csv")
            bootstrap = pd.read_csv(self.report_dir / "bootstrap_confidence_intervals.csv")
            fairness_gaps = pd.read_csv(self.report_dir / "fairness_gaps.csv")
            feature_importance = pd.read_csv(self.report_dir / "permutation_feature_importance.csv")
            experiment = json.loads((self.report_dir / "experiment.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise FeatureExpansionReportError("Could not load v4 reports from {}".format(self.report_dir)) from exc
        if not {"feature_group", "feature_count", "cv_pr_auc_mean"}.issubset(ablation.columns):
            raise FeatureExpansionReportError("Invalid feature ablation report")
        if not {"model", "pr_auc", "roc_auc", "recall", "f1", "brier_score"}.issubset(holdout.columns):
            raise FeatureExpansionReportError("Invalid holdout comparison report")
        return FeatureExpansionReport(ablation, holdout, bootstrap, fairness_gaps, feature_importance, experiment)
