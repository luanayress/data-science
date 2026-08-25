"""Basic model monitoring utilities.

Provides a small, tested `ModelMonitor` with numeric and categorical drift
checks (KS test for numeric, chi-squared for categorical) and a summary
report. This is intentionally lightweight so it can be used in CI and small
projects without heavy dependencies.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency


class ModelMonitor:
    """Minimal model monitoring utilities."""

    def __init__(self) -> None:
        pass

    def detect_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Detect feature drift between reference and current datasets.

        For numeric features use Kolmogorov-Smirnov test; for categorical
        features use chi-squared test on value counts.

        Returns a dict with detailed per-feature results and a list of
        drifted features.
        """
        if reference is None or current is None:
            raise ValueError("reference and current dataframes are required")

        features = list(set(reference.columns) & set(current.columns))
        results: Dict[str, Any] = {
            "drifted_features": [],
            "details": {}
        }

        for col in features:
            ref_col = reference[col].dropna()
            cur_col = current[col].dropna()

            # numeric
            if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col):
                stat, pvalue = ks_2samp(ref_col, cur_col)
                drift = pvalue < alpha
                results["details"][col] = {
                    "type": "numeric",
                    "pvalue": float(pvalue),
                    "statistic": float(stat),
                    "drift": bool(drift)
                }
                if drift:
                    results["drifted_features"].append(col)

            # categorical
            else:
                ref_counts = ref_col.value_counts()
                cur_counts = cur_col.value_counts()
                categories = list(set(ref_counts.index) | set(cur_counts.index))
                ref_freqs = np.array([ref_counts.get(cat, 0) for cat in categories], dtype=float)
                cur_freqs = np.array([cur_counts.get(cat, 0) for cat in categories], dtype=float)
                # If all zeros in one side, skip
                if ref_freqs.sum() == 0 or cur_freqs.sum() == 0:
                    # cannot run chi2 test reliably
                    results["details"][col] = {
                        "type": "categorical",
                        "pvalue": None,
                        "statistic": None,
                        "drift": False,
                        "note": "Insufficient counts"
                    }
                else:
                    # Build contingency table
                    contingency = np.vstack([ref_freqs, cur_freqs])
                    try:
                        chi2, p, dof, _ = chi2_contingency(contingency)
                        drift = p < alpha
                        results["details"][col] = {
                            "type": "categorical",
                            "pvalue": float(p),
                            "statistic": float(chi2),
                            "drift": bool(drift)
                        }
                        if drift:
                            results["drifted_features"].append(col)
                    except Exception as e:
                        results["details"][col] = {
                            "type": "categorical",
                            "pvalue": None,
                            "statistic": None,
                            "drift": False,
                            "note": f"chi2 failed: {e}"
                        }

        return results
