"""CLI entrypoint for the leakage-safe v4 feature expansion experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.feature_expansion_pipeline import run_feature_expansion


if __name__ == "__main__":
    result = run_feature_expansion()
    print("Decision: {}".format(result["decision"]))
    print("Selected feature group: {}".format(result["metadata"]["feature_group"]))
    print("V4 metrics: {}".format(result["metrics"]))
