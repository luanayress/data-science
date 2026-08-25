"""Thin entrypoint for the Phase 2B churn comparison."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.model_comparison_pipeline import run_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_comparison.yaml")
    parser.add_argument("--no-save-v3", action="store_true")
    args = parser.parse_args()
    result = run_model_comparison(args.config, save_challenger=not args.no_save_v3)
    print(json.dumps({key: result[key] for key in ("decision", "reason", "challenger", "final_name", "thresholds", "output_dir")}, indent=2))


if __name__ == "__main__":
    main()
