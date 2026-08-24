"""Thin entrypoint for controlled v2/v3 promotion evaluation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.promotion_pipeline import run_promotion_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/churn_business.yaml")
    args = parser.parse_args()
    result = run_promotion_evaluation(args.config)
    print(json.dumps({
        "status": result["decision"]["status"],
        "classification": result["decision"]["classification"],
        "uplift": result["uplift"],
        "agreement": result["agreement"],
        "output_dir": result["output_dir"],
    }, indent=2))


if __name__ == "__main__":
    main()
