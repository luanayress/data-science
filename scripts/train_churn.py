"""Official thin entrypoint for churn model training."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.training_pipeline import run_training_pipeline


def main() -> None:
    result = run_training_pipeline(save_model=True)
    print("Churn pipeline trained and saved successfully.")
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
