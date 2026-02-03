import pandas as pd
from src.pipelines.inference_pipeline import InferencePipeline

def test_inference_feature_alignment():
    pipeline = InferencePipeline(version="v1")

    X = pd.DataFrame([
        {
            "Age": 45,
            "Tenure": 24,
            "NumOfProducts": 2
        }
    ])

    df = pipeline.build_features(X)
    assert set(pipeline.feature_names).issubset(df.columns)
