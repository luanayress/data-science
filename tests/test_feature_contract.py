import pandas as pd
from src.pipelines.inference_pipeline import InferencePipeline

def test_inference_feature_alignment():
    pipeline = InferencePipeline(version="v2")

    X = pd.DataFrame([
        {
            "Age": 45,
            "Tenure": 24,
            "NumOfProducts": 2
        }
    ])

    df = pipeline.preprocess_data(X)
    assert list(df.columns) == pipeline.feature_names
