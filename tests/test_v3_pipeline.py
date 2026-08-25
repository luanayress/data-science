import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.feature_contract import MODEL_FEATURES, RAW_FEATURES
from src.models.registry import ModelRegistry
from src.pipelines.inference_pipeline import InferencePipeline


CUSTOMER_A = {"Age": 30, "Tenure": 5, "NumOfProducts": 2}
CUSTOMER_B = {"Age": 60, "Tenure": 40, "NumOfProducts": 2}


def test_v3_registry_and_metadata_contract():
    pipeline = ModelRegistry("v3").load_pipeline()
    metadata = ModelRegistry("v3").load_metadata()
    assert metadata["version"] == "v3"
    assert metadata["raw_features"] == list(RAW_FEATURES)
    assert metadata["model_features"] == list(MODEL_FEATURES)
    probabilities = pipeline.predict_proba(pd.DataFrame([CUSTOMER_A]))
    assert probabilities.shape == (1, 2)


def test_v3_single_batch_parity_and_invariance():
    inference = InferencePipeline("v3")
    single = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A]))
    first_batch = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A, CUSTOMER_B]))
    second_batch = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A, dict(CUSTOMER_B, Age=45)]))
    assert single["probabilities"][0] == pytest.approx(first_batch["probabilities"][0])
    assert single["probabilities"][0] == pytest.approx(second_batch["probabilities"][0])


def test_v3_serialization_parity(tmp_path):
    pipeline = ModelRegistry("v3").load_pipeline()
    sample = pd.DataFrame([CUSTOMER_A, CUSTOMER_B])
    path = tmp_path / "v3.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    assert np.allclose(pipeline.predict_proba(sample), loaded.predict_proba(sample))
    assert np.array_equal(pipeline.predict(sample), loaded.predict(sample))
