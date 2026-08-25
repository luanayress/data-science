import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.inference_pipeline import InferencePipeline
from src.pipelines.training_pipeline import run_training_pipeline


CUSTOMER_A = {"Age": 30, "Tenure": 5, "NumOfProducts": 2}
CUSTOMER_B = {"Age": 60, "Tenure": 40, "NumOfProducts": 2}
CUSTOMER_C = {"Age": 45, "Tenure": 12, "NumOfProducts": 1}


@pytest.fixture(scope="module")
def trained_result():
    return run_training_pipeline(save_model=False)


def test_age_and_tenure_are_not_collapsed():
    inference = InferencePipeline("v2")
    transformed = inference.preprocess_data(pd.DataFrame([CUSTOMER_A, CUSTOMER_B]))
    assert not np.allclose(transformed.iloc[0], transformed.iloc[1])


def test_single_prediction_equals_batch_prediction():
    inference = InferencePipeline("v2")
    single = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A]))
    batch = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A, CUSTOMER_B]))
    assert single["probabilities"][0] == pytest.approx(batch["probabilities"][0])


def test_prediction_is_invariant_to_batch_composition():
    inference = InferencePipeline("v2")
    first = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A, CUSTOMER_B]))
    second = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A, CUSTOMER_C]))
    assert first["probabilities"][0] == pytest.approx(second["probabilities"][0])


def test_serialized_pipeline_matches_in_memory_pipeline(tmp_path, trained_result):
    pipeline = trained_result["pipeline"]
    sample = trained_result["X_test"].head(10)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    assert np.allclose(pipeline.predict_proba(sample), loaded.predict_proba(sample))
    assert np.allclose(pipeline[:-1].transform(sample), loaded[:-1].transform(sample))


def test_inference_does_not_fit_transformers(monkeypatch):
    inference = InferencePipeline("v2")

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("fit must not run during inference")

    for step in inference.pipeline.named_steps.values():
        monkeypatch.setattr(step, "fit", forbidden_fit)
    result = inference.predict_with_confidence(pd.DataFrame([CUSTOMER_A]))
    assert len(result["predictions"]) == 1
