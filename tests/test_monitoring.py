import pandas as pd
import numpy as np

from src.monitoring import ModelMonitor


def test_detect_drift_numeric():
    ref = pd.DataFrame({"x": np.random.normal(0, 1, size=1000)})
    cur = pd.DataFrame({"x": np.random.normal(0.5, 1.2, size=1000)})

    monitor = ModelMonitor()
    res = monitor.detect_drift(ref, cur, alpha=0.01)

    assert "x" in res["details"]
    assert "pvalue" in res["details"]["x"]
    assert isinstance(res["details"]["x"]["pvalue"], float) or res["details"]["x"]["pvalue"] is None


def test_detect_drift_categorical():
    ref = pd.DataFrame({"cat": np.random.choice(["a", "b", "c"], size=500, p=[0.6, 0.3, 0.1])})
    cur = pd.DataFrame({"cat": np.random.choice(["a", "b", "c"], size=500, p=[0.2, 0.7, 0.1])})

    monitor = ModelMonitor()
    res = monitor.detect_drift(ref, cur, alpha=0.01)

    assert "cat" in res["details"]
    assert "pvalue" in res["details"]["cat"]

