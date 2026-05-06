import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml_pipeline.preprocess import build_pipeline, SCALER_REGISTRY
from sklearn.dummy import DummyClassifier


def _dummy_model():
    return DummyClassifier()


def test_build_pipeline_no_scaler():
    pipeline = build_pipeline(_dummy_model(), scaler_type="none")
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0][0] == "model"


def test_build_pipeline_with_standard_scaler():
    pipeline = build_pipeline(_dummy_model(), scaler_type="standard")
    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_build_pipeline_with_minmax_scaler():
    pipeline = build_pipeline(_dummy_model(), scaler_type="minmax")
    assert isinstance(pipeline.named_steps["scaler"], MinMaxScaler)


def test_build_pipeline_unknown_scaler_raises():
    with pytest.raises(KeyError):
        build_pipeline(_dummy_model(), scaler_type="unknown")


def test_build_pipeline_returns_pipeline_instance():
    result = build_pipeline(_dummy_model(), scaler_type="none")
    assert isinstance(result, Pipeline)
