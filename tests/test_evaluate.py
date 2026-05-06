import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.evaluate import evaluate_model
from ml_pipeline.train import train_model


def _make_fitted_pipeline():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [6, 5]])
    y = np.array([0, 1, 0, 1, 0, 1])
    return train_model(X, y, "random_forest", "classification", {"n_estimators": 10, "random_state": 42}), X, y


def test_evaluate_returns_dict():
    model, X, y = _make_fitted_pipeline()
    results = evaluate_model(model, X, y, ["accuracy", "f1"])
    assert set(results.keys()) == {"accuracy", "f1"}


def test_evaluate_accuracy_in_range():
    model, X, y = _make_fitted_pipeline()
    results = evaluate_model(model, X, y, ["accuracy"])
    assert 0.0 <= results["accuracy"] <= 1.0


def test_evaluate_roc_auc_uses_proba():
    model, X, y = _make_fitted_pipeline()
    results = evaluate_model(model, X, y, ["roc_auc"])
    assert 0.0 <= results["roc_auc"] <= 1.0


def test_evaluate_unknown_metric_raises():
    model, X, y = _make_fitted_pipeline()
    with pytest.raises(KeyError):
        evaluate_model(model, X, y, ["not_a_metric"])
