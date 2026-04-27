# test_train.py

import numpy as np
from sklearn.pipeline import Pipeline
from ml_pipeline.train import train_model

def test_train_model_returns_pipeline():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])

    result = train_model(X_train, y_train, "random_forest", "classification", {"n_estimators": 10, "random_state": 42})

    assert isinstance(result, Pipeline)

def test_train_model_can_predict():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[2, 3], [6, 7]])

    pipeline = train_model(X_train, y_train, "random_forest", "classification", {"n_estimators": 10, "random_state": 42})
    predictions = pipeline.predict(X_test)

    assert predictions.shape == (2,)