# model evaluation
# evaluate.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

METRIC_REGISTRY = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class="ovr"),
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mae": mean_absolute_error,
    "r2": r2_score
}

def evaluate_model(model, X_test, y_test, metrics):
    y_pred = model.predict(X_test)
    results = {}
    for metric_name in metrics:
        metric_fn = METRIC_REGISTRY[metric_name]
        results[metric_name] = metric_fn(y_test, y_pred)
    return results