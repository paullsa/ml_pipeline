import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

METRIC_REGISTRY = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mae": mean_absolute_error,
    "r2": r2_score,
}


def evaluate_model(model, X_test, y_test, metrics):
    """Evaluate a fitted pipeline against a held-out test set.

    Args:
        model: Fitted sklearn Pipeline (must implement predict).
        X_test: Test feature matrix.
        y_test: True target values.
        metrics: List of metric keys. Supported keys are in METRIC_REGISTRY
            plus "roc_auc". Example: ["accuracy", "f1", "roc_auc"].

    Returns:
        dict[str, float]: Metric name → score.

    Raises:
        ValueError: If "roc_auc" is requested but the model does not support
            predict_proba (e.g. SVC without probability=True).
        KeyError: If an unrecognised metric key is passed.

    Notes:
        "roc_auc" uses predicted probabilities (predict_proba) with the
        one-vs-rest strategy, so it works for binary and multiclass tasks.
        All other metrics use hard predictions from predict().
    """
    y_pred = model.predict(X_test)
    y_prob = None
    if "roc_auc" in metrics and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    results = {}
    for metric_name in metrics:
        if metric_name == "roc_auc":
            if y_prob is None:
                raise ValueError(
                    f"roc_auc requires predict_proba but {type(model).__name__} does not "
                    "support it. Check the model's hyperparameters (e.g. SVC needs "
                    "probability=True) or remove roc_auc from the requested metrics."
                )
            if y_prob.shape[1] == 2:
                results[metric_name] = roc_auc_score(y_test, y_prob[:, 1])
            else:
                results[metric_name] = roc_auc_score(y_test, y_prob, multi_class="ovr")
        else:
            metric_fn = METRIC_REGISTRY[metric_name]
            results[metric_name] = metric_fn(y_test, y_pred)
    return results
