import joblib
import json
import os


def save_model(model, model_path):
    """Persist a fitted model to disk using joblib.

    Args:
        model: Fitted sklearn Pipeline or estimator.
        model_path: Destination file path (e.g. "outputs/run/model.joblib").
            Parent directories are created automatically.
    """
    parent = os.path.dirname(model_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def save_results(results, results_path):
    """Write a results dict to a JSON file.

    Args:
        results: dict of metric name → score (as returned by evaluate_model).
        results_path: Destination file path (e.g. "outputs/run/results.json").
            Parent directories are created automatically.
    """
    parent = os.path.dirname(results_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
