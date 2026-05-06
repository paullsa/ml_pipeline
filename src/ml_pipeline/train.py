import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from ml_pipeline.preprocess import build_pipeline

MODEL_REGISTRY = {
    "xgboost": {
        "classification": xgb.XGBClassifier,
        "regression": xgb.XGBRegressor,
    },
    "random_forest": {
        "classification": RandomForestClassifier,
        "regression": RandomForestRegressor,
    },
    "svm": {
        "classification": SVC,
        "regression": SVR,
    },
}


def get_model(model_type, task, hyperparameters):
    """Instantiate a model from MODEL_REGISTRY.

    Args:
        model_type: Registry key (e.g. "xgboost", "random_forest", "svm").
        task: "classification" or "regression".
        hyperparameters: dict of kwargs passed to the model constructor.

    Returns:
        An unfitted sklearn-compatible estimator.

    Raises:
        ValueError: If model_type or task is not recognised.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODEL_REGISTRY)}")
    if task not in MODEL_REGISTRY[model_type]:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(MODEL_REGISTRY[model_type])}")
    model_class = MODEL_REGISTRY[model_type][task]
    return model_class(**hyperparameters)


def train_model(X_train, y_train, model_type, task, hyperparameters, scaler_type="none", scaler_params=None):
    """Train a model wrapped in an sklearn Pipeline.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        model_type: Registry key (e.g. "xgboost", "random_forest", "svm").
        task: "classification" or "regression".
        hyperparameters: dict of kwargs passed to the model constructor.
        scaler_type: Scaler key from SCALER_REGISTRY. Defaults to "none".
        scaler_params: Optional dict of kwargs for the scaler constructor.

    Returns:
        sklearn.pipeline.Pipeline: Fitted pipeline (scaler + model).
    """
    model = get_model(model_type, task, hyperparameters)
    pipeline = build_pipeline(model, scaler_type=scaler_type, scaler_params=scaler_params)
    pipeline.fit(X_train, y_train)
    return pipeline
