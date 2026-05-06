from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}


def build_pipeline(model, scaler_type="none", scaler_params=None):
    """Build an sklearn Pipeline with an optional scaler followed by a model.

    Args:
        model: A fitted-or-unfitted sklearn estimator.
        scaler_type: Key from SCALER_REGISTRY. Use "none" to skip scaling.
        scaler_params: Optional dict of kwargs passed to the scaler constructor.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline ready to fit.

    Raises:
        KeyError: If scaler_type is not in SCALER_REGISTRY.
    """
    scaler_class = SCALER_REGISTRY[scaler_type]
    steps = []
    if scaler_class is not None:
        steps.append(("scaler", scaler_class(**(scaler_params or {}))))
    steps.append(("model", model))
    return Pipeline(steps=steps)
