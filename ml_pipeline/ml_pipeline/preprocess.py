# preprocess.py
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}

def build_pipeline(model, scaler_type="standard", scaler_params=None):
    scaler_class = SCALER_REGISTRY.get(scaler_type)
    steps = []
    if scaler_class is not None:
        steps.append(("scaler", scaler_class(**(scaler_params or {}))))
    steps.append(("model", model))
    return Pipeline(steps=steps)