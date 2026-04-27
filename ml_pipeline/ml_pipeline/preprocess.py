# preprocess.py
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_pipeline(model):
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    return pipeline