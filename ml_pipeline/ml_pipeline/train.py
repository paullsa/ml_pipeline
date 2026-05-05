# model training
# train.py

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from ml_pipeline.preprocess import build_pipeline

MODEL_REGISTRY = {
    "xgboost": {
        "classification": xgb.XGBClassifier,
        "regression": xgb.XGBRegressor
    },
    "random_forest": {
        "classification": RandomForestClassifier,
        "regression": RandomForestRegressor
    },
    "svm": {
        "classification": SVC,
        "regression": SVR
    },
}

def get_model(model_type, task, hyperparameters):
    model_class = MODEL_REGISTRY[model_type][task]
    model = model_class(**hyperparameters)
    return model

def train_model(X_train, y_train, model_type, task, hyperparameters, scaler_type="standard", scaler_params=None):
    model = get_model(model_type, task, hyperparameters)
    pipeline = build_pipeline(model, scaler_type=scaler_type, scaler_params=scaler_params)
    pipeline.fit(X_train, y_train)
    return pipeline