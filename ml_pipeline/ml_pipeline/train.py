# model training
# train.py
import xgboost as xgb

MODEL_REGISTRY = {
    "xgboost": {
        "classification": xgb.XGBClassifier,
        "regression": xgb.XGBRegressor
    }
}

def get_model(model_type, task, hyperparameters):
    model_class = MODEL_REGISTRY[model_type][task]
    model = model_class(**hyperparameters)
    return model

def train_model(X_train, y_train, model_type, task, hyperparameters):
    model = get_model(model_type, task, hyperparameters)
    model.fit(X_train, y_train)
    return model