# orchestrates the full pipeline


# pipeline.py
import mlflow
import mlflow.xgboost

from ml_pipeline.config_loader import load_config
from ml_pipeline.ingest import load_data, prepare_data, split_data
from ml_pipeline.train import train_model
from ml_pipeline.evaluate import evaluate_model

def run_pipeline(config_path):
    config = load_config(config_path)
    print(f"Starting run: {config['run_name']}")

    mlflow.set_experiment(config["data"]["raw_data_path"].split("/")[-1].replace(".csv", ""))

    with mlflow.start_run(run_name=config["run_name"]):

        # Log parameters
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("task", config["model"]["task"])
        mlflow.log_param("test_size", config["data"]["test_size"])
        for param, value in config["model"]["hyperparameters"].items():
            mlflow.log_param(param, value)

        # Ingest
        df = load_data(config["data"]["raw_data_path"])
        X, y = prepare_data(df, config["data"]["target_column"], config["data"]["exclude_columns"])
        X_train, X_test, y_train, y_test = split_data(X, y, config["data"]["test_size"], config["data"]["random_state"])
        print(f"Data loaded: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")

        # Train
        model = train_model(X_train, y_train, config["model"]["type"], config["model"]["task"], config["model"]["hyperparameters"])
        print(f"Model trained: {config['model']['type']}")

        # Evaluate
        results = evaluate_model(model, X_test, y_test, config["evaluation"]["metrics"])
        print(f"Evaluation results: {results}")

        # Log metrics
        for metric_name, value in results.items():
            mlflow.log_metric(metric_name, value)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

    return model, results