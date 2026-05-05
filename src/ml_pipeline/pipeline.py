# orchestrates the full pipeline

# pipeline.py

import mlflow
import mlflow.sklearn

from ml_pipeline.config_loader import load_config
from ml_pipeline.ingest import load_data, prepare_data, split_data
from ml_pipeline.train import train_model
from ml_pipeline.evaluate import evaluate_model

def run_pipeline(config_path):
    config = load_config(config_path)
    print(f"Starting run: {config['run_name']}")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    print("Tracking URI:", mlflow.get_tracking_uri())

    experiment = mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )
    print("Experiment ID:", experiment.experiment_id)
    print("Experiment name:", experiment.name)

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
        preproc_cfg = config.get("preprocessing", {})
        model = train_model(
            X_train, y_train,
            config["model"]["type"],
            config["model"]["task"],
            config["model"]["hyperparameters"],
            scaler_type=preproc_cfg.get("scaler", "standard"),
            scaler_params=preproc_cfg.get("scaler_params", None),
        )
        print(f"Model trained: {config['model']['type']}")

        # Evaluate
        results = evaluate_model(model, X_test, y_test, config["evaluation"]["metrics"])
        print(f"Evaluation results: {results}")

        # Log metrics
        for metric_name, value in results.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_dict(results, "results.json")

    return model, results

if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1])