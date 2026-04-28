# orchestrates the full pipeline

# pipeline.py

from pyexpat import model

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import os

from ml_pipeline.config_loader import load_config
from ml_pipeline.ingest import load_data, prepare_data, split_data
from ml_pipeline.train import train_model
from ml_pipeline.evaluate import evaluate_model
from ml_pipeline.save import save_model, save_results

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
        model = train_model(X_train, y_train, config["model"]["type"], config["model"]["task"], config["model"]["hyperparameters"])
        print(f"Model trained: {config['model']['type']}")

        # Evaluate
        results = evaluate_model(model, X_test, y_test, config["evaluation"]["metrics"])
        print(f"Evaluation results: {results}")

        # Log metrics
        for metric_name, value in results.items():
            mlflow.log_metric(metric_name, value)

        # Log model
        if config["model"]["type"] == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Save locally for convenience
        run_name = config["run_name"]
        
        # Create outputs directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join("outputs", run_name)
        save_model(model, os.path.join(output_dir, "model.joblib"))
        save_results(results, os.path.join(output_dir, "results.json"))

        # Log results to MLflow
        mlflow.log_artifact(os.path.join(output_dir, "results.json"))

    return model, results

if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1])