import mlflow
import mlflow.sklearn

from ml_pipeline.config_loader import load_config
from ml_pipeline.ingest import load_data, prepare_data, split_data
from ml_pipeline.train import train_model
from ml_pipeline.evaluate import evaluate_model
from ml_pipeline.save import save_model, save_results


def run_pipeline(config_path):
    """Execute the full ML pipeline from a JSON config file.

    Loads data, trains a model, evaluates it, and logs everything to MLflow.

    Args:
        config_path: Path to a JSON config file. Expected top-level keys:
            - run_name (str): Label for the MLflow run.
            - mlflow.tracking_uri (str): MLflow tracking server URI.
            - mlflow.experiment_name (str): MLflow experiment to log under.
            - data.raw_data_path (str): Path to the CSV dataset.
            - data.target_column (str): Column name to predict.
            - data.exclude_columns (list[str]): Columns to drop before training.
            - data.test_size (float): Fraction of data held out for evaluation.
            - data.random_state (int): Seed for reproducibility.
            - model.type (str): Model key from MODEL_REGISTRY (e.g. "xgboost").
            - model.task (str): "classification" or "regression".
            - model.hyperparameters (dict): Passed directly to the model constructor.
            - preprocessing.scaler (str, optional): Scaler key from SCALER_REGISTRY.
              Defaults to "none" (no scaling).
            - preprocessing.scaler_params (dict, optional): kwargs for the scaler.
            - evaluation.metrics (list[str]): Metric keys from METRIC_REGISTRY.

    Returns:
        tuple[Pipeline, dict]: Fitted sklearn Pipeline and dict of metric results.
    """
    config = load_config(config_path)
    print(f"Starting run: {config['run_name']}")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    print("Tracking URI:", mlflow.get_tracking_uri())

    experiment = mlflow.set_experiment(config["mlflow"]["experiment_name"])
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
            scaler_type=preproc_cfg.get("scaler", "none"),
            scaler_params=preproc_cfg.get("scaler_params", None),
        )
        print(f"Model trained: {config['model']['type']}")

        # Evaluate
        results = evaluate_model(model, X_test, y_test, config["evaluation"]["metrics"])
        print(f"Evaluation results: {results}")

        # Log metrics and artifacts
        for metric_name, value in results.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_dict(results, "results.json")

        output_cfg = config.get("output", {})
        if output_cfg.get("model_path"):
            save_model(model, output_cfg["model_path"])
        if output_cfg.get("results_path"):
            save_results(results, output_cfg["results_path"])

    return model, results


if __name__ == "__main__":
    import sys
    run_pipeline(sys.argv[1])
