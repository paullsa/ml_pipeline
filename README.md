# ml_pipeline

Config-driven ML pipeline with MLflow experiment tracking. Define your data, model, preprocessing, and metrics in a JSON config — the pipeline handles the rest.

## Features

- Registry-based model selection (XGBoost, Random Forest — easily extensible)
- Pluggable scalers (standard, minmax, robust, or none)
- Pluggable evaluation metrics (accuracy, f1, and any sklearn metric)
- Full MLflow logging: params, metrics, artifacts, and model registry
- Sklearn `Pipeline` objects for reproducible train/predict behaviour
- Outputs: serialized model (`.joblib`) + results (`.json`)

## Project Structure

```
ml_pipeline/
├── ml_pipeline/          # Package source
│   ├── config_loader.py  # JSON config loading
│   ├── ingest.py         # Data loading & train/test splitting
│   ├── preprocess.py     # Scaler registry & sklearn Pipeline builder
│   ├── train.py          # Model registry & training
│   ├── evaluate.py       # Metric registry & evaluation
│   ├── save.py           # Model & results persistence
│   └── pipeline.py       # Main orchestrator (run_pipeline)
├── configs/              # JSON run configs
│   ├── example_run.json  # Iris + XGBoost baseline
│   └── iris_rf_v1.json   # Iris + Random Forest variant
├── data/
│   └── raw/iris.csv      # Example dataset
├── outputs/              # Auto-created: model.joblib + results.json per run
├── tests/
│   └── test_train.py     # Unit tests
└── pyproject.toml
```

## Installation

```powershell
# Clone and install in editable mode
git clone <repo-url>
cd ml_pipeline
pip install -e .
```

**Dependencies** (add to `pyproject.toml` or install manually):

```
pandas
scikit-learn
xgboost
mlflow
joblib
```

## Quick Start

```python
from ml_pipeline.pipeline import run_pipeline

model, results = run_pipeline("configs/example_run.json")
print(results)  # {"accuracy": 1.0, "f1": 1.0}
```

Or from the command line:

```powershell
python -m ml_pipeline.pipeline configs/example_run.json
```

## Config Reference

All behaviour is controlled by a single JSON file:

```json
{
  "run_name": "iris_baseline_v1",
  "data": {
    "raw_data_path": "data/raw/iris.csv",
    "target_column": "species",
    "exclude_columns": [],
    "test_size": 0.2,
    "random_state": 42
  },
  "model": {
    "type": "xgboost",
    "task": "classification",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 3,
      "learning_rate": 0.1
    }
  },
  "preprocessing": {
    "scaler": "standard"
  },
  "evaluation": {
    "metrics": ["accuracy", "f1"]
  },
  "output": {
    "model_path": "outputs/iris_baseline_v1/model.joblib",
    "results_path": "outputs/iris_baseline_v1/results.json"
  },
  "mlflow": {
    "experiment_name": "Iris_Experiment",
    "tracking_uri": "file:///C:/Python/ml_pipeline/mlruns"
  }
}
```

### Supported values

| Key | Options |
|-----|---------|
| `model.type` | `xgboost`, `random_forest` |
| `model.task` | `classification` |
| `preprocessing.scaler` | `standard`, `minmax`, `robust`, `none` |
| `evaluation.metrics` | `accuracy`, `f1` (any sklearn metric in the registry) |

## Pipeline Execution Flow

```
run_pipeline(config_path)
  ├─ load_config()          → dict
  ├─ mlflow.start_run()
  │  ├─ load_data()         → DataFrame
  │  ├─ prepare_data()      → X, y
  │  ├─ split_data()        → train/test splits
  │  ├─ train_model()
  │  │  ├─ get_model()      → estimator from MODEL_REGISTRY
  │  │  └─ build_pipeline() → sklearn Pipeline([scaler, model])
  │  ├─ evaluate_model()    → metrics dict
  │  ├─ mlflow.log_param/metric/artifact
  │  ├─ save_model()        → model.joblib
  │  └─ save_results()      → results.json
  └─ return (model, results)
```

## Extending the Pipeline

**Add a model:**

```python
# train.py — MODEL_REGISTRY
from sklearn.svm import SVC

MODEL_REGISTRY = {
    "svm": {"classification": SVC},
    # ...existing entries
}
```

**Add a scaler:**

```python
# preprocess.py — SCALER_REGISTRY
from sklearn.preprocessing import MaxAbsScaler

SCALER_REGISTRY = {
    "maxabs": MaxAbsScaler,
    # ...existing entries
}
```

**Add a metric:**

```python
# evaluate.py — METRIC_REGISTRY
from sklearn.metrics import roc_auc_score

METRIC_REGISTRY = {
    "roc_auc": roc_auc_score,
    # ...existing entries
}
```

## MLflow Tracking

Runs are logged automatically. Launch the UI with:

```powershell
mlflow ui --workers 1
```

Then open `http://127.0.0.1:5000` to browse experiments, compare runs, and download artifacts.

## Running Tests

```powershell
pytest tests/
```

Tests verify that `train_model` returns an sklearn `Pipeline` and that it can generate predictions on held-out data.

## Output Files

Each run produces:

| File | Contents |
|------|---------|
| `outputs/{run_name}/model.joblib` | Serialized sklearn Pipeline |
| `outputs/{run_name}/results.json` | Evaluation metrics dict |
