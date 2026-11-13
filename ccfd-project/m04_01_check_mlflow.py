"""
MLOps 04 Step-01: check the connection to the MLflow server.

    export MLFLOW_TRACKING_URI=http://127.0.0.1:7080
    python m04_01_check_mlflow.py

It reads only MLFLOW_TRACKING_URI (exit 1 when it is not set), asks the server for /health, and creates or
selects the experiment ccfd-fraud-detection. MLOps 07 runs it again with the SageMaker MLflow App's ARN.
"""

import os
import sys

import mlflow
from mlflow import MlflowClient

from mlops_utils import config, mlflow_layer


def main() -> int:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("STOP: MLFLOW_TRACKING_URI is not set. Run: export MLFLOW_TRACKING_URI=http://127.0.0.1:7080")
        return 1
    print(f"Tracking URI: {config.mask(uri)}")
    try:
        mlflow_layer.connect(uri)
    except RuntimeError as error:
        print(f"Server: NOT OK ({error})")
        return 1
    print("Server: OK")
    experiment = mlflow.get_experiment_by_name(mlflow_layer.EXPERIMENT)
    runs = MlflowClient().search_runs([experiment.experiment_id], max_results=1000)
    print(f"Experiment: {experiment.name} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
