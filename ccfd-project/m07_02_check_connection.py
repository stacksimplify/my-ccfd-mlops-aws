"""
MLOps 07 Step-02: check the connection to the SageMaker AI MLflow App.

    export MLFLOW_TRACKING_URI=<the App ARN>     (terraform -chdir=infra/02-mlflow-app output -raw mlflow_app_arn)
    python m07_02_check_connection.py

Every call is signed with SigV4 by the sagemaker-mlflow plugin, using your AWS profile: no password, no token; IAM
decides who may use the App. The check logs one run in the experiment mlops-connection-check, reads it back and prints
CONNECTION OK. It deletes nothing, so it can run again and again.
"""

import os
import sys
from importlib.metadata import version

import mlflow
from mlflow import MlflowClient

from mlops_utils import config


def main() -> int:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("STOP: MLFLOW_TRACKING_URI is not set. Run: "
              "export MLFLOW_TRACKING_URI=$(terraform -chdir=infra/02-mlflow-app output -raw mlflow_app_arn)")
        return 1
    print(f"Tracking URI:  {config.mask(uri)}")
    print(f"MLflow client: {version('mlflow')}, sagemaker-mlflow {version('sagemaker-mlflow')}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("mlops-connection-check")
    with mlflow.start_run(run_name="sigv4-check") as run:
        mlflow.log_param("check", "sigv4")
        mlflow.log_metric("ok", 1.0)
        mlflow.set_tag("course", "mlops-on-aws")
    back = MlflowClient().get_run(run.info.run_id).data
    print(f"Run {run.info.run_id} logged; read back: param check={back.params.get('check')}, "
          f"metric ok={back.metrics.get('ok')}")
    if back.params.get("check") != "sigv4" or back.metrics.get("ok") != 1.0:
        print("CONNECTION FAILED: the run did not read back")
        return 1
    print("CONNECTION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
