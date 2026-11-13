"""
MLOps 05 Step-01: register the first champion.

    python m05_01_register_first_champion.py

The newest MLOps 04 run with journey_stage=fe1 (the base project's p4_03, $39,700, 9 inputs = the serving contract) becomes
version 1 of the registered model ccfd-fraud-model, with tags and a description, and the alias @champion points at
it. Run it again: it prints 'already registered' and changes nothing.
"""

from mlflow import MlflowClient

import m05_common
from mlops_utils import mlflow_layer

m05_common.connect()
run = m05_common.stage_run("fe1")
champion = mlflow_layer.get_alias(mlflow_layer.CHAMPION)
if champion is not None:
    print(f"Already registered: @champion -> version {champion.version}. Nothing changed.")
    raise SystemExit(0)

metrics, tags = run.data.metrics, run.data.tags
version = mlflow_layer.register(run.info.run_id, {
    "net_benefit": int(metrics["net_benefit"]), "f1": round(metrics["f1"], 4), "data_md5": tags.get("data_md5"),
    "git_sha": tags.get("git_sha"), "journey_stage": "fe1", "gate_result": "FIRST_CHAMPION"})
MlflowClient().update_model_version(mlflow_layer.MODEL_NAME, version.version,
                                    description="The base project's FE1 model (p4_03): the first champion, 9 inputs incl. amount_ratio")
mlflow_layer.set_alias(mlflow_layer.CHAMPION, version.version)
print(f"Registered model {mlflow_layer.MODEL_NAME} version {version.version} "
      f"(source run {run.info.run_name}, Net Benefit ${int(metrics['net_benefit']):,})")
print(f"Alias {mlflow_layer.CHAMPION} -> version {version.version}")
