"""
MLOps 04 Step-04: the base project's tuned model (p5_03, best config) as the run 'p5_03-production', trained from the config of the
newest 'p5_02-gridsearch' run in MLflow (not from a file someone left on disk).

    python m04_04_track_production.py [--dvc-repo ~/mlops-lab/ccfd-project] [--data-tag data-v1]

It downloads p5_02_best_config.yaml from that run into artifacts/, runs the base project's p5_03 unchanged, and logs the model
with the tag config_source_run_id = the grid search run (lineage: which search chose these settings).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import mlflow

import m04_tracking as track
from mlops_utils import mlflow_layer

FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"
CONFIG = "artifacts/p5_02_best_config.yaml"

args = track.arguments("Track the base project's p5_03 tuned model in MLflow.")
track.connect()
source = mlflow_layer.latest_run("journey_stage", "gridsearch")
if source is None:
    sys.exit("STOP: no p5_02-gridsearch run in MLflow. Run m04_03_track_gridsearch.py first.")
with tempfile.TemporaryDirectory() as folder:
    path = mlflow.artifacts.download_artifacts(run_id=source.info.run_id, artifact_path=Path(CONFIG).name,
                                               dst_path=folder)
    Path(CONFIG).parent.mkdir(exist_ok=True)
    shutil.copyfile(path, CONFIG)
print(f"Config: {CONFIG} from run {source.info.run_id} (p5_02-gridsearch)")
tags = track.data_tags("p5_03_train_production_model.py", "production", args)
base = track.run_base("p5_03_train_production_model.py")
run_id = track.track("p5_03-production", base["pipeline"], base["metrics"]["business_metrics"]["net_benefit"],
                     {**tags, "config_source_run_id": source.info.run_id}, train_csv=FE_FILE,
                     params=base["best_params"])
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(CONFIG, artifact_path="inputs")
