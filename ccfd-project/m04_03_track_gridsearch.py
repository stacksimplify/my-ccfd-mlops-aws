"""
MLOps 04 Step-03: the base project's grid search result as the parent run 'p5_02-gridsearch', with the winning config
artifacts/p5_02_best_config.yaml logged on it.

    python m04_03_track_gridsearch.py [--dvc-repo ~/mlops-lab/ccfd-project] [--data-tag data-v1]

Copies the base project's frozen grid-search winner config/p5_02_best_config.yaml to artifacts/, trains it with the
base project's p5_03 (unchanged) and logs that model, its Net Benefit and the YAML (seconds).
The 300-fit grid search is not re-run: its winner can change with the machine when 2 settings score almost the same, so
every machine logs the same $50,550 settings.
"""

import shutil
from pathlib import Path

import mlflow

import m04_tracking as track

FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"
FROZEN_CONFIG = "config/p5_02_best_config.yaml"
CONFIG = "artifacts/p5_02_best_config.yaml"

args = track.arguments("Track the base project's p5_02 grid search result in MLflow.")
track.connect()
tags = track.data_tags("p5_02_class_weight_gridsearch.py", "gridsearch", args)

Path(CONFIG).parent.mkdir(exist_ok=True)
shutil.copyfile(FROZEN_CONFIG, CONFIG)
print(f"Settings: the frozen {FROZEN_CONFIG} (the base project's grid search winner; no re-run)")
base = track.run_base("p5_03_train_production_model.py")
run_id = track.track("p5_02-gridsearch", base["pipeline"], base["metrics"]["business_metrics"]["net_benefit"],
                     {**tags, "config_source": f"frozen {FROZEN_CONFIG}"}, train_csv=FE_FILE,
                     params=base["best_params"])

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(CONFIG)
print(f"Logged {CONFIG} on run {run_id}")
