"""
MLOps 04 Step-03-03 (optional): run the base project's FULL grid search on this machine and track it in MLflow, next
to the frozen result of Step-03-02.

    python m04_03_track_gridsearch_full_search.py [--dvc-repo ~/mlops-lab/ccfd-project] [--data-tag data-v1]

It runs the base project's p5_02 unchanged: GridSearchCV over 60 settings x 5 folds = 300 fits with the Net Benefit
scorer (about 4 minutes on the EC2 VM), then logs:
  the parent run 'p5_02-gridsearch-full-search' the winner, its Net Benefit on the 2,000 test rows, its parameters,
                                                the tags best_cv_net_benefit, search_minutes and journey_stage gridsearch-full-search
  60 child runs candidate-01 .. candidate-60    each setting with its cv_net_benefit and rank
  artifacts/p5_02_gridsearch_full_search_config.yaml  the winner this full search found (logged on the parent run)
and prints this winner next to the frozen config/p5_02_best_config.yaml ($50,550).

It changes nothing the course uses: its own journey_stage tag (Step-04 reads only journey_stage=gridsearch), and the
frozen artifacts/p5_02_best_config.yaml that the base p5_02 overwrites is put back when the run ends.
"""

import shutil
import time
from pathlib import Path

import mlflow
import yaml

import m04_tracking as track

FE_FILE = "data/credit_card_transactions_fe1_amount_ratio.csv"
FROZEN_CONFIG = "config/p5_02_best_config.yaml"
BASE_ARTIFACT = "artifacts/p5_02_best_config.yaml"            # the base p5_02 writes its winner here
SEARCH_ARTIFACT = "artifacts/p5_02_gridsearch_full_search_config.yaml"
RUN_NAME = "p5_02-gridsearch-full-search"

args = track.arguments("Run the base project's full p5_02 grid search on this machine and track it in MLflow.")
track.connect()
tags = track.data_tags("p5_02_class_weight_gridsearch.py", "gridsearch-full-search", args)

Path(BASE_ARTIFACT).parent.mkdir(exist_ok=True)
saved = Path(BASE_ARTIFACT).read_bytes() if Path(BASE_ARTIFACT).exists() else None
start = time.time()
try:
    base = track.run_base("p5_02_class_weight_gridsearch.py")
    shutil.copyfile(BASE_ARTIFACT, SEARCH_ARTIFACT)
finally:
    # put the course's file back as it was (the frozen copy of Step-03-02), whatever the search did
    if saved is None:
        Path(BASE_ARTIFACT).unlink(missing_ok=True)
    else:
        Path(BASE_ARTIFACT).write_bytes(saved)
minutes = (time.time() - start) / 60

search = base["grid_search"]
run_id = track.track(RUN_NAME, base["best_pipeline"], base["metrics"]["business_metrics"]["net_benefit"],
                     {**tags, "best_cv_net_benefit": f"{search.best_score_:.0f}", "search_minutes": f"{minutes:.1f}"},
                     train_csv=FE_FILE, params=base["best_params_clean"])
children = track.log_children(run_id, search)
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(SEARCH_ARTIFACT)

found = yaml.safe_load(Path(SEARCH_ARTIFACT).read_text())
frozen = yaml.safe_load(Path(FROZEN_CONFIG).read_text())
print(f"Child runs: {children} (best CV Net Benefit ${search.best_score_:,.0f}); the search took {minutes:.1f} minutes")
print(f"Logged {SEARCH_ARTIFACT} on run {run_id}")
print("This full search vs the frozen config of the course:")
print(f"  full search : {found['best_params']}")
print(f"                test Net Benefit ${found['test_net_benefit']:,}")
print(f"  frozen      : {frozen['best_params']}")
print(f"                test Net Benefit ${frozen['test_net_benefit']:,}")
same = found["best_params"] == frozen["best_params"]
print("  SAME winner as the frozen config" if same else "  DIFFERENT winner from the frozen config (see the child runs' ranks)")
