#!/usr/bin/env bash
# m07_03_rerun_on_app.sh: MLOps 07 Step-03, Option 2. Replay the unchanged MLOps 04 and 05 scripts against the
# SageMaker AI MLflow App: the only change is MLFLOW_TRACKING_URI (the App ARN).
#
#   source .env.mlops && export MLFLOW_TRACKING_URI="$MLOPS_MLFLOW_APP_ARN"
#   bash scripts/m07_03_rerun_on_app.sh          data v1 from the lab repo of MLOps 03: ~/mlops-lab/ccfd-project
#
# Order as in MLOps 04 and 05: data v1 from your lab repo (dvc get, tag data-v1), the FE1 file, baseline, FE1,
# random search, grid search, production, the journey report, the first champion, the gate in the order MLOps 04
# made the models (random search: REJECT, then the tuned model, stage production: PROMOTE).
# Every line goes through the account ID mask.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_need_course_env
[[ "${MLFLOW_TRACKING_URI:-}" == arn:* ]] || mlops_stop "set MLFLOW_TRACKING_URI to the App ARN first: source .env.mlops && export MLFLOW_TRACKING_URI=\"\$MLOPS_MLFLOW_APP_ARN\""
LAB="$HOME/mlops-lab/ccfd-project"               # the lab repo of MLOps 03 (the m04 scripts read it there too)
mlops_need dvc
cd "$(dirname "$0")/../ccfd-project"
export PYTHONWARNINGS=ignore   # MLflow's schema and dataset-source hints are noise here
{
  mkdir -p data
  # Data v1 from the lab repo, the way a teammate gets it (--force: a re-run replaces the file; the md5 check of
  # every m04 script proves it is data v1)
  dvc get "$LAB" "$MLOPS_DATA_FILE" --rev data-v1 -o "$MLOPS_DATA_FILE" --force
  python p4_01_feature_engineering_fe1.py >/dev/null
  for script in m04_02_track_baseline m04_02_track_fe1 m04_03_track_randomsearch m04_03_track_gridsearch \
                m04_04_track_production; do
    python "$script.py" | grep -E "^Run |^Child runs|^Config|STOP"
  done
  python m04_04_journey_report.py
  python m05_01_register_first_champion.py | grep -E "^Registered|^Alias|^Already"
  python m05_02_net_benefit_gate.py --candidate-stage randomsearch | grep -E "^Gate|^@"
  python m05_02_net_benefit_gate.py --candidate-stage production | grep -E "^Gate|^@"
} 2>&1 | grep -vE "WARNING|INFO mlflow|Created version|already exists|Successfully registered" | sed -E "$MLOPS_MASK_SED"
