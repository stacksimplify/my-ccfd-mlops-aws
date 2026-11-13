#!/usr/bin/env bash
# m09_04_retrain_on_data_v2.sh: MLOps 09 Step-04-02, Option 2. The MLOps 08 jobs again, on data v2: the processing
# job (data checks + amount_ratio) on the data-v2 DVC object, then the tuned-model training (managed spot) with the same
# frozen settings. Prints the names and the model.tar.gz you register next with m09_01_register_package.py.
# It stops at once (nothing started) when the course environment mlops-aws-env is not active.
#
#   bash scripts/m09_04_retrain_on_data_v2.sh
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
[[ $# -eq 0 ]] || { echo "Usage: bash $0   (no options; it starts 2 SageMaker jobs)"; exit 2; }
mlops_need_course_env
cd "$(dirname "$0")/../ccfd-project"
python m08_02_run_processing_job.py --dvc-rev data-v2
python m08_03_run_training_job.py --dvc-rev data-v2
