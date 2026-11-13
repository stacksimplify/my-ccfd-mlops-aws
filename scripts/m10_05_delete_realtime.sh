#!/usr/bin/env bash
# m10_05_delete_realtime.sh: MLOps 10. Delete ccfd-rt: autoscaling first, then the endpoint, every ccfd-rt config and model.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_need_course_env                     # python -m mlops_utils.endpoints needs boto3 and the course packages
cd "$(dirname "$0")/../ccfd-project"
python -m mlops_utils.endpoints delete --endpoint ccfd-rt
