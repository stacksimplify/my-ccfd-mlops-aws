#!/usr/bin/env bash
# m06_04_dvc_s3_remote.sh: MLOps 06 Step-04, Option 2. Add the S3 DVC remote to your lab repo and make it the default.
#
#   bash scripts/m06_04_dvc_s3_remote.sh          the lab repo of MLOps 03: ~/mlops-lab/ccfd-project
#
# It reads the dvc bucket name from .env.mlops (ccfd-mlops-dvc-<MLOPS_SUFFIX>; create-env.sh writes .env.mlops) and runs the same 2 commands as
# the README's Option 1: dvc remote add -d -f s3remote s3://<bucket>/dvc, and git commit .dvc/config. The remote
# localremote of MLOps 03 stays. No key goes into .dvc/config: DVC uses your AWS CLI credentials.
# It does not push: you run dvc push yourself (README Step-04). Run it on your EC2 VM: the lab repo is there.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAB="$HOME/mlops-lab/ccfd-project"

[[ -f "$ROOT/.env.mlops" ]] || { echo "STOP: no .env.mlops next to scripts/. Make it first (README Step-03)."; exit 1; }
SUFFIX="$(sed -n 's/^export MLOPS_SUFFIX="\(.*\)"$/\1/p' "$ROOT/.env.mlops")"
[[ -n "$SUFFIX" ]] || { echo "STOP: .env.mlops has no MLOPS_SUFFIX. Run: bash scripts/create-env.sh --write-env-only"; exit 1; }
BUCKET="ccfd-mlops-dvc-$SUFFIX"
if [[ ! -d "$LAB/.dvc" ]]; then
  echo "STOP: $LAB is not the MLOps 03 lab repo (no .dvc folder)."
  on_course_vm || echo "The lab repo lives on your EC2 VM: run this there (Section 00-B Step-01-10 connects)."
  exit 1
fi

cd "$LAB"
dvc remote add -d -f s3remote "s3://$BUCKET/dvc"
git add .dvc/config
if git diff --cached --quiet -- .dvc/config; then
  echo "(.dvc/config already committed)"
else
  git commit -q -m "dvc: s3remote = s3://$BUCKET/dvc (default remote)"
fi
dvc remote list
