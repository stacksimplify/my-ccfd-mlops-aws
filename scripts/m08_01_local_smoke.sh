#!/usr/bin/env bash
# m08_01_local_smoke.sh: MLOps 08 Step-01-03, Option 2. Try the training image on your EC2 VM before SageMaker runs it (free).
#
#   bash scripts/m08_01_local_smoke.sh <image>      for example the pushed ccfd-training:sha-xxxxxxx
#
# The same as Option 1: 1. the pinned versions inside the image. 2. Data v1 from your S3 DVC remote (the object the
# processing job reads) into ccfd-project/data/ (Step-02-02 and Step-04-01 use it). 3. The data checks job on it, in
# ~/mlops-smoke, a folder shaped like SageMaker's /opt/ml/processing (input/raw, output/fe, output/report), with a raw
# file named by its md5 and no extension, as S3 DVC objects are. 4. What the check wrote; then ~/mlops-smoke is removed.
# Docker runs on your EC2 VM (x86: the linux/amd64 image runs natively there).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
IMAGE="${1:?give the image, for example <repository url>:sha-xxxxxxx}"
mlops_need_course_env
mlops_find_python
mlops_paths
mlops_read_env
mlops_need_env MLOPS_SUFFIX
V1_MD5="$(mlops_manifest_get data.v1.md5)"
V1_KEY="dvc/files/md5/${V1_MD5:0:2}/${V1_MD5:2}"
SMOKE="$HOME/mlops-smoke"

docker run --rm --entrypoint python3 "$IMAGE" -c \
  "import sys, sklearn, mlflow; print('Python', sys.version.split()[0], '/ scikit-learn', sklearn.__version__, '/ mlflow', mlflow.__version__)"

mkdir -p "$MLOPS_PROJECT/data" "$SMOKE/input/raw" "$SMOKE/output/fe" "$SMOKE/output/report"
aws s3 cp --only-show-errors "s3://ccfd-mlops-dvc-$MLOPS_SUFFIX/$V1_KEY" "$MLOPS_PROJECT/$MLOPS_DATA_FILE" \
  || mlops_stop "no data v1 in s3://ccfd-mlops-dvc-$MLOPS_SUFFIX/$V1_KEY. After a full exit (--all), see Coming Back After a Full Exit in MLOps 16."
echo "data v1 from S3: $MLOPS_DATA_FILE (md5 $(md5sum "$MLOPS_PROJECT/$MLOPS_DATA_FILE" | cut -d' ' -f1))"
cp "$MLOPS_PROJECT/$MLOPS_DATA_FILE" "$SMOKE/input/raw/$V1_MD5"
docker run --rm -v "$SMOKE:/opt/ml/processing" --entrypoint python3 "$IMAGE" \
  /opt/program/m08_02_data_checks.py | grep -E "status|rows|Hard rules|PASS|FAIL" | tail -4
ls "$SMOKE/output/fe" "$SMOKE/output/report"
rm -rf "$SMOKE"
