#!/usr/bin/env bash
# m10_01_local_container_test.sh: MLOps 10 Step-01. The package's own image and model, on your EC2 VM, before an endpoint.
#
#   bash scripts/m10_01_local_container_test.sh <package version>
#
# It reads the package (image URI + model.tar.gz), pulls the image from ECR, untars the model into a folder in your
# home, runs the image with the model at /opt/ml/model, calls /ping and /invocations with the base
# project's 3 test transactions, then stops it. Docker runs on your EC2 VM (x86: the image runs natively there).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
VERSION="${1:?give the package version, for example 1}"
mlops_need_course_env
mlops_need_docker
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARN="$(aws sagemaker list-model-packages --model-package-group-name ccfd-fraud-detector \
  --query "ModelPackageSummaryList[?ModelPackageVersion==\`$VERSION\`].ModelPackageArn" --output text)"
[[ -n "$ARN" ]] || { echo "STOP: no version $VERSION in ccfd-fraud-detector"; exit 1; }
IMAGE="$(aws sagemaker describe-model-package --model-package-name "$ARN" --query 'InferenceSpecification.Containers[0].Image' --output text)"
TAR="$(aws sagemaker describe-model-package --model-package-name "$ARN" --query 'InferenceSpecification.Containers[0].ModelDataUrl' --output text)"
MODEL="$HOME/.mlops-model-v$VERSION"
rm -rf "$MODEL" && mkdir -p "$MODEL"
aws s3 cp "$TAR" - --only-show-errors | tar -xz -C "$MODEL"
echo "Model of version $VERSION: $(cd "$MODEL" && printf '%s ' *)"
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${IMAGE%%/*}" >/dev/null 2>&1
docker rm -f ccfd-local-test >/dev/null 2>&1 || true
docker run -d --name ccfd-local-test -p 8080:8080 -v "$MODEL:/opt/ml/model:ro" "$IMAGE" >/dev/null
for _ in $(seq 1 60); do curl -sf -o /dev/null localhost:8080/ping && break; sleep 2; done
curl -s -o /dev/null -w "/ping -> HTTP %{http_code}\n" localhost:8080/ping
curl -s -X POST localhost:8080/invocations -H 'Content-Type: application/json' \
  -d @"$ROOT/ccfd-project/tests/payloads/sample_transactions.json" \
  | python3 -c "import json,sys; [print(f\"{r['prediction']} {100*r['probability']:.2f}% {r['risk_score']}\") for r in json.load(sys.stdin)]"
docker rm -f ccfd-local-test >/dev/null
echo "Stopped. The same image and model run on the endpoint next."
