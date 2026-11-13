#!/usr/bin/env bash
# m06_05_push_image.sh: MLOps 06 Step-05, Option 2. Build the serving image for linux/amd64 and push it to the ECR
# repository ccfd-serving, tagged with the git commit (sha-<7 characters>).
#
#   bash scripts/m06_05_push_image.sh
#
# The same commands as the README's Option 1. The image goes up as a Docker v2 manifest (--provenance=false
# --sbom=false): SageMaker refuses an OCI index. Run it on your EC2 VM: Docker is there (x86, a native build).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_need_docker
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(aws ecr describe-repositories --repository-names ccfd-serving --query 'repositories[0].repositoryUri' \
  --output text 2>/dev/null || true)"
[[ "$REPO" == *ccfd-serving ]] || { echo "STOP: no ECR repository ccfd-serving. Apply infra/01-foundation first (README Step-03)."; exit 1; }
TAG="sha-$(git -C "$ROOT" rev-parse --short=7 HEAD)"
REGISTRY="${REPO%%/*}"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY" 2>&1 \
  | sed -E 's/[0-9]{12}/<ACCOUNT_ID>/g'
cd "$ROOT/ccfd-project"
docker buildx build --platform linux/amd64 --provenance=false --sbom=false -f Dockerfile.serving \
  -t "$REPO:$TAG" --push . 2>&1 | grep -E "pushing|naming|ERROR|DONE" | tail -3 | sed -E 's/[0-9]{12}/<ACCOUNT_ID>/g'
aws ecr describe-images --repository-name ccfd-serving --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].[imageTags[0],imageSizeInBytes,imageManifestMediaType,imagePushedAt]' --output text
