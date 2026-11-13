#!/usr/bin/env bash
# m08_01_build_push_training_image.sh: MLOps 08 Step-01, Option 2. Build the training image for linux/amd64 and push
# it to the ECR repository ccfd-training, tagged with the git commit (sha-<7 characters>), as a Docker v2 manifest.
# Then .env.mlops gets MLOPS_TRAINING_IMAGE_URI (create-env.sh --write-env-only). Run it on your EC2 VM: Docker is there
# (x86, a native build).
#
#   bash scripts/m08_01_build_push_training_image.sh
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_need_docker
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(aws ecr describe-repositories --repository-names ccfd-training --query 'repositories[0].repositoryUri' \
  --output text 2>/dev/null || true)"
[[ "$REPO" == *ccfd-training ]] || { echo "STOP: no ECR repository ccfd-training. Apply infra/01-foundation first."; exit 1; }
TAG="sha-$(git -C "$ROOT" rev-parse --short=7 HEAD)"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${REPO%%/*}" 2>&1 \
  | sed -E 's/[0-9]{12}/<ACCOUNT_ID>/g'
cd "$ROOT/ccfd-project"
# Without a terminal, buildx prints plain lines; the last ones are the push ("pushing layer ... done", "DONE")
docker buildx build --platform linux/amd64 --provenance=false --sbom=false -f Dockerfile.training \
  -t "$REPO:$TAG" --push . 2>&1 | grep -E "pushing|naming|ERROR|DONE" | tail -3 | sed -E 's/[0-9]{12}/<ACCOUNT_ID>/g'
aws ecr describe-images --repository-name ccfd-training --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].[imageTags[0],imageSizeInBytes,imageManifestMediaType]' --output text
bash "$ROOT/scripts/create-env.sh" --write-env-only | grep -E "written|STOP"
