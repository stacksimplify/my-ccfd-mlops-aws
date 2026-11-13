#!/usr/bin/env bash
# m12_03_shift_weights.sh: MLOps 12 Step-03-03, Option 2. Move A/B traffic without a redeploy: champion=1 challenger=0
# (the defaults). Option 1 in the README runs the same 3 AWS CLI commands.
set -euo pipefail
CHAMPION="${1:-1}"; CHALLENGER="${2:-0}"
aws sagemaker update-endpoint-weights-and-capacities --endpoint-name ccfd-prod --desired-weights-and-capacities \
  "VariantName=champion,DesiredWeight=$CHAMPION" "VariantName=challenger,DesiredWeight=$CHALLENGER" --query EndpointArn --output text \
  | sed -E 's/[0-9]{12}/<ACCOUNT_ID>/g'
aws sagemaker wait endpoint-in-service --endpoint-name ccfd-prod
aws sagemaker describe-endpoint --endpoint-name ccfd-prod --query 'ProductionVariants[].[VariantName,CurrentWeight]' --output text
