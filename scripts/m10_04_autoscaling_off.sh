#!/usr/bin/env bash
# m10_04_autoscaling_off.sh: MLOps 10 Step-04. Remove the ccfd-rt policy and scalable target (before an update or delete).
set -euo pipefail
RID="endpoint/ccfd-rt/variant/AllTraffic"
aws application-autoscaling delete-scaling-policy --service-namespace sagemaker --resource-id "$RID" \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount --policy-name ccfd-rt-invocations-target 2>/dev/null || true
aws application-autoscaling deregister-scalable-target --service-namespace sagemaker --resource-id "$RID" \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount 2>/dev/null || true
echo "Autoscaling off: $(aws application-autoscaling describe-scalable-targets --service-namespace sagemaker \
  --resource-ids "$RID" --query 'length(ScalableTargets)' --output text) scalable target(s) left on ccfd-rt"
