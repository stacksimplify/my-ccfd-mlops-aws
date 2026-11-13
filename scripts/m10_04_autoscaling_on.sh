#!/usr/bin/env bash
# m10_04_autoscaling_on.sh: MLOps 10 Step-04. Target tracking on ccfd-rt: 1 to 2 instances, 300 calls per instance per minute.
set -euo pipefail
export AWS_PAGER=""
RID="endpoint/ccfd-rt/variant/AllTraffic"
aws application-autoscaling register-scalable-target --service-namespace sagemaker --resource-id "$RID" \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount --min-capacity 1 --max-capacity 2 \
  --query ScalableTargetARN --output text > /dev/null
aws application-autoscaling put-scaling-policy --service-namespace sagemaker --resource-id "$RID" \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount --policy-name ccfd-rt-invocations-target \
  --policy-type TargetTrackingScaling --target-tracking-scaling-policy-configuration \
  '{"TargetValue": 300, "PredefinedMetricSpecification": {"PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"}, "ScaleOutCooldown": 60, "ScaleInCooldown": 300}' \
  --query 'Alarms[].AlarmName' --output text | tr '\t' '\n' | sed 's/^/alarm: /'
echo "Autoscaling on: ccfd-rt AllTraffic, 1 to 2 instances, target 300 invocations per instance per minute"
