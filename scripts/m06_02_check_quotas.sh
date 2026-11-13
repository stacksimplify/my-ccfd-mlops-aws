#!/usr/bin/env bash
# m06_02_check_quotas.sh: MLOps 06 Step-02-03, Option 2. The 2 SageMaker AI quotas the course needs. READ ONLY, free.
# The same 2 reads as Option 1. A quota below the need gets the free request to file now (AWS answers by email,
# sometimes in a couple of days). Every other quota the course uses is high enough on a new account.
# Exit codes: 0 both are enough, 1 a request to file (or an AWS error).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_masked "$0" "$@"
mlops_r11_check
quota() {   # quota get-service-quota|get-aws-default-service-quota <code>: the quota's name and value
  aws service-quotas "$1" --service-code sagemaker --quota-code "$2" --query 'Quota.[QuotaName,Value]' --output text
}
LOW=0
# <code>:<need> = ml.m5.large endpoints (the course runs up to 4 at once), ml.m5.large batch transform (MLOps 11)
for row in L-614B09FD:4 L-236AE59F:1; do
  code="${row%:*}" need="${row#*:}"
  # an account that never changed a quota may have no applied value: then the AWS default applies
  line="$(quota get-service-quota "$code" 2>/dev/null || quota get-aws-default-service-quota "$code")"
  IFS=$'\t' read -r name value <<< "$line"
  if [[ "${value%.*}" -ge "$need" ]]; then printf '%-36s %5s  (need %s)  OK\n' "$name" "${value%.*}" "$need"; continue; fi
  printf '%-36s %5s  (need %s)  LOW: file this free request now:\n' "$name" "${value%.*}" "$need"
  echo "  aws service-quotas request-service-quota-increase --service-code sagemaker --quota-code $code --desired-value $need"
  LOW=1
done
[[ "$LOW" -eq 0 ]] || { echo "AWS answers by email, sometimes in a couple of days; run this check again then."; exit 1; }
echo "Both quotas are enough: nothing to request."
