#!/usr/bin/env bash
# open-mlflow-ui.sh: MLOps 07 Step-02-03 (and every later section), Option 2. Run it on your LAPTOP (macOS, Linux or
# Windows with WSL2; your EC2 VM has no browser): it
# makes a sign-in link for the MLflow UI of the SageMaker AI MLflow App ccfd-mlflow with your AWS credentials and
# opens it in your browser. There is no MLflow password: IAM decides access.
#
#   bash scripts/open-mlflow-ui.sh
#
# The App ARN is looked up by the App's name (your laptop has no .env.mlops of this section; it is on your EC2 VM). The
# link is valid 300 seconds to open (the session then lasts longer). It is a sign-in: it is opened, never printed.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_masked "$0" "$@"
mlops_need aws
# The course VM (any EC2 machine) has no browser: stop there, never on a laptop
if command -v mlops-vm-check >/dev/null || grep -qs "Amazon EC2" /sys/devices/virtual/dmi/id/sys_vendor; then
  mlops_stop "this opens a browser: run it on your laptop (your EC2 VM has no browser)."
fi
# The command that opens a link in the laptop's browser
if [[ "$(uname -s)" == Darwin ]]; then
  OPENER=open
elif grep -qsi microsoft /proc/version && command -v explorer.exe >/dev/null; then
  OPENER=explorer.exe   # Windows (WSL2): the Windows default browser
elif command -v xdg-open >/dev/null; then
  OPENER=xdg-open       # Linux desktop
else
  mlops_stop "found no command to open a browser (open, explorer.exe or xdg-open): use Option 1 of MLOps 07 Step-02-03."
fi
ARN="$(aws sagemaker list-mlflow-apps --region "$MLOPS_REGION" \
  --query "Summaries[?Name=='ccfd-mlflow' && Status!='Deleted'].Arn | [0]" --output text)"
[[ "$ARN" == arn:* ]] || mlops_stop "no MLflow App ccfd-mlflow in $MLOPS_REGION: create it first (MLOps 07 Step-01)."
URL="$(aws sagemaker create-presigned-mlflow-app-url --arn "$ARN" --expires-in-seconds 300 \
  --region "$MLOPS_REGION" --query AuthorizedUrl --output text)"
"$OPENER" "$URL" || [[ "$OPENER" == explorer.exe ]]   # explorer.exe returns 1 even when the page opened
echo "MLflow UI of ccfd-mlflow: sign-in link made (valid 300 seconds) and opened in your browser"
