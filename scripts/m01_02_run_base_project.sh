#!/usr/bin/env bash
# m01_02_run_base_project.sh: MLOps 01 Step-02, Option 2 (optional). The same commands as the README's Option 1, in one go:
# make the data and the tuned model ($50,550), start the base project's API on port 8001, run the base project's
# 3-case test, evaluate it on the 2,000 test transactions, stop the API. Runs on your EC2 VM (it bills while it runs).
# It stops at once unless the course env is active: conda activate mlops-aws-env
#
#   bash scripts/m01_02_run_base_project.sh                from the section folder (its ccfd-project/ is used)
#   bash scripts/m01_02_run_base_project.sh --train-only   data + model only (MLOps 02 and 03 start from here)
#
# p5_02 (the 300-fit grid search) is not re-run: its result is frozen in config/p5_02_best_config.yaml.
set -euo pipefail
cd "$(dirname "$0")/../ccfd-project"

echo "== The course environment"
if [ "${CONDA_DEFAULT_ENV:-}" != "mlops-aws-env" ]; then
  echo "STOP: activate the course environment first: conda activate mlops-aws-env (now: ${CONDA_DEFAULT_ENV:-no conda environment})"
  exit 1
fi
if ! python -c "import pandas, sklearn" > /dev/null 2>&1; then
  echo "STOP: mlops-aws-env is active, but the course packages are missing on this machine. Install them once (Section 00-B Step-01-11): cd ~/mlops-real-world-project-implementation-on-aws/00B_AWS_Access_and_Your_EC2_VM/ccfd-project && pip install -q -r requirements-laptop.txt"
  exit 1
fi
echo "mlops-aws-env: OK ($(python --version 2>&1))"

echo "== Data and the tuned model"
mkdir -p artifacts
cp config/p5_02_best_config.yaml artifacts/p5_02_best_config.yaml
python p1_01_generate_initial_dataset.py | grep -E "Total Transactions|Fraud Rate"
python p4_01_feature_engineering_fe1.py >/dev/null
python p5_03_train_production_model.py | grep -E "Net Benefit|Fraud Caught|Fraud Missed|False Alarms"

if [[ "${1:-}" == "--train-only" ]]; then
  echo "Model ready: models/p5_03_production"
  exit 0
fi

echo "== The base project's API on port 8001 and the base project's 3-case test"
python p5_04_inference_api.py >/tmp/mlops_p5_04.log 2>&1 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  curl -s -o /dev/null http://localhost:8001/docs && break
  sleep 1
done
python p5_05_inference_test.py | grep -E "Tests passed|FRAUD|LEGITIMATE" | tail -4
echo "== The 2,000 test transactions on port 8001"
python m01_03_evaluate_endpoint.py --url http://localhost:8001 | tail -2
echo "== Done: the API is stopped. Stop your EC2 VM when you finish (Section 00-B Step-03)."
