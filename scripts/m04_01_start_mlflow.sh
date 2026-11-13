#!/usr/bin/env bash
# m04_01_start_mlflow.sh: MLOps 04 and 05, Option 2. The same mlflow server command as Option 1, in the background:
# it survives a closed terminal, and its log goes to ~/mlops-lab/mlflow/server.log. No AWS change.
#
#   bash scripts/m04_01_start_mlflow.sh           start it (already running: it says so and changes nothing)
#   bash scripts/m04_01_start_mlflow.sh --stop    stop the server this script started
#
# Run it in the course environment (conda activate mlops-aws-env): it has mlflow 3.16.0.
set -euo pipefail
URL="http://127.0.0.1:7080"
DATA="$HOME/mlops-lab/mlflow"
up() { [[ "$(curl -fsS --max-time 2 "$URL/health" 2>/dev/null)" == "OK" ]]; }
case "${1:-}" in
  ""|--stop) ;;
  *) echo "Usage: bash $0 [--stop]"; exit 2 ;;
esac

if [[ "${1:-}" == "--stop" ]]; then
  pid="$(cat "$DATA/server.pid" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! ps -o command= -p "$pid" 2>/dev/null | grep -q "mlflow server"; then
    rm -f "$DATA/server.pid"
    up && { echo "This script did not start the server on port 7080: press Ctrl+C in its terminal."; exit 1; }
    echo "No MLflow server is running on port 7080."; exit 0
  fi
  kill -- "-$pid" 2>/dev/null || kill "$pid"   # the server and its workers: the process group it got at the start
  for _ in $(seq 1 20); do up || break; sleep 0.5; done
  up && { echo "ERROR: a server still answers on $URL. Run this again in a few seconds."; exit 1; }
  rm -f "$DATA/server.pid"
  echo "MLflow server stopped. Port 7080 is free (your runs stay in ~/mlops-lab/mlflow)."
  exit 0
fi

if up; then echo "MLflow server already running: $URL (nothing changed)"; exit 0; fi
mkdir -p "$DATA/artifacts"
cd "$DATA"
set -m                                # its own process group, so --stop ends the server and all its workers
nohup mlflow server --host 127.0.0.1 --port 7080 --backend-store-uri "sqlite:///$DATA/mlflow.db" \
  --artifacts-destination "$DATA/artifacts" > server.log 2>&1 &
pid=$!
set +m
echo "$pid" > server.pid
# The first start creates the database tables: up to 60 seconds (it stops waiting when the server exits)
for _ in $(seq 1 60); do up && break; kill -0 "$pid" 2>/dev/null || break; sleep 1; done
if ! up; then echo "ERROR: the server did not answer on $URL. The last lines of its log:"; tail -n 20 server.log; exit 1; fi
echo "MLflow server started: $URL"
echo "Data: ~/mlops-lab/mlflow (mlflow.db = runs and models, artifacts/ = model files, server.log = the log)"
echo "Connect: export MLFLOW_TRACKING_URI=$URL"
echo "Stop:    bash $0 --stop"
