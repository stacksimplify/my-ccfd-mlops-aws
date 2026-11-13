#!/usr/bin/env bash
# m03_02_make_lab_repo.sh: MLOps 03 Step-02. Copies this section's ccfd-project to the lab folder, outside the
# course clone, so your own git and DVC repo never sits inside the course repo. It copies code only: no data, models,
# results or caches. It runs NO git and NO dvc command: you type those yourself in the README.
#
#   bash ../scripts/m03_02_make_lab_repo.sh                  the lab at ~/mlops-lab/ccfd-project
#   bash ../scripts/m03_02_make_lab_repo.sh <folder>         the lab somewhere else (outside the course clone)
#
# It stops, and changes nothing, when the target already exists or is inside a git repo.
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/../ccfd-project" && pwd)"
TARGET="${1:-$HOME/mlops-lab/ccfd-project}"

if [[ -e "$TARGET" ]]; then
  echo "STOP: $TARGET already exists. Nothing was changed. Use it, or move it aside: mv $TARGET $TARGET.old"
  exit 1
fi
PARENT="$(dirname "$TARGET")"
mkdir -p "$PARENT"
if git -C "$PARENT" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "STOP: $PARENT is inside the git repo $(git -C "$PARENT" rev-parse --show-toplevel)."
  echo "The lab repo must live outside any git repo (for example ~/mlops-lab/ccfd-project)."
  exit 1
fi

mkdir -p "$TARGET"
# Code only: data, models, results, local runs and caches stay behind (each data version is made in the lab)
rsync -a \
  --exclude 'data/' --exclude 'models/' --exclude 'results/' --exclude 'artifacts/' --exclude 'mlruns/' \
  --exclude 'logs/' --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
  --exclude '*.csv' --exclude '*.pkl' --exclude '.env*' --exclude '.DS_Store' \
  "$SOURCE/" "$TARGET/"

echo "Lab folder ready: ~/${TARGET#"$HOME"/} ($(find "$TARGET" -type f | wc -l | tr -d ' ') files, no data, not a git repo yet)"
echo "Next: Step-02-02 checks DVC, then Step-02-03 makes the git repo in ~/${TARGET#"$HOME"/}"
