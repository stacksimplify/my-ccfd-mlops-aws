# shellcheck shell=bash
# The MLOPS_* settings below are used by the scripts that source this file:
# shellcheck disable=SC2034
# scripts/mlops_common.sh: the helpers every course ops script shares. It is sourced, never run on its own:
#   . "$(dirname "$0")/mlops_common.sh"
#
# What is here:
#   mlops_masked           runs the calling script again with stdout AND stderr through the account ID mask
#   mlops_paths            the folders: the project root (scripts/, ccfd-project/, infra/), .env.mlops
#   mlops_read_env         reads .env.mlops (written only by create-env.sh) without running it
#   mlops_r11_check        the identity check: region us-east-1, never the root user, the identity (masked), one account
#   mlops_find_python, mlops_find_terraform (terraform 1.15.x on the PATH), mlops_need
#   mlops_need_course_env  the course environment mlops-aws-env is active, else a STOP line
#   on_course_vm        true when the script runs ON your EC2 VM (the marker file /etc/mlops-course-vm)
#   mlops_need_docker      Docker is here, else a STOP that says where Docker runs
#   mlops_build_push_local on your EC2 VM: one image built with the VM's own Docker and pushed to ECR
#   mlops_tf_state         the resources in a layer's state; "no state yet" is empty, any other error stops
#   mlops_manifest_get     one value of manifest.yaml (the frozen md5s)
#   mlops_ensure_gate_holdout, mlops_ensure_lab_repo, mlops_push_s3remote
#                       the data path: files are remade on this machine when missing and
#                       must have the md5 of manifest.yaml; nothing is assumed to be in S3
#
# Works with the bash 3.2 that macOS ships (no associative arrays, no mapfile).

MLOPS_REGION="us-east-1"
MLOPS_MASK_SED='s/[0-9]{12}/<ACCOUNT_ID>/g'
MLOPS_COURSE_TAG="mlops-on-aws"
MLOPS_GROUP="ccfd-fraud-detector"
MLOPS_DATA_FILE="data/credit_card_transactions_latest.csv"
MLOPS_VM_NAME="mlops-vm"                           # your EC2 VM (Section 00-B Step-01)

export AWS_PAGER=""                                # the AWS CLI never opens a pager inside a script
export DVC_NO_ANALYTICS=1                          # DVC sends no usage data from these scripts
export TF_IN_AUTOMATION=1                          # shorter Terraform output (no "run terraform plan next" hints)
# AWS APIs allow a few calls a second (the model registry fewest): a ThrottlingException ("Rate exceeded") is
# retried with back-off (adaptive mode, up to 10 attempts) instead of stopping the script after 2 retries.
export AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}" AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}"

# ---------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------
mlops_mask() { sed -E "$MLOPS_MASK_SED"; }

# Call as the first line after sourcing:  mlops_masked "$0" "$@"
# The script runs again as a child whose stdout and stderr both go through the mask, so no 12-digit
# account ID reaches the terminal, even from an AWS error. Scripts it starts inherit the mask.
# sed -u: each line reaches the screen at once, also when you pipe the output on (| tee run.log), so a
# question such as "Type yes" is never stuck in a buffer. trap '' INT: Ctrl-C does not stop the mask; it
# ends when the script closes its output, so Terraform's graceful stop after a Ctrl-C can still print and
# save its state (a mask killed by Ctrl-C would leave a dead pipe, and Terraform dies of SIGPIPE on it).
mlops_masked() {
  if [[ -z "${_MLOPS_MASKED:-}" ]]; then
    local script="$1" unbuffered=()
    shift
    if sed -u -n '' < /dev/null > /dev/null 2>&1; then
      unbuffered=(-u)                              # BSD and GNU sed both have -u
    fi
    _MLOPS_MASKED=1 bash "$script" "$@" 2>&1 \
      | (trap '' INT; exec sed ${unbuffered[@]+"${unbuffered[@]}"} -E "$MLOPS_MASK_SED")
    exit "${PIPESTATUS[0]}"
  fi
}

mlops_say()  { printf '%s\n' "$*"; }
mlops_step() { printf '\n== %s\n' "$*"; }
mlops_stop() { printf 'STOP: %s\n' "$*"; exit 1; }

mlops_elapsed() {   # mlops_elapsed <seconds> -> "4m 12s"
  printf '%dm %02ds' "$(($1 / 60))" "$(($1 % 60))"
}

# ---------------------------------------------------------------------------------------------------------
# Folders and .env.mlops
# ---------------------------------------------------------------------------------------------------------
mlops_paths() {
  MLOPS_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  MLOPS_ROOT="$(dirname "$MLOPS_SCRIPTS")"
  MLOPS_PROJECT="$MLOPS_ROOT/ccfd-project"
  MLOPS_INFRA="$MLOPS_ROOT/infra"
  MLOPS_ENV_FILE="$MLOPS_ROOT/.env.mlops"
}

# Read .env.mlops line by line; only lines of the form  export MLOPS_NAME="value"  count. The file is never
# sourced (nothing in it runs). The values of the file win over the environment (a MLOPS_SUFFIX left in your
# shell from an older environment must not point a script at the wrong buckets); only MLOPS_OWNER and MLOPS_PURPOSE
# may be set from the environment (create-env.sh then writes the new value).
# Every MLOPS_* value is exported, so the Python helpers these scripts start see the same values.
mlops_read_env() {
  local line key value re='^(export[[:space:]]+)?(MLOPS_[A-Z0-9_]+)="([^"$`\\]*)"$'
  if [[ -f "$MLOPS_ENV_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      if [[ "$line" =~ $re ]]; then
        key="${BASH_REMATCH[2]}"
        value="${BASH_REMATCH[3]}"
        case "$key" in
          MLOPS_OWNER|MLOPS_PURPOSE) [[ -n "${!key:-}" ]] || printf -v "$key" '%s' "$value" ;;
          *) printf -v "$key" '%s' "$value" ;;
        esac
        export "${key?}"
      elif [[ "$line" =~ ^export[[:space:]]+TF_VAR_alert_email=\"([^\"$\`\\]*)\"$ ]]; then
        # The alert email of MLOps 06 (option B): a value in this terminal wins; else the file keeps it, so a re-run
        # of create-env.sh in a new terminal never drops it (K101 2026-11-02)
        [[ -n "${TF_VAR_alert_email:-}" ]] || export TF_VAR_alert_email="${BASH_REMATCH[1]}"
      fi
    done < "$MLOPS_ENV_FILE"
  fi
}

mlops_need_env() {   # mlops_need_env MLOPS_SUFFIX MLOPS_JOBS_BUCKET ...: stop when one is missing
  local name missing=""
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || missing="$missing $name"
  done
  if [[ -n "$missing" ]]; then
    mlops_stop "missing in .env.mlops:$missing. Run bash scripts/create-env.sh first (it is the only writer of .env.mlops)."
  fi
}

# ---------------------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------------------
mlops_need() {   # mlops_need aws git dvc ...
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || mlops_stop "'$tool' is not installed or not on the PATH."
  done
}

# The course environment (mlops-aws-env) is active and its python has the course packages; else one STOP line
# before anything runs (a script that calls python in (base) would fail later, in the middle of its output).
mlops_need_course_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" != "mlops-aws-env" ]]; then
    mlops_stop "activate the course environment first: conda activate mlops-aws-env (now: ${CONDA_DEFAULT_ENV:-no conda environment})"
  fi
  if ! python -c "import mlflow, sklearn" > /dev/null 2>&1; then
    mlops_stop "mlops-aws-env is active, but the course packages are missing on this machine. Install them once (Section 00-B Step-01-11): cd ~/mlops-real-world-project-implementation-on-aws/00B_AWS_Access_and_Your_EC2_VM/ccfd-project && pip install -q -r requirements-laptop.txt"
  fi
}

# True when this script runs ON your EC2 VM: vm-setup.sh writes the marker file /etc/mlops-course-vm there. On the VM
# Docker builds run directly (no SSH to itself), the region comes from the VM's AWS config and ~/mlops-lab is local.
on_course_vm() { [[ -f /etc/mlops-course-vm ]]; }

mlops_need_docker() {
  command -v docker >/dev/null 2>&1 && return 0
  if on_course_vm; then
    mlops_stop "'docker' is not on the PATH of your EC2 VM. Check the VM toolset: mlops-vm-check"
  fi
  mlops_stop "no Docker here: the course runs Docker on your EC2 VM. Connect to it (Section 00-B Step-01-10) and run this there."
}

# mlops_build_push_local <ccfd-project folder> <commit> <Dockerfile> <ECR repository URL:tag>, on your EC2 VM only.
# The build of create-env.sh: the COMMITTED tree of that commit (git archive: exactly what the tag sha-<commit>
# says), Docker logged in to ECR with the VM's role, buildx with --provenance=false --sbom=false and a Docker v2
# manifest (SageMaker refuses an OCI index), pushed. The caller reads the manifest type back from ECR. Returns 1 on
# any error (each step is checked: a caller's "|| mlops_stop" switches set -e off in here).
mlops_build_push_local() {
  local project="$1" commit="$2" dockerfile="$3" target="$4" prefix top work status=0
  prefix="$(git -C "$project" rev-parse --show-prefix)" || { mlops_say "STOP: $project is not in a git clone"; return 1; }
  top="$(git -C "$project" rev-parse --show-toplevel)" || return 1
  git -C "$project" cat-file -e "$commit:${prefix}$dockerfile" 2>/dev/null \
    || { mlops_say "STOP: $dockerfile is not in commit $commit"; return 1; }
  work="$(mktemp -d "$HOME/.mlops-build-XXXXXX")" || return 1
  mlops_say "${target##*/}: not in ECR yet, building it on $MLOPS_VM_NAME ($dockerfile, linux/amd64)"
  # git archive runs from the top of the clone: run in a subfolder, it would keep only paths under that subfolder,
  # and the tree <commit>:<prefix> has none (an empty build context)
  git -C "$top" archive --format=tar "$commit:$prefix" | tar -C "$work" -xf - || status=1
  if [[ "$status" -eq 0 && ! -f "$work/$dockerfile" ]]; then
    mlops_say "STOP: the archive of commit $commit has no $dockerfile (an empty build context)"
    status=1
  fi
  if [[ "$status" -eq 0 ]]; then
    aws ecr get-login-password --region "$MLOPS_REGION" \
      | docker login --username AWS --password-stdin "${target%%/*}" > /dev/null || status=1
  fi
  if [[ "$status" -eq 0 ]]; then
    ( cd "$work" || exit 1
      docker buildx build --platform linux/amd64 --provenance=false --sbom=false -f "$dockerfile" \
        --output "type=image,name=$target,push=true,oci-mediatypes=false" . 2>&1 | tail -n 3
      exit "${PIPESTATUS[0]}" ) || status=1
  fi
  rm -rf "$work"
  [[ "$status" -eq 0 ]] || return 1
  mlops_say "${target##*/}: pushed to ECR"
}

# PY = the Python of the course env (mlops-aws-env), with PyYAML and boto3
mlops_find_python() {
  PY="${PYTHON:-python}"
  if ! "$PY" -c 'import yaml, boto3' >/dev/null 2>&1; then
    mlops_stop "'$PY' cannot import yaml and boto3. Activate the course env first: conda activate mlops-aws-env"
  fi
}

# TF = terraform on the PATH, version 1.15.x (every layer says required_version = "~> 1.15.0"). Silent when it is
# right; a STOP otherwise.
mlops_find_terraform() {
  local version
  TF="terraform"
  version="$("$TF" version -json 2>/dev/null | sed -n 's/.*"terraform_version": *"\([^"]*\)".*/\1/p' | head -n 1 || true)"
  [[ "$version" == 1.15.* ]] && return 0
  if on_course_vm; then
    mlops_stop "Terraform on your EC2 VM is '${version:-not found}', not 1.15.x. Check the VM toolset: mlops-vm-check"
  fi
  mlops_stop "Terraform here is '${version:-not found}', not 1.15.x (every layer needs ~> 1.15.0). Run this on your EC2 VM."
}

# mlops_tf_state <layer folder>: the resources in the layer's state, one per line (run terraform init first).
# Nothing is printed when the layer has no state yet (Terraform says "No state file was found!"). Any other
# error (no access to the state bucket, a broken state file, the network) prints Terraform's text and fails:
# an error is never read as "nothing there". Use it as  X="$(mlops_tf_state <folder>)" || exit 1
mlops_tf_state() {
  local out
  if out="$("$TF" -chdir="$1" state list -no-color 2>&1)"; then
    [[ -z "$out" ]] || printf '%s\n' "$out"
    return 0
  fi
  case "$out" in
    *"No state file was found!"*) return 0 ;;
  esac
  {
    printf '%s\n' "$out"
    printf 'STOP: terraform state list failed in infra/%s (above): the state could not be read, so nothing is assumed.\n' "${1##*/}"
  } >&2
  return 1
}

# ---------------------------------------------------------------------------------------------------------
# The identity check: one AWS account, us-east-1
# ---------------------------------------------------------------------------------------------------------
# Prints the identity masked. Stops on another region, the root user, or an account other than the one in .env.mlops.
mlops_r11_check() {
  local region arn account env_account=""
  region="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || true)}}"
  if [[ "$region" != "$MLOPS_REGION" ]]; then
    mlops_stop "the AWS region is '${region:-not set}'. This course uses $MLOPS_REGION only: aws configure set region $MLOPS_REGION"
  fi
  if ! arn="$(aws sts get-caller-identity --query Arn --output text 2>&1)"; then
    mlops_stop "no AWS identity (aws sts get-caller-identity): $(printf '%s' "$arn" | mlops_mask)"
  fi
  mlops_say "Identity check: region $region, identity $(printf '%s' "$arn" | mlops_mask)"
  case "$arn" in
    *:root) mlops_stop "you are signed in as the root user. Use an IAM user." ;;
  esac
  account="$(printf '%s' "$arn" | cut -d: -f5)"
  if [[ -n "${MLOPS_EXECUTION_ROLE_ARN:-}" ]]; then
    env_account="$(printf '%s' "$MLOPS_EXECUTION_ROLE_ARN" | cut -d: -f5)"
  fi
  if [[ -n "$env_account" && "$env_account" != "$account" ]]; then
    mlops_stop ".env.mlops belongs to another AWS account than the one you are signed in to (one account). Sign in to the right account."
  fi
}

# ---------------------------------------------------------------------------------------------------------
# manifest.yaml: the frozen inputs (md5s), next to scripts/ in the section folder
# ---------------------------------------------------------------------------------------------------------
mlops_manifest_file() {
  printf '%s\n' "$MLOPS_ROOT/manifest.yaml"
}

mlops_manifest_get() {   # mlops_manifest_get data.v1.md5
  "$PY" - "$(mlops_manifest_file)" "$1" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    value = yaml.safe_load(stream)
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

mlops_md5() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | cut -d' ' -f1
  else
    md5 -q "$1"
  fi
}

# Run a manifest.yaml command (for example "python m03_03_simulate_new_data.py --mode ...") in the current
# folder: split into words (never eval), with "python" replaced by the course env's Python.
mlops_run_manifest_command() {
  local words
  read -r -a words <<< "$1"
  if [[ "${words[0]}" == "python" ]]; then
    words[0]="$PY"
  fi
  "${words[@]}"
}

# ---------------------------------------------------------------------------------------------------------
# The gate holdout, the local part (manifest.yaml gate_holdout)
# ---------------------------------------------------------------------------------------------------------
mlops_ensure_gate_holdout() {
  local file="$MLOPS_PROJECT/data/gate_holdout.csv" want cmd got
  want="$(mlops_manifest_get gate_holdout.md5)"
  cmd="$(mlops_manifest_get gate_holdout.command)"
  if [[ ! -f "$file" ]]; then
    mlops_say "gate holdout missing (data/gate_holdout.csv is gitignored): making it with: $cmd"
    mkdir -p "$MLOPS_PROJECT/data"
    if ! (cd "$MLOPS_PROJECT" && mlops_run_manifest_command "$cmd") >/dev/null; then
      mlops_stop "the command failed. Run it yourself to see why: (cd ccfd-project && $cmd)"
    fi
  fi
  got="$(mlops_md5 "$file")"
  if [[ "$got" != "$want" ]]; then
    mlops_say "ccfd-project/data/gate_holdout.csv has md5 $got, manifest.yaml says $want."
    mlops_stop "make it again: rm ccfd-project/data/gate_holdout.csv && (cd ccfd-project && $cmd)"
  fi
}

# ---------------------------------------------------------------------------------------------------------
# The lab repo with data v1 and v2, never assumed to exist
# ---------------------------------------------------------------------------------------------------------
mlops_lab_paths() {   # mlops_lab_paths [<lab repo>]: LAB (default ~/mlops-lab/ccfd-project) and its local DVC remote
  LAB="${1:-$HOME/mlops-lab/ccfd-project}"
  LAB_REMOTE="$(dirname "$LAB")/dvc-local-remote"
}

mlops_pointer_md5() {   # mlops_pointer_md5 <git tag>: the md5 in the DVC pointer of the data file at that tag
  git -C "$LAB" show "$1:$MLOPS_DATA_FILE.dvc" 2>/dev/null | sed -n 's/^- md5: \([0-9a-f]\{32\}\)$/\1/p' | head -n 1
}

mlops_lab_ok() {   # true when the lab repo has tags data-v1 and data-v2 with the manifest.yaml md5s
  [[ -d "$LAB/.git" && -d "$LAB/.dvc" ]] || return 1
  [[ "$(mlops_pointer_md5 data-v1)" == "$(mlops_manifest_get data.v1.md5)" ]] || return 1
  [[ "$(mlops_pointer_md5 data-v2)" == "$(mlops_manifest_get data.v2.md5)" ]] || return 1
}

# Copy the code of ccfd-project into the new lab repo: what a clone holds (the folders and files that
# ccfd-project/.gitignore ignores stay out). Prints the copied paths, one per line.
mlops_copy_project_code() {
  "$PY" - "$MLOPS_PROJECT" "$1" <<'PY'
import shutil
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
SKIP_DIRS = {"data", "models", "results", "artifacts", "mlruns", "sm-local", "__pycache__", ".pytest_cache",
             ".ruff_cache", ".venv", "venv", ".ipynb_checkpoints", "eda_plots", "auto_eda_reports"}
SKIP_NAMES = {".DS_Store", "data_checks_report.json", "gate_decision.json", "evaluation.json",
              "metadata_latest.json"}
SKIP_SUFFIXES = (".csv", ".pkl", ".pyc", ".tar.gz", ".tfstate", ".log")
for path in sorted(source.rglob("*")):
    relative = path.relative_to(source)
    if path.is_dir() or any(part in SKIP_DIRS for part in relative.parts[:-1]):
        continue
    if relative.name in SKIP_NAMES or relative.name.endswith(SKIP_SUFFIXES):
        continue
    if relative.name.startswith(".env"):
        continue
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    print(relative.as_posix())
PY
}

# A data version in the lab repo: run its manifest command, check the md5, dvc add, commit, tag
mlops_lab_data_version() {   # mlops_lab_data_version v1 "<commit message>"
  local version="$1" message="$2" cmd want got
  cmd="$(mlops_manifest_get "data.$version.command")"
  want="$(mlops_manifest_get "data.$version.md5")"
  mlops_say "data $version: $cmd"
  if ! (cd "$LAB" && mlops_run_manifest_command "$cmd") >/dev/null; then
    mlops_stop "the command failed. Run it yourself to see why: (cd $LAB && $cmd)"
  fi
  got="$(mlops_md5 "$LAB/$MLOPS_DATA_FILE")"
  if [[ "$got" != "$want" ]]; then
    mlops_say "data $version has md5 $got, manifest.yaml says $want. Nothing was committed for it."
    mlops_stop "the command that makes it: (cd $LAB && $cmd)"
  fi
  mlops_say "data $version: md5 $got = manifest.yaml"
  (cd "$LAB" && dvc add -q "$MLOPS_DATA_FILE")
  git -C "$LAB" add -- "$MLOPS_DATA_FILE.dvc"
  if [[ -f "$LAB/data/.gitignore" ]]; then   # DVC writes it only when the course .gitignore does not ignore the file
    git -C "$LAB" add -- data/.gitignore
  fi
  git -C "$LAB" commit -q -m "$message"
  git -C "$LAB" tag "data-$version"
}

mlops_make_lab_repo() {
  local list
  mkdir -p "$(dirname "$LAB")" "$LAB_REMOTE"
  mlops_say "making the lab repo $LAB, the MLOps 03 way"
  list="$(mktemp)"
  mlops_copy_project_code "$LAB" > "$list"
  git -C "$LAB" init -q -b main
  (cd "$LAB" && dvc init -q && dvc remote add -d localremote "$LAB_REMOTE" >/dev/null)
  # Each copied file by name (never 'git add .'), then DVC's own files
  tr '\n' '\0' < "$list" | (cd "$LAB" && xargs -0 git add --)
  git -C "$LAB" add -- .dvc/config .dvc/.gitignore .dvcignore
  git -C "$LAB" commit -q -m "Lab repo: the course ccfd-project with DVC (remote localremote)"
  rm -f "$list"
  mlops_lab_data_version v1 "data v1: the base project's p1_01, seed 42 (10,000 rows)"
  mlops_lab_data_version v2 "data v2: m03_03 --mode append --rows 5000 --seed 201 (15,000 rows)"
  (cd "$LAB" && dvc push -q --all-commits -r localremote)
  mlops_say "lab repo ready: tags data-v1 and data-v2, both pushed to localremote ($LAB_REMOTE)"
}

# Keep a lab repo that already has both md5s
# right; make a missing one; never touch a folder that is something else.
mlops_ensure_lab_repo() {
  if mlops_lab_ok; then
    mlops_say "lab repo kept: $LAB has tags data-v1 and data-v2 with the manifest.yaml md5s"
    return 0
  fi
  if [[ -e "$LAB" ]]; then
    mlops_say "$LAB exists but does not hold tags data-v1 and data-v2 with the manifest.yaml md5s."
    mlops_stop "nothing was changed there. Move it aside and run again: mv $LAB $LAB.old"
  fi
  mlops_make_lab_repo
}

# s3remote = the dvc bucket of the current .env.mlops, the default remote (MLOps 06
# Step-04; localremote stays). Set every time (a lab repo kept from an earlier environment may still point at
# a destroyed bucket), committed in the lab repo, then every version pushed.
mlops_push_s3remote() {
  local url="s3://ccfd-mlops-dvc-${MLOPS_SUFFIX}/dvc"
  (cd "$LAB" && dvc remote add -f -d s3remote "$url" >/dev/null)
  if ! git -C "$LAB" diff --quiet -- .dvc/config; then
    git -C "$LAB" add -- .dvc/config
    git -C "$LAB" commit -q -m "dvc: s3remote = $url (default remote)"
  fi
  (cd "$LAB" && dvc fetch -q --all-commits -r localremote)
  # Not "dvc push --all-commits -r s3remote": DVC reads each commit's own .dvc/config, and the data-v* commits were
  # made before s3remote existed, so it skips them ("failed to collect ..., skipping") and only HEAD's data is pushed
  # (Run 1, 2026-10-21). So each tag's pointer is pushed by name (the objects are in the cache), then checked in S3.
  local ptr="data/credit_card_transactions_latest.csv.dvc" tag md5 missing=0
  for tag in $(git -C "$LAB" tag -l 'data-v*'); do
    git -C "$LAB" show "$tag:$ptr" > "$LAB/$ptr"
    mlops_say "dvc push -r s3remote $ptr (as at $tag)"
    (cd "$LAB" && dvc push -q -r s3remote "$ptr")
  done
  git -C "$LAB" checkout -q -- "$ptr"
  (cd "$LAB" && dvc push -q -r s3remote)
  for tag in $(git -C "$LAB" tag -l 'data-v*'); do
    md5="$(git -C "$LAB" show "$tag:$ptr" | sed -n 's/^- md5: //p')"
    if aws s3api head-object --bucket "ccfd-mlops-dvc-${MLOPS_SUFFIX}" --key "dvc/files/md5/${md5:0:2}/${md5:2}" \
         --region "$MLOPS_REGION" >/dev/null 2>&1; then
      mlops_say "s3remote has $tag (md5 $md5)"
    else
      mlops_say "s3remote is MISSING $tag (md5 $md5)"; missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || mlops_stop "a data version is missing in s3remote (see above)"
}
