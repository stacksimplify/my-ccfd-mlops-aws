#!/usr/bin/env bash
# create-env.sh: make (or bring up to date) the course environment on AWS, layer by layer.
#
#   bash scripts/create-env.sh --layers 00,01       the layers you name (00 to 05), always applied in order
#   bash scripts/create-env.sh --write-env-only     no change on AWS: write .env.mlops from the layers that exist
#
# Run it again at any time: what exists stays (Terraform changes only what differs; an image tag already in ECR is
# not pushed again). The images are tagged sha-<7 characters of the git commit>, so with 01 it first checks that
# ccfd-project has no uncommitted change to a tracked file (commit first).
# The tag values: the ones of .env.mlops, else the state bucket's own tags, else your Terraform exports of MLOps 06
# (TF_VAR_owner, TF_VAR_purpose, TF_VAR_created). Layer 03 reads TF_VAR_alert_email (the SNS email subscription).
# Layer 04 reads TF_VAR_github_repository, TF_VAR_github_owner_id, TF_VAR_github_repository_id and
# TF_VAR_create_github_oidc_provider (true only in an account that has no GitHub OIDC provider yet; default false).
#
# In order: the cost line; the identity check (region us-east-1, never the root user); each layer asked for (after 01
# a missing image is built and pushed on your EC2 VM, and the gate holdout and the base project's fixed YAML are
# uploaded to the jobs bucket; after 02 the MLflow App's version is checked); .env.mlops after each layer.
# This script is the ONLY writer of .env.mlops. Every line it prints goes through the account ID mask.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_masked "$0" "$@"

MLFLOW_PIN="3.16.0"                                   # = the client pin in every lock
APP_NAME="ccfd-mlflow"
APP_MODE="AutoModelRegistrationDisabled"              # MLflow never writes the SageMaker registry
V2_MANIFEST="application/vnd.docker.distribution.manifest.v2+json"

usage() {
  sed -n '4,5p' "$0"
}

LAYERS=""
ENV_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --layers) [[ $# -ge 2 ]] || { usage; exit 2; }; LAYERS="$2"; shift 2 ;;
    --layers=*) LAYERS="${1#--layers=}"; shift ;;
    --write-env-only) ENV_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

layer_dir() {
  case "$1" in
    00) echo 00-state-bucket ;;
    01) echo 01-foundation ;;
    02) echo 02-mlflow-app ;;
    03) echo 03-registry-eventing ;;
    04) echo 04-cicd-oidc ;;
    05) echo 05-monitoring ;;
    *) return 1 ;;
  esac
}

# The layers asked for: split on commas and spaces, checked, used once each, in order 00 -> 05
WANT=""
if [[ "$ENV_ONLY" -eq 0 ]]; then
  TOKENS="$(printf '%s\n' "$LAYERS" | tr -s ', \t' '\n' | { grep -v '^$' || true; })"
  if [[ -z "$TOKENS" ]]; then
    echo "Name the layers (00 to 05), for example --layers 00,01, or use --write-env-only"
    usage
    exit 2
  fi
  for id in $TOKENS; do
    layer_dir "$id" >/dev/null || { echo "Unknown layer '$id': use 00 to 05, for example --layers 00,01"; exit 2; }
  done
  for id in 00 01 02 03 04 05; do
    if printf '%s\n' "$TOKENS" | grep -qx "$id"; then
      WANT="$WANT $id"
    fi
  done
  WANT="${WANT# }"
fi
wants() { [[ " $WANT " == *" $1 "* ]]; }

# ---------------------------------------------------------------------------------------------------------
# 1. The cost line first
# ---------------------------------------------------------------------------------------------------------
if [[ "$ENV_ONLY" -eq 1 ]]; then
  echo "Cost: 0 USD. --write-env-only reads Terraform outputs and ECR tags; it changes nothing on AWS."
else
  echo "Cost (estimate): layers $WANT. Nothing here bills by the hour: no endpoint and no job is started."
  wants 00 && echo "  00: the state bucket, S3 0.023 USD per GB-month (a few KB of state: about 0 USD)."
  wants 01 && echo "  01: 3 buckets (S3 0.023 USD per GB-month) and 2 ECR repositories (0.10 USD per GB-month; both images about 0.45 GB: about 0.05 USD a month); the IAM role is free."
  wants 02 && echo "  02: the MLflow App has no hourly charge (its artifacts are S3 in the mlflow bucket); its role is free."
  wants 03 && echo "  03: SNS topic and EventBridge rule: free at course volume."
  wants 04 && echo "  04: IAM only (the GitHub Actions role): free."
  wants 05 && echo "  05: 2 alarms and 1 dashboard: inside the CloudWatch free tier (10 alarms, 3 dashboards)."
  echo "  None of it bills by the hour: the end-of-session destroy-env.sh keeps it; destroy-env.sh --all removes it."
fi

# ---------------------------------------------------------------------------------------------------------
# 2. Where we are, who we are, the tools
# ---------------------------------------------------------------------------------------------------------
mlops_paths
ENV_CREATED="${MLOPS_CREATED:-${TF_VAR_created:-}}"       # a date you exported yourself wins for a NEW environment
mlops_read_env
mlops_need aws git sed
mlops_find_python
mlops_find_terraform
mlops_r11_check

# ---------------------------------------------------------------------------------------------------------
# 3. Terraform helpers (the state of 00 in its own folder; 01-05 in the state bucket through infra/backend.hcl)
# ---------------------------------------------------------------------------------------------------------
tf_init() {   # tf_init <layer id>
  local dir log args=()
  dir="$(layer_dir "$1")"
  if [[ "$1" != "00" ]]; then
    [[ -f "$MLOPS_INFRA/backend.hcl" ]] || mlops_stop "no infra/backend.hcl: 00 is not applied here yet (bash scripts/create-env.sh --layers 00)."
    args=(-reconfigure -backend-config=../backend.hcl)   # backend.hcl is the truth (a new bucket after --all)
  fi
  log="$(mktemp)"
  if ! "$TF" -chdir="$MLOPS_INFRA/$dir" init -input=false -no-color -lockfile=readonly ${args[@]+"${args[@]}"} > "$log" 2>&1; then
    cat "$log"
    rm -f "$log"
    mlops_stop "terraform init failed in infra/$dir (see above)."
  fi
  rm -f "$log"
}

tf_has_state() {   # true when the layer's state holds at least one resource; a state that cannot be read stops
  local list
  [[ -d "$MLOPS_INFRA/$(layer_dir "$1")" ]] || return 1   # a section folder holds only the layers it teaches
  list="$(mlops_tf_state "$MLOPS_INFRA/$(layer_dir "$1")")" || exit 1
  [[ -n "$list" ]]
}

tf_out() {   # tf_out <layer id> <output name>
  "$TF" -chdir="$MLOPS_INFRA/$(layer_dir "$1")" output -raw "$2"
}

# The images are tagged sha-<7 characters of HEAD> and built from that commit (git archive), and a tag already in ECR
# is kept on a re-run. So a change to a tracked file of ccfd-project that is not committed stops the script before
# anything changes on AWS: it would not be in the image. Untracked files (outputs of the steps) never stop it.
code_is_committed() {
  local dirty
  IMAGE_SHA="$(git -C "$MLOPS_ROOT" rev-parse --short=7 HEAD 2>/dev/null)" \
    || mlops_stop "the images are tagged sha-<7 characters of the git commit>: run this from a git clone."
  dirty="$(git -C "$MLOPS_ROOT" status --porcelain --untracked-files=no -- ccfd-project)" \
    || mlops_stop "git status failed in $MLOPS_ROOT."
  if [[ -n "$dirty" ]]; then
    printf '%s\n' "$dirty" | head -n 20 | sed 's/^/  /'
    mlops_stop "ccfd-project has changes that are not committed (above). The images are built from the last commit ($IMAGE_SHA), so these changes would not be in them. Commit first (git add <each file>, then git commit), then run this again. Nothing was changed on AWS."
  fi
}

# ---------------------------------------------------------------------------------------------------------
# 4. .env.mlops: only this script writes it
# ---------------------------------------------------------------------------------------------------------
ENV_KEYS="MLOPS_OWNER MLOPS_PURPOSE MLOPS_CREATED MLOPS_SUFFIX MLOPS_JOBS_BUCKET MLOPS_EXECUTION_ROLE_ARN MLOPS_SERVING_IMAGE_URI MLOPS_TRAINING_IMAGE_URI MLOPS_MODEL_PACKAGE_GROUP MLOPS_MLFLOW_APP_ARN"

write_env() {   # written again after each layer, so it always holds what exists; env_written prints the count once
  local tmp key value
  for key in $ENV_KEYS TF_VAR_alert_email; do
    value="${!key:-}"
    case "$value" in
      *[\"\$\`\\]*|*$'\n'*) mlops_stop "the value of $key has a character .env.mlops cannot hold." ;;
    esac
  done
  tmp="$(mktemp "$MLOPS_ROOT/.env.mlops.XXXXXX")"
  {
    echo "# .env.mlops: written by scripts/create-env.sh, its only writer. Never edit it by hand, never commit it."
    echo "# Read by mlops_utils/config.py and the scripts. In a terminal: source .env.mlops"
    for key in $ENV_KEYS; do
      value="${!key:-}"
      [[ -n "$value" ]] || continue
      printf 'export %s="%s"\n' "$key" "$value"
    done
    # Terraform's inputs (K101 2026-11-01, option B): a new terminal or a VM restart needs only `source .env.mlops`
    echo "# Terraform reads these TF_VAR_* values (the same tags and suffix, and the alert email of MLOps 06)."
    [[ -n "${MLOPS_OWNER:-}" ]] && printf 'export TF_VAR_owner="%s"\n' "$MLOPS_OWNER"
    [[ -n "${MLOPS_PURPOSE:-}" ]] && printf 'export TF_VAR_purpose="%s"\n' "$MLOPS_PURPOSE"
    [[ -n "${MLOPS_CREATED:-}" ]] && printf 'export TF_VAR_created="%s"\n' "$MLOPS_CREATED"
    [[ -n "${MLOPS_SUFFIX:-}" ]] && printf 'export TF_VAR_suffix="%s"\n' "$MLOPS_SUFFIX"
    [[ -n "${TF_VAR_alert_email:-}" ]] && printf 'export TF_VAR_alert_email="%s"\n' "$TF_VAR_alert_email"
    true
  } > "$tmp"
  chmod 600 "$tmp"
  mv "$tmp" "$MLOPS_ENV_FILE"
}

env_written() {
  echo ".env.mlops written ($(grep -c '^export MLOPS_' "$MLOPS_ENV_FILE") variables, plus the TF_VAR_* lines for Terraform)"
  grep -q '^export TF_VAR_alert_email=' "$MLOPS_ENV_FILE" ||
    echo "TF_VAR_alert_email: not set, so no email alerts (set it as in MLOps 06 Step-03-01, then run this again)"
}

# ---------------------------------------------------------------------------------------------------------
# 5. The suffix (MLOPS_SUFFIX, made once by 00) and the tag values
# ---------------------------------------------------------------------------------------------------------
if [[ "$ENV_ONLY" -eq 0 ]] && wants 01; then
  code_is_committed                            # before any change on AWS (the tag names the code)
fi
SUFFIX_NOW=""
if [[ -d "$MLOPS_INFRA/00-state-bucket" ]]; then
  tf_init 00
  if tf_has_state 00; then SUFFIX_NOW="$(tf_out 00 suffix)"; fi
elif [[ -f "$MLOPS_INFRA/backend.hcl" ]]; then
  # A section folder from MLOps 07 on has no 00: its infra/backend.hcl (copied from your MLOps 06 folder) names the
  # state bucket ccfd-mlops-tfstate-<suffix>
  SUFFIX_NOW="$(sed -n 's/^bucket *= *"ccfd-mlops-tfstate-\([0-9a-f]\{6\}\)"$/\1/p' "$MLOPS_INFRA/backend.hcl")"
  [[ -n "$SUFFIX_NOW" ]] || mlops_stop "infra/backend.hcl names no ccfd-mlops-tfstate-<6 hex> bucket."
fi
if [[ -n "$SUFFIX_NOW" ]]; then
  # A .env.mlops of an environment that is gone (after destroy-env.sh --all): its values are not used again
  if [[ -n "${MLOPS_SUFFIX:-}" && "$MLOPS_SUFFIX" != "$SUFFIX_NOW" ]]; then
    if [[ "$ENV_ONLY" -eq 0 ]]; then
      mlops_stop ".env.mlops says MLOPS_SUFFIX=$MLOPS_SUFFIX but the state here holds $SUFFIX_NOW. Rebuild .env.mlops first: bash scripts/create-env.sh --write-env-only"
    fi
    echo "NOTE: .env.mlops said MLOPS_SUFFIX=$MLOPS_SUFFIX; the state here holds $SUFFIX_NOW: .env.mlops is rebuilt for $SUFFIX_NOW."
    MLOPS_JOBS_BUCKET="" MLOPS_EXECUTION_ROLE_ARN="" MLOPS_SERVING_IMAGE_URI="" MLOPS_TRAINING_IMAGE_URI="" MLOPS_MLFLOW_APP_ARN="" MLOPS_CREATED=""
  fi
  MLOPS_SUFFIX="$SUFFIX_NOW"
  # Tag values missing here (no .env.mlops yet, for example after 00 and 01 were applied by hand in MLOps 06): read
  # them from the state bucket's own tags (read only), so the next apply keeps the tags the environment has.
  if [[ -z "${MLOPS_OWNER:-}" || -z "${MLOPS_PURPOSE:-}" || -z "${MLOPS_CREATED:-}" ]]; then
    BUCKET_TAGS="$(aws s3api get-bucket-tagging --bucket "ccfd-mlops-tfstate-$MLOPS_SUFFIX" \
                   --query 'TagSet[].[Key,Value]' --output text 2>/dev/null || true)"
    while IFS=$'\t' read -r key value; do
      case "$key" in
        owner) MLOPS_OWNER="${MLOPS_OWNER:-$value}" ;;
        purpose) MLOPS_PURPOSE="${MLOPS_PURPOSE:-$value}" ;;
        created) MLOPS_CREATED="${MLOPS_CREATED:-$value}" ;;
      esac
    done <<< "$BUCKET_TAGS"
  fi
  MLOPS_CREATED="${MLOPS_CREATED:-${ENV_CREATED:-$(date -u +%Y-%m-%d)}}"
else
  if [[ "$ENV_ONLY" -eq 1 ]] || ! wants 00; then
    mlops_stop "infra/00-state-bucket has no state here: apply it first (bash scripts/create-env.sh --layers 00,01)."
  fi
  STATE_BUCKETS="$(aws s3api list-buckets --prefix ccfd-mlops-tfstate- --query 'Buckets[].Name' --output text)"
  if [[ -n "$STATE_BUCKETS" && "$STATE_BUCKETS" != "None" ]]; then
    mlops_stop "a course state bucket exists already ($STATE_BUCKETS), but this folder has no state for it. Run this in the folder that made it (your MLOps 06 folder), or remove that environment there first: bash scripts/destroy-env.sh --all"
  fi
  if [[ -n "${MLOPS_SUFFIX:-}" ]]; then
    echo "A new environment: the MLOPS_SUFFIX=$MLOPS_SUFFIX of .env.mlops belongs to one that is gone (destroy-env.sh --all)."
  fi
  MLOPS_SUFFIX=""
  MLOPS_JOBS_BUCKET="" MLOPS_EXECUTION_ROLE_ARN="" MLOPS_SERVING_IMAGE_URI="" MLOPS_TRAINING_IMAGE_URI="" MLOPS_MLFLOW_APP_ARN=""
  MLOPS_CREATED="${ENV_CREATED:-$(date -u +%Y-%m-%d)}"
fi

# The owner and purpose tags: from .env.mlops or the state bucket above, else the TF_VAR_* exports of MLOps 06
MLOPS_OWNER="${MLOPS_OWNER:-${TF_VAR_owner:-}}"
MLOPS_PURPOSE="${MLOPS_PURPOSE:-${TF_VAR_purpose:-}}"
if [[ -z "$MLOPS_OWNER" || -z "$MLOPS_PURPOSE" ]]; then
  mlops_stop "the owner and purpose tags are not set. Set them first, as in MLOps 06: export TF_VAR_owner=\"<your name>\" TF_VAR_purpose=mlops-course"
fi
export MLOPS_MODEL_PACKAGE_GROUP="$MLOPS_GROUP"

export TF_VAR_owner="$MLOPS_OWNER" TF_VAR_purpose="$MLOPS_PURPOSE" TF_VAR_created="$MLOPS_CREATED"
export TF_VAR_alert_email="${ALERT_EMAIL:-${TF_VAR_alert_email:-}}"
[[ -n "$MLOPS_SUFFIX" ]] && export TF_VAR_suffix="$MLOPS_SUFFIX"

# ---------------------------------------------------------------------------------------------------------
# 6. The steps of each layer
# ---------------------------------------------------------------------------------------------------------
apply_layer() {   # apply_layer <id>: init and apply, printing Terraform's output
  local id="$1" note=""
  [[ "$id" == "02" ]] && note=" (about 5 minutes the first time: the MLflow App)"
  mlops_step "Layer $id: infra/$(layer_dir "$id")$note"
  [[ "$id" == "04" ]] && echo "create_github_oidc_provider = ${TF_VAR_create_github_oidc_provider:-false}"
  tf_init "$id"
  "$TF" -chdir="$MLOPS_INFRA/$(layer_dir "$id")" apply -input=false -auto-approve -no-color \
    || mlops_stop "terraform apply failed in infra/$(layer_dir "$id") (see above)."
}

read_01_outputs() {
  local purpose name
  for purpose in dvc mlflow jobs; do
    name="$(tf_out 01 "${purpose}_bucket_name")"
    [[ "$name" == "ccfd-mlops-${purpose}-${MLOPS_SUFFIX}" ]] || mlops_stop "01's $purpose bucket is $name, not ccfd-mlops-${purpose}-${MLOPS_SUFFIX}."
  done
  MLOPS_JOBS_BUCKET="ccfd-mlops-jobs-${MLOPS_SUFFIX}"
  MLOPS_EXECUTION_ROLE_ARN="$(tf_out 01 execution_role_arn)"
}

# MLOPS_MLFLOW_APP_ARN when 02 is not applied in this run (for example --layers 01 in MLOps 10): 02's state in the
# state bucket is the truth (MLOps 07 makes the App, destroy-env.sh --all deletes it), so a new .env.mlops gets the ARN
# and an ARN of a deleted App is not written again. A folder without infra/02-mlflow-app (MLOps 06) has no App ARN.
APP_ARN_RE='^arn:aws:sagemaker:us-east-1:[0-9]{12}:mlflow-app/app-[A-Za-z0-9-]+$'
read_02_arn() {
  [[ -d "$MLOPS_INFRA/02-mlflow-app" && -f "$MLOPS_INFRA/backend.hcl" ]] || return 0
  tf_init 02
  if tf_has_state 02; then
    MLOPS_MLFLOW_APP_ARN="$(tf_out 02 mlflow_app_arn)"
    [[ "$MLOPS_MLFLOW_APP_ARN" =~ $APP_ARN_RE ]] || mlops_stop "02's output mlflow_app_arn is not an MLflow App ARN."
  else
    MLOPS_MLFLOW_APP_ARN=""
  fi
}

image_tag_exists() {   # image_tag_exists <repository> <tag>
  aws ecr describe-images --region "$MLOPS_REGION" --repository-name "$1" --image-ids "imageTag=$2" >/dev/null 2>&1
}

newest_image_tag() {   # newest_image_tag <repository>: the tag of the newest pushed image ("" when the repository is empty)
  local tag
  tag="$(aws ecr describe-images --region "$MLOPS_REGION" --repository-name "$1" --filter tagStatus=TAGGED \
    --query 'sort_by(imageDetails,&imagePushedAt)[-1].imageTags[0]' --output text 2>/dev/null || true)"
  [[ "$tag" == "None" ]] && tag=""
  printf '%s' "$tag"
}

push_images() {   # linux/amd64 and a Docker v2 manifest (SageMaker refuses an OCI index), built on your EC2 VM
  local tag="sha-$IMAGE_SHA" serving_url training_url repo dockerfile url key media
  mlops_step "Images in ECR: ccfd-serving and ccfd-training, tag $tag"
  serving_url="$(tf_out 01 serving_repository_url)"
  training_url="$(tf_out 01 training_repository_url)"
  for item in "ccfd-serving|Dockerfile.serving|$serving_url|MLOPS_SERVING_IMAGE_URI" \
              "ccfd-training|Dockerfile.training|$training_url|MLOPS_TRAINING_IMAGE_URI"; do
    IFS='|' read -r repo dockerfile url key <<< "$item"
    # A section folder holds only the Dockerfiles it uses (Dockerfile.training from MLOps 08 on)
    if [[ ! -f "$MLOPS_PROJECT/$dockerfile" ]]; then
      echo "$repo: not built here (this section's ccfd-project has no $dockerfile; a later section builds it)"
      printf -v "$key" '%s' ""
      continue
    fi
    if image_tag_exists "$repo" "$tag"; then
      echo "$repo:$tag is in ECR already (kept)"
    else
      on_course_vm || mlops_stop "$repo:$tag is not in ECR yet, and the images are built on your EC2 VM: run this there."
      mlops_build_push_local "$MLOPS_PROJECT" "$IMAGE_SHA" "$dockerfile" "$url:$tag" \
        || mlops_stop "the image build on your EC2 VM failed (above). Fix it, then run this again."
    fi
    media="$(aws ecr describe-images --region "$MLOPS_REGION" --repository-name "$repo" --image-ids "imageTag=$tag" \
             --query 'imageDetails[0].imageManifestMediaType' --output text)"
    if [[ "$media" != "$V2_MANIFEST" ]]; then
      mlops_stop "$repo:$tag has the manifest type $media, not $V2_MANIFEST: SageMaker refuses it. Delete that tag (aws ecr batch-delete-image) and run this again."
    fi
    printf -v "$key" '%s' "$url:$tag"
  done
  export MLOPS_SERVING_IMAGE_URI MLOPS_TRAINING_IMAGE_URI
}

upload_inputs() {   # the gate holdout and the base project's fixed YAML, md5-checked, in the jobs bucket (MLOps 08, 13)
  local cfg_file cfg_want cfg_got gate_key cfg_key
  mlops_step "Inputs in the jobs bucket: the gate holdout and the base project's fixed YAML"
  mlops_ensure_gate_holdout
  cfg_file="$MLOPS_ROOT/$(mlops_manifest_get config.file)"
  cfg_want="$(mlops_manifest_get config.md5)"
  [[ -f "$cfg_file" ]] || mlops_stop "The base project's fixed YAML is missing: $(mlops_manifest_get config.file) (it is committed)."
  cfg_got="$(mlops_md5 "$cfg_file")"
  [[ "$cfg_got" == "$cfg_want" ]] || mlops_stop "$(mlops_manifest_get config.file) has md5 $cfg_got, manifest.yaml says $cfg_want: restore it from git (git checkout -- that file)."
  gate_key="$(mlops_manifest_get gate_holdout.s3_key)"
  cfg_key="$(mlops_manifest_get config.s3_key)"
  aws s3 cp --region "$MLOPS_REGION" --only-show-errors "$MLOPS_PROJECT/data/gate_holdout.csv" "s3://$MLOPS_JOBS_BUCKET/$gate_key"
  aws s3 cp --region "$MLOPS_REGION" --only-show-errors "$cfg_file" "s3://$MLOPS_JOBS_BUCKET/$cfg_key"
  echo "uploaded: s3://$MLOPS_JOBS_BUCKET/$gate_key and s3://$MLOPS_JOBS_BUCKET/$cfg_key"
}

# After 02 is applied: AWS picks the App's MLflow version (there is no setting for it). This course's client is
# MLFLOW_PIN; it works with a 3.x App at that version or newer. Silent when the App is right; a STOP otherwise.
app_version_ok() {   # app_version_ok <version>: MLFLOW_PIN or a newer 3.x
  [[ "$1" =~ ^3\.[0-9]+\.[0-9]+$ ]] \
    && [[ "$(printf '%s\n%s\n' "$MLFLOW_PIN" "$1" | sort -V | head -n 1)" == "$MLFLOW_PIN" ]]
}

app_check() {
  local out version mode
  out="$(aws sagemaker describe-mlflow-app --region "$MLOPS_REGION" --arn "$MLOPS_MLFLOW_APP_ARN" \
         --query '[MlflowVersion,ModelRegistrationMode]' --output text)" \
    || mlops_stop "aws sagemaker describe-mlflow-app failed (above)."
  read -r version mode <<< "$out"
  if [[ "$mode" != "$APP_MODE" ]]; then
    mlops_stop "the MLflow App $APP_NAME has the registration mode $mode, not $APP_MODE (layer 02 sets it). Do not log runs to it: run this again."
  fi
  if ! app_version_ok "$version"; then
    mlops_stop "the MLflow App $APP_NAME runs MLflow $version, but this course's client is $MLFLOW_PIN and needs a 3.x App at $MLFLOW_PIN or newer. Stop here: do not log runs to this App."
  fi
}

NEW_SUBSCRIPTION=0
check_03_email() {   # 03 holds your email subscription: a run without your address would delete it
  local subscribed=""
  if tf_has_state 03; then
    subscribed="$(tf_out 03 alert_email_subscribed 2>/dev/null || true)"
  fi
  if [[ -z "$TF_VAR_alert_email" && "$subscribed" == "true" ]]; then
    mlops_stop "layer 03 has your email subscription. Set the same address first (export TF_VAR_alert_email=\"<your email address>\"), or Terraform would delete the subscription."
  fi
  if [[ -n "$TF_VAR_alert_email" && "$subscribed" != "true" ]]; then
    NEW_SUBSCRIPTION=1
  fi
}

check_04_provider() {   # 04 made the GitHub OIDC provider: a run with false would delete it
  local list
  [[ "${TF_VAR_create_github_oidc_provider:-false}" == "true" ]] && return 0
  [[ -d "$MLOPS_INFRA/04-cicd-oidc" ]] || return 0
  list="$(mlops_tf_state "$MLOPS_INFRA/04-cicd-oidc")" || exit 1
  if printf '%s\n' "$list" | grep -qxF 'aws_iam_openid_connect_provider.github[0]'; then
    mlops_stop "layer 04 made the GitHub OIDC provider. Set export TF_VAR_create_github_oidc_provider=true first, or Terraform would delete it."
  fi
}

# ---------------------------------------------------------------------------------------------------------
# 7. --write-env-only: read what exists, write .env.mlops, change nothing
# ---------------------------------------------------------------------------------------------------------
if [[ "$ENV_ONLY" -eq 1 ]]; then
  mlops_step "Reading the outputs of the layers that exist"
  if [[ -d "$MLOPS_INFRA/01-foundation" ]]; then   # a section folder holds only the layers it teaches
    tf_init 01
  fi
  if tf_has_state 01; then
    read_01_outputs
    sha="$(git -C "$MLOPS_ROOT" rev-parse --short=7 HEAD 2>/dev/null || true)"
    for repo in ccfd-serving ccfd-training; do
      key=MLOPS_SERVING_IMAGE_URI
      [[ "$repo" == "ccfd-training" ]] && key=MLOPS_TRAINING_IMAGE_URI
      url="$(tf_out 01 "${repo#ccfd-}_repository_url")"
      if [[ -n "$sha" ]] && image_tag_exists "$repo" "sha-$sha"; then
        printf -v "$key" '%s' "$url:sha-$sha"
      elif [[ -n "${!key:-}" ]] && image_tag_exists "$repo" "${!key##*:}"; then
        :                                            # the image of .env.mlops is still in ECR: kept
      elif newest="$(newest_image_tag "$repo")" && [[ -n "$newest" ]]; then
        # The clone moved on (pull-latest.sh) since the image was built: keep the newest image in ECR (K101 2026-11-01)
        printf -v "$key" '%s' "$url:$newest"
        echo "$repo: kept the newest image in ECR, $newest (built from an earlier version of the course code)"
      else
        printf -v "$key" '%s' ""
        echo "no $repo image in ECR yet: bash scripts/create-env.sh --layers 01 builds and pushes it"
      fi
    done
  else
    echo "01-foundation: no state (no bucket, role or image values)"
  fi
  read_02_arn
  write_env
  env_written
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------
# 8. The layers asked for, in order
# ---------------------------------------------------------------------------------------------------------
# The ARN of an App that already exists, read FIRST: .env.mlops is written again after every layer, so a run stopped
# between layer 01 and layer 02 (Ctrl+C during the image build) must not leave a file without MLOPS_MLFLOW_APP_ARN.
# Layer 02, when asked for, sets it again from its own apply below.
read_02_arn
for id in $WANT; do
  case "$id" in
    00)
      apply_layer 00
      MLOPS_SUFFIX="$(tf_out 00 suffix)"
      [[ "$MLOPS_SUFFIX" =~ ^[0-9a-f]{6}$ ]] || mlops_stop "00's output suffix is not 6 hex characters."
      export TF_VAR_suffix="$MLOPS_SUFFIX"
      write_env
      ;;
    01)
      apply_layer 01
      read_01_outputs
      write_env
      push_images
      write_env
      upload_inputs
      ;;
    02)
      tf_init 01
      tf_has_state 01 || mlops_stop "02 needs 01-foundation (its mlflow bucket and execution role): --layers 01,02"
      apply_layer 02
      MLOPS_MLFLOW_APP_ARN="$(tf_out 02 mlflow_app_arn)"
      [[ "$MLOPS_MLFLOW_APP_ARN" =~ $APP_ARN_RE ]] || mlops_stop "02's output mlflow_app_arn is not an MLflow App ARN."
      write_env
      app_check
      ;;
    03)
      tf_init 03
      check_03_email
      apply_layer 03
      write_env
      if [[ "$NEW_SUBSCRIPTION" -eq 1 ]]; then
        echo "AWS emails a confirmation link to your alert address: open it on your laptop and click it yourself."
      fi
      ;;
    04)
      tf_init 04
      check_04_provider
      apply_layer 04
      write_env
      ;;
    05)
      apply_layer 05
      write_env
      ;;
  esac
done

mlops_step "Done"
echo "Layers applied: $WANT"
env_written
