#!/usr/bin/env bash
# shellcheck disable=SC2016  # the --query strings are JMESPath: their `...` literals stay in single quotes
# destroy-env.sh: end an AWS session (the default), or remove the whole course environment (--all).
# Then run bash scripts/orphan-sweep.sh (with --all after --all).
#
#   bash scripts/destroy-env.sh          end of a session: delete only what bills by the hour; keep the rest
#   bash scripts/destroy-env.sh --all    leaving the course (MLOps 16): delete everything, the state bucket too
#
# End of a session, steps 1-5: the ccfd- endpoints with their autoscaling, configs and models; running executions
# of the ccfd- pipelines and running jobs tagged course=mlops-on-aws (stopped, then waited for); the rollout alarms,
# the (B) Lambda proxy with its role and log group, and the (B) HTTP API (both front an endpoint). Everything else
# has no hourly charge and stays, like a real team's: the data in the buckets, the ECR images, the MLflow App with
# its runs, the model registry, the pipelines, the endpoint log groups, the (B) Glue database and every Terraform
# layer. The last lines say so in one line.
# --all: steps 1-5 (step 5 also deletes the (B) Glue database), then 6 the ccfd- pipelines, 7 the model packages,
# 8 wait until the group is empty, 9 the group, 10 the 3 buckets emptied, 11 terraform destroy 06 -> 01 (the
# buckets emptied again right before 01), 12 the state bucket (every version) and layer 00, only in the folder that
# holds 00 (your MLOps 06 folder) and only when step 11 left every layer empty, 13 the endpoint log groups (last: a
# deleted endpoint's instances still write to its log group for a few minutes).
# Only names that start with ccfd- (the group: only ccfd-fraud-detector) and jobs with the course tag are touched;
# anything else that shows up is listed as "left alone". SageMaker's lineage and experiment records are free and
# stay. An AWS error is never read as "nothing to delete": the script stops.
# --all refuses at once (nothing deleted) while a layer this folder does not hold still has resources: destroy that
# layer first from the section folder that holds it. In a folder without layer 00 the last line says where the
# state bucket goes: your MLOps 06 folder.
# Exit 0 = everything listed is gone; exit 1 = a stop, or a teardown that did not finish (the last lines say what
# is left). Run it again after a stop: every step skips what is gone.
# Every line it prints goes through the account ID mask.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_masked "$0" "$@"

ALL=0
YES=0
for arg in "$@"; do
  case "$arg" in
    --all) ALL=1 ;;
    --yes) YES=1 ;;                           # no question (a run without a terminal)
    -h|--help) sed -n '3,28p' "$0"; exit 0 ;;
    *) echo "Usage: bash scripts/destroy-env.sh [--all]"; exit 2 ;;
  esac
done
GROUP_NAME="ccfd-fraud-detector"
LAMBDA="ccfd-inference-proxy"                 # MLOps 11 (B)
LAMBDA_ROLE="ccfd-inference-proxy-role"
HTTP_API="ccfd-inference-api"
GLUE_DB="ccfd_monitoring"                     # MLOps 15 Step-06 (B)

mlops_paths
mlops_read_env
STATE_BUCKET_TEXT="ccfd-mlops-tfstate-${MLOPS_SUFFIX:-<MLOPS_SUFFIX>}"

# ---------------------------------------------------------------------------------------------------------
# The list first: what goes
# ---------------------------------------------------------------------------------------------------------
if [[ "$ALL" -eq 1 ]]; then
  echo "What goes (--all, leaving the course): everything the course made, in this order: the ccfd- endpoints with"
  echo "    their configs and models; running pipeline executions and course jobs (stopped); the rollout alarms, the (B)"
  echo "    Lambda, HTTP API and Glue database; the ccfd- pipelines; the model registry $GROUP_NAME; the data in the"
  echo "    dvc, mlflow and jobs buckets; the Terraform layers 06 -> 01 of this folder; the state bucket and layer 00"
  echo "    (only in your MLOps 06 folder); the endpoint log groups. Anything without the ccfd- name or the course tag"
  echo "    is never touched."
else
  echo "What goes (end of a session): only what bills by the hour: the ccfd- endpoints with their configs and models,"
  echo "    running pipeline executions and course jobs (stopped), the rollout alarms, the (B) Lambda and HTTP API."
  echo "    Everything else has no hourly charge and stays; --all deletes it."
fi

mlops_need aws
mlops_find_python
mlops_r11_check
# --all needs Terraform 1.15.x (steps 11 and 12): checked before anything is deleted. The end of a session never
# runs Terraform: every layer stays.
TF=""
if [[ "$ALL" -eq 1 ]]; then
  mlops_find_terraform
fi

# --all from a section folder: every layer this folder does not hold must already be empty (its state in the state
# bucket lists no resource). Else nothing is deleted: a lower layer must never go from under a layer that uses it
# (the App uses 01's bucket and role), and step 10 would empty buckets it still uses.
if [[ "$ALL" -eq 1 && -n "${MLOPS_SUFFIX:-}" ]]; then
  OTHER_LEFT=""
  for dir in 06-loop 05-monitoring 04-cicd-oidc 03-registry-eventing 02-mlflow-app 01-foundation; do
    [[ -d "$MLOPS_INFRA/$dir" ]] && continue
    n="$(aws s3 cp "s3://$STATE_BUCKET_TEXT/mlops/$dir/terraform.tfstate" - --region "$MLOPS_REGION" 2>/dev/null \
         | "$PY" -c 'import json,sys; print(len(json.load(sys.stdin).get("resources", [])))' 2>/dev/null || echo 0)"
    [[ "$n" == "0" ]] || OTHER_LEFT="$OTHER_LEFT $dir"
  done
  if [[ -n "$OTHER_LEFT" ]]; then
    mlops_stop "--all refused, nothing was deleted: layer(s)$OTHER_LEFT still hold resources and are not in this folder. Destroy them first from the section folder that holds them, then run this again."
  fi
fi

if [[ "$YES" -eq 0 ]]; then
  [[ -t 0 ]] || mlops_stop "not a terminal: add --yes to delete (the lines above say what goes)."
  echo "Type yes and press Enter to delete what the lines above list (anything else stops):"
  read -r answer
  [[ "$answer" == "yes" ]] || { echo "Stopped: nothing was deleted."; exit 1; }
fi

START=$SECONDS
PARTIAL=""                                    # what could not be deleted; non-empty = exit 1 at the end
not_done() { PARTIAL="${PARTIAL:+$PARTIAL; }$*"; }

# del <command ...>: print a delete or stop command, then run it
del() {
  echo "  $*"
  "$@"
}

# del_gone <aws delete command ...>: like del, for steps 2 and 3. A list right after step 1 can still show a config
# or model that step 1 deleted (for some seconds); its delete then answers "Could not find": that name is gone
# already, so one line says so and the script goes on. Any other error still stops the script.
del_gone() {
  local out
  echo "  $*"
  if out="$("$@" 2>&1)"; then
    if [[ -n "$out" ]]; then echo "$out"; fi
    return 0
  fi
  if grep -q 'Could not find' <<< "$out"; then
    echo "    gone already (step 1 deleted it; the list was a few seconds behind)"
    return 0
  fi
  echo "$out" >&2
  return 1
}

# items <command ...>: the --output text items of a read-only command, one per line (tabs split, None dropped).
# Always used as  X="$(items ...)" : when the AWS call fails, the script stops there (set -e), so an error is
# never read as "nothing to delete".
items() {
  local out
  out="$("$@")" || return 1
  printf '%s\n' "$out" | tr '\t' '\n' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
    | { grep -v -e '^$' -e '^None$' || true; }
}

# take_ours <what> <names>: OURS = the names that start with ccfd-; every other name is listed and left alone
take_ours() {
  local name
  OURS=""
  for name in $2; do
    case "$name" in
      ccfd-*) OURS="${OURS:+$OURS }$name" ;;
      *) echo "  $1 $name: the name does not start with ccfd-, so it is not the course's: left alone, report it" ;;
    esac
  done
}

# found <not-found code> <command ...>: 0 when the thing exists, 1 when AWS answers with <not-found code>
# (for example ResourceNotFoundException); any other error stops the script (never read as "nothing there")
found() {
  local code="$1" err
  shift
  if err="$("$@" 2>&1 > /dev/null)"; then
    return 0
  fi
  case "$err" in
    *"($code)"*) return 1 ;;
  esac
  printf '%s\n' "$err" | head -n 5
  mlops_stop "$1 $2 $3 failed (above): it could not be checked, so nothing is assumed."
}

sm() { aws sagemaker "$@" --region "$MLOPS_REGION" --output text; }
R=(--region "$MLOPS_REGION" --output text)

none_or() {   # none_or <list>: "none" when the list is empty
  if [[ -z "$1" ]]; then echo "  none"; fi
}

endpoints_py() {   # the endpoint helper of mlops_utils
  (cd "$MLOPS_PROJECT" && "$PY" -m mlops_utils.endpoints "$@") | sed 's/^/  /'
}

# ---------------------------------------------------------------------------------------------------------
# 1-3. Endpoints, then their configs and models
# ---------------------------------------------------------------------------------------------------------
mlops_step "1. Endpoints named ccfd- (autoscaling first; each with its configs and models)"
LISTED="$(items sm list-endpoints --name-contains ccfd --query 'Endpoints[].EndpointName')"
take_ours endpoint "$LISTED"
ENDPOINTS="$OURS"
none_or "$ENDPOINTS"
for name in $ENDPOINTS; do
  endpoints_py delete --endpoint "$name"
done

mlops_step "2. Endpoint configs named ccfd- (what is left)"
LISTED="$(items sm list-endpoint-configs --name-contains ccfd --query 'EndpointConfigs[].EndpointConfigName')"
take_ours "endpoint config" "$LISTED"
CONFIGS="$OURS"
none_or "$CONFIGS"
for name in $CONFIGS; do
  del_gone aws sagemaker delete-endpoint-config --region "$MLOPS_REGION" --endpoint-config-name "$name"
done

mlops_step "3. Models named ccfd- (what is left)"
LISTED="$(items sm list-models --name-contains ccfd --query 'Models[].ModelName')"
take_ours model "$LISTED"
MODELS="$OURS"
none_or "$MODELS"
for name in $MODELS; do
  del_gone aws sagemaker delete-model --region "$MLOPS_REGION" --model-name "$name"
done

# ---------------------------------------------------------------------------------------------------------
# 4. Stop what runs: ccfd- pipeline executions and jobs tagged course=mlops-on-aws; wait until Stopped
# ---------------------------------------------------------------------------------------------------------
mlops_step "4. Running pipeline executions (ccfd- pipelines) and running jobs tagged course=$MLOPS_COURSE_TAG: stop, then wait"
LISTED="$(items sm list-pipelines --pipeline-name-prefix ccfd --query 'PipelineSummaries[].PipelineName')"
take_ours pipeline "$LISTED"
PIPELINES="$OURS"
STOPPED=""
for pipeline in $PIPELINES; do
  RUNNING="$(items sm list-pipeline-executions --pipeline-name "$pipeline" \
             --query 'PipelineExecutionSummaries[?PipelineExecutionStatus==`Executing`].PipelineExecutionArn')"
  for arn in $RUNNING; do
    del aws sagemaker stop-pipeline-execution --region "$MLOPS_REGION" --pipeline-execution-arn "$arn"
    STOPPED="$STOPPED pipeline|$arn"
  done
done
for kind in training processing transform; do
  case "$kind" in
    training) summaries="TrainingJobSummaries[].TrainingJobArn" ;;
    processing) summaries="ProcessingJobSummaries[].ProcessingJobArn" ;;
    transform) summaries="TransformJobSummaries[].TransformJobArn" ;;
  esac
  RUNNING="$(items sm "list-$kind-jobs" --status-equals InProgress --query "$summaries")"
  for arn in $RUNNING; do
    tagged="$(items sm list-tags --resource-arn "$arn" --query "Tags[?Key=='course' && Value=='$MLOPS_COURSE_TAG'].Key")"
    if [[ -z "$tagged" ]]; then
      echo "  running $kind job ${arn##*/}: not tagged course=$MLOPS_COURSE_TAG, not the course's: left alone, report it"
      continue
    fi
    del aws sagemaker "stop-$kind-job" --region "$MLOPS_REGION" "--$kind-job-name" "${arn##*/}"
    STOPPED="$STOPPED $kind|${arn##*/}"
  done
done
none_or "${STOPPED# }"
if [[ -n "$STOPPED" ]]; then
  deadline=$((SECONDS + 1800))
  for entry in $STOPPED; do
    kind="${entry%%|*}"
    id="${entry#*|}"
    while :; do
      case "$kind" in
        pipeline) status="$(sm describe-pipeline-execution --pipeline-execution-arn "$id" --query PipelineExecutionStatus)" ;;
        training) status="$(sm describe-training-job --training-job-name "$id" --query TrainingJobStatus)" ;;
        processing) status="$(sm describe-processing-job --processing-job-name "$id" --query ProcessingJobStatus)" ;;
        transform) status="$(sm describe-transform-job --transform-job-name "$id" --query TransformJobStatus)" ;;
      esac
      case "$status" in
        Executing|InProgress|Stopping) ;;
        *) echo "  $kind ${id##*/}: $status"; break ;;
      esac
      [[ $SECONDS -lt $deadline ]] || mlops_stop "$kind ${id##*/} is still $status after 30 minutes: check it in the console."
      sleep 15
    done
  done
fi

# ---------------------------------------------------------------------------------------------------------
# 5. What Terraform does not know: what fronts an endpoint (and, only with --all, the (B) Glue database)
# ---------------------------------------------------------------------------------------------------------
if [[ "$ALL" -eq 1 ]]; then
  mlops_step "5. Objects made outside Terraform: the rollout alarms, the (B) Lambda and HTTP API, the (B) Glue database"
else
  mlops_step "5. Objects made outside Terraform: the rollout alarms, the (B) Lambda and HTTP API"
fi
endpoints_py delete-rollout-alarms
LAMBDA_FOUND=0
if found ResourceNotFoundException aws lambda get-function --region "$MLOPS_REGION" --function-name "$LAMBDA"; then
  del aws lambda delete-function --region "$MLOPS_REGION" --function-name "$LAMBDA"
  LAMBDA_FOUND=1
fi
if found NoSuchEntity aws iam get-role --role-name "$LAMBDA_ROLE"; then
  INLINE="$(items aws iam list-role-policies --role-name "$LAMBDA_ROLE" --query 'PolicyNames[]' --output text)"
  for policy in $INLINE; do
    del aws iam delete-role-policy --role-name "$LAMBDA_ROLE" --policy-name "$policy"
  done
  ATTACHED="$(items aws iam list-attached-role-policies --role-name "$LAMBDA_ROLE" --query 'AttachedPolicies[].PolicyArn' --output text)"
  for policy in $ATTACHED; do
    del aws iam detach-role-policy --role-name "$LAMBDA_ROLE" --policy-arn "$policy"
  done
  del aws iam delete-role --role-name "$LAMBDA_ROLE"
  LAMBDA_FOUND=1
fi
LISTED="$(items aws logs describe-log-groups "${R[@]}" --log-group-name-prefix "/aws/lambda/$LAMBDA" --query 'logGroups[].logGroupName')"
for group in $LISTED; do
  if [[ "$group" == "/aws/lambda/$LAMBDA" ]]; then      # only its own log group, never a longer name
    del aws logs delete-log-group --region "$MLOPS_REGION" --log-group-name "$group"
    LAMBDA_FOUND=1
  fi
done
[[ "$LAMBDA_FOUND" -eq 1 ]] || echo "  (B) Lambda $LAMBDA with its role and log group: none"
API_IDS="$(items aws apigatewayv2 get-apis "${R[@]}" --query "Items[?Name=='$HTTP_API'].ApiId")"
for api in $API_IDS; do
  del aws apigatewayv2 delete-api --region "$MLOPS_REGION" --api-id "$api"
done
[[ -n "$API_IDS" ]] || echo "  (B) HTTP API $HTTP_API: none"
if [[ "$ALL" -eq 1 ]]; then
  if found EntityNotFoundException aws glue get-database --region "$MLOPS_REGION" --name "$GLUE_DB"; then
    TABLES="$(items aws glue get-tables "${R[@]}" --database-name "$GLUE_DB" --query 'TableList[].Name')"
    if [[ -n "$TABLES" ]]; then
      # shellcheck disable=SC2086  # one argument per table name
      del aws glue batch-delete-table --region "$MLOPS_REGION" --database-name "$GLUE_DB" --tables-to-delete $TABLES
    fi
    del aws glue delete-database --region "$MLOPS_REGION" --name "$GLUE_DB"
  else
    echo "  (B) Glue database $GLUE_DB: none"
  fi
fi

# ---------------------------------------------------------------------------------------------------------
# The end of a session stops here: everything below has no hourly charge
# ---------------------------------------------------------------------------------------------------------
if [[ "$ALL" -eq 0 ]]; then
  echo
  echo "kept (no hourly charge; --all deletes them): your data, images, MLflow App, registry, pipelines, log groups, layers"
  echo "destroy-env.sh done in $(mlops_elapsed $((SECONDS - START)))."
  echo "Next: bash scripts/orphan-sweep.sh (it must end with '0 resources found'), then stop your EC2 VM."
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------
# 6-9 (--all). The pipelines, the packages, wait, the group (only ccfd-fraud-detector, the one group of the list)
# ---------------------------------------------------------------------------------------------------------
mlops_step "6. Pipelines named ccfd-"
none_or "$PIPELINES"
for pipeline in $PIPELINES; do
  del aws sagemaker delete-pipeline --region "$MLOPS_REGION" --pipeline-name "$pipeline"
done

mlops_step "7. Model packages in $GROUP_NAME"
LISTED="$(items sm list-model-package-groups --name-contains ccfd --query 'ModelPackageGroupSummaryList[].ModelPackageGroupName')"
GROUP_FOUND=""
for group in $LISTED; do
  if [[ "$group" == "$GROUP_NAME" ]]; then
    GROUP_FOUND="$group"
  else
    echo "  model package group $group: not $GROUP_NAME, the one group the list names: left alone, report it"
  fi
done
PACKAGES=""
for group in $GROUP_FOUND; do
  IN_GROUP="$(items sm list-model-packages --model-package-group-name "$group" --query 'ModelPackageSummaryList[].ModelPackageArn')"
  for arn in $IN_GROUP; do
    PACKAGES="$PACKAGES $arn"
    del aws sagemaker delete-model-package --region "$MLOPS_REGION" --model-package-name "$arn"
  done
done
none_or "${PACKAGES# }"

mlops_step "8. Wait until $GROUP_NAME is empty"
for group in $GROUP_FOUND; do
  deadline=$((SECONDS + 600))
  while :; do
    left="$(sm list-model-packages --model-package-group-name "$group" --query 'length(ModelPackageSummaryList)')"
    echo "  $group: $left package(s) left"
    [[ "$left" == "0" ]] && break
    [[ $SECONDS -lt $deadline ]] || mlops_stop "$group still holds $left package(s) after 10 minutes."
    sleep 10
  done
done
none_or "$GROUP_FOUND"

mlops_step "9. The model package group $GROUP_NAME"
none_or "$GROUP_FOUND"
for group in $GROUP_FOUND; do
  del aws sagemaker delete-model-package-group --region "$MLOPS_REGION" --model-package-group-name "$group"
done

# ---------------------------------------------------------------------------------------------------------
# 10 (--all). The 3 buckets emptied (they have no force_destroy)
# ---------------------------------------------------------------------------------------------------------
empty_buckets() {   # every object of the dvc, mlflow and jobs buckets of MLOPS_SUFFIX
  local purpose bucket count
  if [[ -z "${MLOPS_SUFFIX:-}" ]]; then
    echo "  no MLOPS_SUFFIX in .env.mlops: the bucket names are not known here (nothing emptied)"
    return 0
  fi
  for purpose in dvc mlflow jobs; do
    bucket="ccfd-mlops-${purpose}-${MLOPS_SUFFIX}"
    if ! found 404 aws s3api head-bucket --bucket "$bucket"; then
      echo "  $bucket: not there"
      continue
    fi
    # --output json: the CLI adds up every page (text output prints one number per page)
    count="$(aws s3api list-objects-v2 --bucket "$bucket" --query 'length(Contents || `[]`)' --output json)"
    echo "  $bucket: $count object(s)"
    if [[ "$count" != "0" ]]; then
      del aws s3 rm "s3://$bucket" --recursive --only-show-errors
    fi
  done
}

mlops_step "10. Empty the dvc, mlflow and jobs buckets (they have no force_destroy)"
empty_buckets

# ---------------------------------------------------------------------------------------------------------
# 11-12 (--all). Terraform: 06 -> 01, then the state bucket and 00 only when 06 -> 01 are all gone
# ---------------------------------------------------------------------------------------------------------
layer_dir() {
  case "$1" in
    00) echo 00-state-bucket ;; 01) echo 01-foundation ;; 02) echo 02-mlflow-app ;;
    03) echo 03-registry-eventing ;; 04) echo 04-cicd-oidc ;; 05) echo 05-monitoring ;; 06) echo 06-loop ;;
  esac
}

# Is anything Terraform-made known in this folder? infra/backend.hcl (00 writes it; 00's destroy removes it) or
# a 00 state that still lists resources (a state file that cannot be read counts as "yes": nothing is assumed).
TF_HERE=0
STATE00="$MLOPS_INFRA/00-state-bucket/terraform.tfstate"
if [[ -f "$MLOPS_INFRA/backend.hcl" ]] || { [[ -f "$STATE00" ]] && ! grep -q '"resources": \[\]' "$STATE00"; }; then
  TF_HERE=1
fi

TF_READY=1
if [[ -z "${MLOPS_OWNER:-}" || -z "${MLOPS_PURPOSE:-}" || -z "${MLOPS_CREATED:-}" || -z "${MLOPS_SUFFIX:-}" ]]; then
  TF_READY=0
  echo "NOTE: .env.mlops has no tag values or suffix (MLOPS_OWNER, MLOPS_PURPOSE, MLOPS_CREATED, MLOPS_SUFFIX): the Terraform layers"
  echo "      cannot be read or destroyed from here. bash scripts/create-env.sh --write-env-only rebuilds .env.mlops."
elif [[ ! -f "$MLOPS_INFRA/backend.hcl" ]]; then
  TF_READY=0
  echo "NOTE: no infra/backend.hcl here: layers 01 to 06 cannot be read from this folder."
fi
export TF_VAR_owner="${MLOPS_OWNER:-}" TF_VAR_purpose="${MLOPS_PURPOSE:-}" TF_VAR_created="${MLOPS_CREATED:-}"
export TF_VAR_suffix="${MLOPS_SUFFIX:-}"

tf_layer_init() {   # tf_layer_init <id>: init quietly (no change on AWS); 1 when it fails
  local dir log args=()
  dir="$(layer_dir "$1")"
  [[ "$1" == "00" ]] || args=(-reconfigure -backend-config=../backend.hcl)   # backend.hcl is the truth (new bucket after --all)
  log="$(mktemp)"
  if ! "$TF" -chdir="$MLOPS_INFRA/$dir" init -input=false -no-color -lockfile=readonly ${args[@]+"${args[@]}"} > "$log" 2>&1; then
    cat "$log"
    rm -f "$log"
    return 1
  fi
  rm -f "$log"
}

TF_LEFT=""                                    # layers that still list resources after their destroy
destroy_layer() {   # destroy_layer <id>
  local id="$1" dir resources left
  dir="$(layer_dir "$id")"
  if ! tf_layer_init "$id"; then
    mlops_stop "terraform init failed in infra/$dir (see above)."
  fi
  resources="$(mlops_tf_state "$MLOPS_INFRA/$dir")" || exit 1
  if [[ -z "$resources" ]]; then
    echo "  $id ($dir): nothing in its state"
    return 0
  fi
  echo "  $id ($dir): $(printf '%s\n' "$resources" | grep -c .) resource(s) in the state"
  if [[ "$id" == "01" ]]; then
    echo "  right before 01: the 3 buckets once more (a late capture file or MLflow artifact would stop 01)"
    empty_buckets
  fi
  "$TF" -chdir="$MLOPS_INFRA/$dir" destroy -input=false -auto-approve -no-color
  left="$(mlops_tf_state "$MLOPS_INFRA/$dir")" || exit 1
  left="$(printf '%s\n' "$left" | { grep -v -e '^data\.' -e '^$' || true; })"
  if [[ -n "$left" ]]; then
    echo "  $id ($dir): still in the state after the destroy:"
    printf '%s\n' "$left" | sed 's/^/    /'
    TF_LEFT="$TF_LEFT $id"
    not_done "layer $id still holds resources after terraform destroy"
  else
    echo "  $id ($dir): destroyed, the state is empty"
  fi
}

mlops_step "11. terraform destroy 06 -> 01 (the ECR repositories go with their images: force_delete)"
if [[ "$TF_READY" -eq 1 ]]; then
  NOT_HERE=""
  for id in 06 05 04 03 02 01; do
    dir="$(layer_dir "$id")"
    if [[ ! -d "$MLOPS_INFRA/$dir" ]]; then
      NOT_HERE="$NOT_HERE $dir"                 # empty: the check before "Type yes" refuses otherwise
      continue
    fi
    if [[ -n "$TF_LEFT" ]]; then
      # A layer above still holds resources that use this one (the App uses 01's bucket and role): never pull a
      # lower layer out from under it
      echo "  $dir: NOT destroyed: layer(s)$TF_LEFT above it still hold resources that use it"
      continue
    fi
    destroy_layer "$id"
  done
  [[ -z "$NOT_HERE" ]] || echo "  not in this folder (their state is empty):$NOT_HERE"
elif [[ "$TF_HERE" -eq 0 ]]; then
  echo "  nothing to destroy from this folder: no infra/backend.hcl and no resource in infra/00-state-bucket's state"
  echo "  (no layer was set up here, or destroy-env.sh --all removed them)"
else
  echo "  NOT destroyed: the layers set up from this folder stay in AWS (see the NOTE above)"
  not_done "the Terraform layers 06 -> 01 were NOT destroyed (see the NOTE before step 11)"
fi

mlops_step "12. The state bucket and layer 00"
if [[ ! -d "$MLOPS_INFRA/00-state-bucket" ]]; then
  # A section folder from MLOps 07 on holds no layer 00: the state bucket goes from the MLOps 06 folder, after the
  # layers of this folder (step 11) are empty there too
  echo "  layer 00 is not in this folder: $STATE_BUCKET_TEXT goes from your MLOps 06 folder (the last line)"
elif [[ "$TF_HERE" -eq 0 ]]; then
  echo "  nothing to remove from this folder: no infra/backend.hcl and no resource in infra/00-state-bucket's state"
elif [[ "$TF_READY" -eq 0 ]]; then
  mlops_stop "--all refused: step 11 did not run (see the NOTE before step 11). The state bucket $STATE_BUCKET_TEXT holds the state of layers 01-06; deleting it would leave them in AWS with no Terraform state. Nothing in it was deleted and 00 was not destroyed."
elif [[ -n "$TF_LEFT" ]]; then
  mlops_stop "--all refused: layer(s)$TF_LEFT still hold resources after step 11. Nothing in the state bucket $STATE_BUCKET_TEXT was deleted and 00 was not destroyed."
else
  tf_layer_init 00 || mlops_stop "terraform init failed in infra/00-state-bucket (see above)."
  STATE_00="$(mlops_tf_state "$MLOPS_INFRA/00-state-bucket")" || exit 1
  if [[ -z "$STATE_00" ]]; then
    echo "  00: nothing in its state"
  else
    echo "  step 11 left 06 -> 01 with an empty state: the state bucket can go"
    echo "  every object version and delete marker of $STATE_BUCKET_TEXT (no force_destroy on it)"
    "$PY" - "$STATE_BUCKET_TEXT" "$MLOPS_REGION" <<'PY'
import sys

import boto3

bucket, region = sys.argv[1], sys.argv[2]
s3 = boto3.client("s3", region_name=region)
found = []
for page in s3.get_paginator("list_object_versions").paginate(Bucket=bucket):
    for item in page.get("Versions", []) + page.get("DeleteMarkers", []):
        found.append({"Key": item["Key"], "VersionId": item["VersionId"]})
for start in range(0, len(found), 1000):
    answer = s3.delete_objects(Bucket=bucket, Delete={"Objects": found[start:start + 1000], "Quiet": True})
    if answer.get("Errors"):
        for error in answer["Errors"][:5]:
            print(f"  could not delete {error.get('Key')}: {error.get('Code')} {error.get('Message')}")
        sys.exit(f"  {len(answer['Errors'])} object version(s) of {bucket} could not be deleted (above).")
print(f"  {bucket}: {len(found)} object version(s) and delete marker(s) deleted")
PY
    destroy_layer 00
  fi
fi

# ---------------------------------------------------------------------------------------------------------
# 13 (--all). The endpoint log groups, once, at the very end: no hourly charge, so the end of a session keeps
# them. Last, because a deleted endpoint's instances still write to its log group for a few minutes (a late
# line makes a deleted group again); if the sweep lists one, run destroy-env.sh --all again.
# ---------------------------------------------------------------------------------------------------------
mlops_step "13. Log groups /aws/sagemaker/Endpoints/ccfd-* (never a shared group)"
LOGS="$(items aws logs describe-log-groups "${R[@]}" --log-group-name-prefix /aws/sagemaker/Endpoints/ccfd- --query 'logGroups[].logGroupName')"
none_or "$LOGS"
for group in $LOGS; do
  del aws logs delete-log-group --region "$MLOPS_REGION" --log-group-name "$group"
done

echo
if [[ -n "$PARTIAL" ]]; then
  echo "destroy-env.sh did NOT finish (in $(mlops_elapsed $((SECONDS - START)))): $PARTIAL."
  echo "Fix that and run it again (every step skips what is gone)."
  exit 1
fi
echo "destroy-env.sh done in $(mlops_elapsed $((SECONDS - START)))."
if [[ ! -d "$MLOPS_INFRA/00-state-bucket" ]]; then
  echo "Next: in your MLOps 06 folder, bash scripts/destroy-env.sh --all (the state bucket)"
else
  echo "Next: on your laptop bash scripts/vm/09-destroy-vm.sh, then bash scripts/orphan-sweep.sh --all (it must end with '0 resources found')."
fi
