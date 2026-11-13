#!/usr/bin/env bash
# shellcheck disable=SC2016  # the --query strings are JMESPath: their `...` literals stay in single quotes
# orphan-sweep.sh: at the end of every AWS session, list everything the course could have left behind in us-east-1.
# READ ONLY: it never creates, changes or deletes anything, and it costs nothing.
#
#   bash scripts/orphan-sweep.sh          after destroy-env.sh (the end of a session): only what bills by the hour
#                                         counts; what the course keeps between sessions is shown, not counted
#   bash scripts/orphan-sweep.sh --all    after destroy-env.sh --all and deleting the VM (then everything counts)
#
# It looks for: anything running in the account whatever its name (endpoints, training, processing and transform
# jobs); your EC2 VM with its security group, key pair, role and alarm; everything named ccfd (SageMaker, ECR, S3,
# IAM roles, log groups, CloudWatch, EventBridge, SNS, Lambda, API Gateway, Glue); with --all also every live
# resource tagged course=mlops-on-aws. Left out: SageMaker's lineage and experiment records and finished job records
# (free: they never bill). Each line prints its count and its items, one per line, every 12-digit number masked.
#
# Kept between sessions, unless --all: what has no hourly charge, by its known course name, shown on its line as
# "(n kept between sessions)" and not counted: the state bucket and the dvc, jobs and mlflow buckets of MLOPS_SUFFIX,
# the ECR repositories, the Terraform roles, the MLflow App, the model registry, the ccfd- pipelines, the endpoint log
# groups, the 2 monitoring alarms, the dashboard, the 2 EventBridge rules, the SNS topic, the (B) Glue database, and
# your EC2 VM with its parts (only while exactly one mlops-vm exists). Any other ccfd name counts.
#
# The end: '0 resources found' and exit 0, or '<n> resources found' and exit 1.
# A course orphan is deleted (destroy-env.sh, or by hand); anything else it shows is not the course's: report it,
# never touch it.
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=mlops_common.sh
. "$(dirname "$0")/mlops_common.sh"
mlops_masked "$0" "$@"

ALL=0
case "${1:-}" in
  "") ;;
  --all) ALL=1 ;;
  -h|--help) sed -n '3,24p' "$0"; exit 0 ;;
  *) echo "Usage: bash scripts/orphan-sweep.sh [--all]"; exit 2 ;;
esac

mlops_paths
mlops_read_env
mlops_need aws

echo "orphan-sweep.sh: read only, no cost; region $MLOPS_REGION; key ccfd; tag course=$MLOPS_COURSE_TAG"
mlops_r11_check

TOTAL=0
S="${MLOPS_SUFFIX:-no-suffix}"                    # the bucket names of this .env.mlops

# show <label> <command ...>: run a read-only command; split --output text on tabs; drop empty and None lines;
# without --all, take out what matches KEEP_GLOBS (kept between sessions); print the count and the items; add the
# count to TOTAL. An AWS error is counted too (a line we could not check is never a pass).
show() {
  local label="$1" out item count=0 lines="" kept=0 pat
  local -a pats=()
  [[ "$ALL" -eq 0 ]] && read -r -a pats <<< "${KEEP_GLOBS:-}"
  shift
  out="$("$@" 2>&1 | tr '\t' '\n' | mlops_mask | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        | grep -v '^$' | grep -v '^None$')"
  if [[ -n "$out" ]]; then
    while IFS= read -r item; do
      for pat in ${pats[@]+"${pats[@]}"}; do
        # shellcheck disable=SC2053  # the right side is a glob pattern on purpose
        if [[ "$item" == $pat ]]; then
          kept=$((kept + 1))
          continue 2
        fi
      done
      lines="${lines}    ${item}"$'\n'
      count=$((count + 1))
    done <<< "$out"
  fi
  if [[ "$kept" -gt 0 ]]; then
    printf '%-54s %s   (%s kept between sessions)\n' "$label" "$count" "$kept"
  else
    printf '%-54s %s\n' "$label" "$count"
  fi
  [[ "$count" -gt 0 ]] && printf '%s' "$lines"
  TOTAL=$((TOTAL + count))
}

# shellcheck disable=SC2329  # sm is called through show
sm() { aws sagemaker "$@" --region "$MLOPS_REGION" --output text; }
R=(--region "$MLOPS_REGION" --output text)
VM_FILTERS=(--filters "Name=tag:Name,Values=mlops-vm" "Name=instance-state-name,Values=pending,running,stopping,stopped")

mlops_step "Running anything, whatever its name (endpoints and running jobs bill by the hour)"
show "endpoints (any name)"                  sm list-endpoints --query 'Endpoints[].EndpointName'
show "training jobs InProgress (any name)"   sm list-training-jobs --status-equals InProgress --query 'TrainingJobSummaries[].TrainingJobName'
show "processing jobs InProgress (any name)" sm list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName'
show "transform jobs InProgress (any name)"  sm list-transform-jobs --status-equals InProgress --query 'TransformJobSummaries[].TransformJobName'

# Your EC2 VM (Section 00-B Step-01) and its 5 parts, on one line: the instance, the security group, the key pair,
# the instance profile, the role and the auto-stop alarm. Kept between sessions unless --all, while exactly one
# mlops-vm exists (running or stopped); else every part counts.
# shellcheck disable=SC2329  # vm_parts is called through show
vm_parts() {
  aws ec2 describe-instances "${R[@]}" "${VM_FILTERS[@]}" --query 'Reservations[].Instances[].InstanceId'
  aws ec2 describe-security-groups "${R[@]}" --filters Name=group-name,Values=mlops-vm-sg --query 'SecurityGroups[].GroupId'
  aws ec2 describe-key-pairs "${R[@]}" --filters Name=key-name,Values=mlops-vm-key --query 'KeyPairs[].KeyName'
  aws iam list-instance-profiles --query 'InstanceProfiles[?InstanceProfileName==`mlops-vm-role`].Arn' --output text
  aws iam list-roles --query 'Roles[?RoleName==`mlops-vm-role`].Arn' --output text
  aws cloudwatch describe-alarms "${R[@]}" --alarm-names mlops-vm-auto-stop --query 'MetricAlarms[].AlarmName'
}
VM_KEEP=""
if [[ "$ALL" -eq 0 ]]; then
  VM_COUNT="$(aws ec2 describe-instances "${R[@]}" "${VM_FILTERS[@]}" --query 'length(Reservations[].Instances[])' 2>/dev/null)"
  if [[ "$VM_COUNT" == "1" ]]; then
    VM_KEEP="i-* sg-* mlops-vm-key arn:aws:iam::*:instance-profile/mlops-vm-role arn:aws:iam::*:role/mlops-vm-role mlops-vm-auto-stop"
  fi
fi
mlops_step "Your EC2 VM (scripts/vm/09-destroy-vm.sh deletes it at the full exit)"
KEEP_GLOBS="$VM_KEEP" \
  show "EC2 VM mlops-vm and its parts"         vm_parts

mlops_step "SageMaker objects named ccfd"
KEEP_GLOBS="ccfd-mlflow" \
  show "MLflow Apps (Status not Deleted)"      sm list-mlflow-apps --query 'Summaries[?Status!=`Deleted`].Name'
show "endpoint configs ccfd"                 sm list-endpoint-configs --name-contains ccfd --query 'EndpointConfigs[].EndpointConfigName'
show "models ccfd"                           sm list-models --name-contains ccfd --query 'Models[].ModelName'
KEEP_GLOBS="ccfd-*" \
  show "pipelines ccfd"                        sm list-pipelines --pipeline-name-prefix ccfd --query 'PipelineSummaries[].PipelineName'
KEEP_GLOBS="ccfd-fraud-detector" \
  show "model package groups ccfd"             sm list-model-package-groups --name-contains ccfd --query 'ModelPackageGroupSummaryList[].ModelPackageGroupName'

mlops_step "Storage, images, roles and logs named ccfd"
KEEP_GLOBS="ccfd-serving ccfd-training" \
  show "ECR repositories ccfd"                 aws ecr describe-repositories "${R[@]}" --query 'repositories[?starts_with(repositoryName,`ccfd`)].repositoryName'
KEEP_GLOBS="ccfd-mlops-tfstate-$S ccfd-mlops-dvc-$S ccfd-mlops-jobs-$S ccfd-mlops-mlflow-$S" \
  show "S3 buckets ccfd"                       aws s3api list-buckets --prefix ccfd --query 'Buckets[].Name' --output text
KEEP_GLOBS="ccfd-sagemaker-execution-role ccfd-mlflow-app-role ccfd-github-actions ccfd-loop-events-role" \
  show "IAM roles ccfd"                        aws iam list-roles --query 'Roles[?starts_with(RoleName,`ccfd`)].RoleName' --output text
KEEP_GLOBS="/aws/sagemaker/Endpoints/ccfd-*" \
  show "log groups /aws/sagemaker (name has ccfd)" aws logs describe-log-groups "${R[@]}" --log-group-name-prefix /aws/sagemaker --query 'logGroups[?contains(logGroupName,`ccfd`)].logGroupName'
show "log groups /aws/lambda/ccfd"           aws logs describe-log-groups "${R[@]}" --log-group-name-prefix /aws/lambda/ccfd --query 'logGroups[].logGroupName'

mlops_step "Alarms, dashboards, events, topics and the (B) extras named ccfd"
KEEP_GLOBS="ccfd-maxpsi-high ccfd-netbenefit-drop" \
  show "CloudWatch alarms ccfd"                aws cloudwatch describe-alarms "${R[@]}" --query 'MetricAlarms[?contains(AlarmName,`ccfd`)].AlarmName'
KEEP_GLOBS="ccfd-mlops-dashboard" \
  show "CloudWatch dashboards ccfd"            aws cloudwatch list-dashboards "${R[@]}" --dashboard-name-prefix ccfd --query 'DashboardEntries[].DashboardName'
KEEP_GLOBS="ccfd-registry-state-change ccfd-netbenefit-drop-starts-retrain" \
  show "EventBridge rules ccfd"                aws events list-rules "${R[@]}" --name-prefix ccfd --query 'Rules[].Name'
KEEP_GLOBS="*:ccfd-mlops-alerts" \
  show "SNS topics ccfd"                       aws sns list-topics "${R[@]}" --query 'Topics[?contains(TopicArn,`ccfd`)].TopicArn'
show "Lambda functions ccfd"                 aws lambda list-functions "${R[@]}" --query 'Functions[?starts_with(FunctionName,`ccfd`)].FunctionName'
show "API Gateway HTTP APIs ccfd"            aws apigatewayv2 get-apis "${R[@]}" --query 'Items[?starts_with(Name,`ccfd`)].Name'
KEEP_GLOBS="ccfd_monitoring" \
  show "Glue databases ccfd"                   aws glue get-databases "${R[@]}" --query 'DatabaseList[?starts_with(Name,`ccfd`)].Name'

# --all only: every live resource with the course tag, whatever its name (the last proof of the full exit)
if [[ "$ALL" -eq 1 ]]; then
  mlops_step "Everything tagged course=$MLOPS_COURSE_TAG (live; free records left out)"
  # Free records, never counted: finished job records (they cannot be deleted) and SageMaker's lineage and
  # experiment records (contexts, actions, artifacts, experiments, trials, trial components)
  FREE='!contains(ResourceARN,`:training-job/`) && !contains(ResourceARN,`:processing-job/`)'
  FREE="$FREE"' && !contains(ResourceARN,`:transform-job/`) && !contains(ResourceARN,`:context/`)'
  FREE="$FREE"' && !contains(ResourceARN,`:action/`) && !contains(ResourceARN,`:artifact/`)'
  FREE="$FREE"' && !contains(ResourceARN,`:experiment`)'
  # A Deleted MLflow App stays in the tagging index for a while: its ARN is left out here only when
  # list-mlflow-apps says Deleted (the "MLflow Apps (Status not Deleted)" line above still counts a live one).
  DELETED_APPS="$(aws sagemaker list-mlflow-apps --region "$MLOPS_REGION" --query 'Summaries[?Status==`Deleted`].Arn' --output text 2>/dev/null | tr '\t' '\n')"
  # shellcheck disable=SC2329  # tagged_live is called through show
  tagged_live() {
    aws resourcegroupstaggingapi get-resources "${R[@]}" --tag-filters "Key=course,Values=$MLOPS_COURSE_TAG" \
        --query "ResourceTagMappingList[?$FREE].ResourceARN" \
      | tr '\t' '\n' | { if [[ -n "$DELETED_APPS" ]]; then grep -v -x -F "$DELETED_APPS" || true; else cat; fi; } \
      | ec2_still_there
  }
  # The tagging index lists a terminated instance and a deleted volume for a while (seen with your EC2 VM): such a
  # line is dropped only when EC2 itself says the instance is terminated or gone, or the volume does not exist. EC2
  # lists a terminated instance for about an hour, the tagging index much longer (a VM deleted the day before was
  # still in it): an instance EC2 no longer lists at all (an empty answer to the instance-id filter) is gone too.
  # An AWS error keeps the line (a line we could not check is never a pass).
  # shellcheck disable=SC2329  # called inside tagged_live, which show calls
  ec2_still_there() {
    local arn id state
    while IFS= read -r arn; do
      case "$arn" in
        *:instance/i-*)
          id="${arn##*/}"
          if state="$(aws ec2 describe-instances --region "$MLOPS_REGION" --filters "Name=instance-id,Values=$id" \
                        --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)"; then
            case "$state" in
              ""|None|terminated) ;;              # gone, or terminated: not a leftover
              *) printf '%s\n' "$arn" ;;          # pending, running, stopping, stopped, shutting-down: counted
            esac
          else
            printf '%s\n' "$arn"
          fi ;;
        *:volume/vol-*)
          id="${arn##*/}"
          [[ "$(aws ec2 describe-volumes --region "$MLOPS_REGION" --filters "Name=volume-id,Values=$id" --query 'length(Volumes)' --output text 2>/dev/null || echo 1)" == 0 ]] \
            || printf '%s\n' "$arn" ;;
        *) printf '%s\n' "$arn" ;;
      esac
    done
  }
  show "tagged course=$MLOPS_COURSE_TAG"          tagged_live
fi

echo
if [[ "$TOTAL" -eq 0 ]]; then
  echo "0 resources found"
  exit 0
fi
echo "$TOTAL resources found"
echo "Delete the course ones (bash scripts/destroy-env.sh$( [[ "$ALL" -eq 1 ]] && echo ' --all'), or by hand); anything that is not the course's: report it, never touch it."
exit 1
