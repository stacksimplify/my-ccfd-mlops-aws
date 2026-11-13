#!/usr/bin/env bash
# 04-enable-auto-stop.sh: stops your EC2 VM by itself when it has been idle (a safety net for the bill).
# Runs on your laptop. CHANGES AWS: one CloudWatch alarm (about 0.10 USD a month). Safe to run again: it updates the
# alarm. Options: --help
#
# The alarm mlops-vm-auto-stop watches the VM's CPU: average under 2% for VM_IDLE_MINUTES minutes in a row
# (120 = 2 hours by default, vm-settings.sh) -> the EC2 action "stop". A stopped VM bills only its disk; start it
# again with 06-stop-start-vm.sh start. An open SSH terminal with nothing running, or a long wait for AWS (an
# endpoint update, a pipeline run), uses almost no CPU and counts as idle: a VM you forgot stops after 2 hours.
# 09-destroy-vm.sh deletes the alarm with the VM.
# Exit codes: 0 OK, 1 Error or STOP.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=vm-settings.sh
. "$(cd "$(dirname "$0")" && pwd)/vm-settings.sh"
vm_masked "$0" "$@"

case "${1:-}" in
  --help|-h) vm_help "$0"; exit 0 ;;
  "") ;;
  *) vm_stop "unknown option \"$1\". Run: bash $0 --help" ;;
esac

echo "$(basename "$0")   region $VM_REGION   $(date -u +%F)"
vm_check_identity
ID="$(vm_instance_id)"
[ -n "$ID" ] || vm_stop "there is no $VM_NAME. Create it first: bash $(dirname "$0")/03-create-vm.sh"
aws cloudwatch put-metric-alarm --alarm-name "$VM_ALARM" \
  --alarm-description "MLOps course VM: stop after $VM_IDLE_MINUTES idle minutes (04-enable-auto-stop.sh)" \
  --namespace AWS/EC2 --metric-name CPUUtilization --dimensions "Name=InstanceId,Value=$ID" \
  --statistic Average --period 300 --evaluation-periods $(( VM_IDLE_MINUTES / 5 )) \
  --threshold 2 --comparison-operator LessThanThreshold --treat-missing-data notBreaching \
  --alarm-actions "arn:aws:automate:$VM_REGION:ec2:stop" \
  --tags "Key=course,Value=$COURSE_TAG" "Key=owner,Value=$TAG_OWNER" "Key=purpose,Value=course-vm"
# The alarm as AWS has it now: periods x 300 seconds = the idle minutes
# shellcheck disable=SC2046  # the 4 values are split into $1-$4 on purpose
set -- $(aws cloudwatch describe-alarms --alarm-names "$VM_ALARM" \
           --query 'MetricAlarms[0].[StateValue,EvaluationPeriods,Period,Threshold]' --output text)
vm_say "Auto-stop: ON  ($VM_ALARM: stop after $(( $2 * $3 / 60 )) min under $4% CPU; state now $1)"
echo
echo "OK: $VM_NAME ($ID) stops by itself after $VM_IDLE_MINUTES idle minutes."
echo "Next: bash $(dirname "$0")/05-connect-vm.sh --setup"
