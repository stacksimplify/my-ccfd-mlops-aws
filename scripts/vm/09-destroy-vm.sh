#!/usr/bin/env bash
# 09-destroy-vm.sh: deletes your EC2 VM and everything 03-create-vm.sh and 04-enable-auto-stop.sh made.
# Runs on your laptop. CHANGES AWS: it lists what goes, you type mlops-vm, then it deletes and waits.
# Used at the end of MLOps 16 (or when you leave the course earlier). Options: --help
#
# It deletes: the auto-stop alarm, the instance and its disk (everything on the VM is gone: push your work first),
# the security group, the key pair mlops-vm-key in EC2 AND its 2 files in keypair-local/, the role and instance
# profile mlops-vm-role; it removes the VM's host key from ~/.ssh/known_hosts. (Stopping the VM keeps the key.)
# Exit codes: 0 OK (deleted, or nothing to delete), 1 Error or STOP (nothing was deleted).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=vm-settings.sh
. "$(cd "$(dirname "$0")" && pwd)/vm-settings.sh"
vm_masked "$0" "$@"
if on_course_vm; then vm_stop "you are on your EC2 VM. Run this on your laptop (type exit to go back to it)."; fi

YES=0
for arg in "$@"; do
  case "$arg" in
    --yes) YES=1 ;;
    --help|-h) vm_help "$0"; exit 0 ;;
    *) vm_stop "unknown option \"$arg\". Run: bash $0 --help" ;;
  esac
done

echo "$(basename "$0")   region $VM_REGION   $(date -u +%F)"
vm_check_identity
ID="$(vm_instance_id)"
SG="$(vm_sg_id)"
ALARM="$(aws cloudwatch describe-alarms --alarm-names "$VM_ALARM" --query 'MetricAlarms[0].AlarmName' --output text 2>/dev/null | grep -v '^None$' || true)"
KEY="$(aws ec2 describe-key-pairs --key-names "$VM_KEY_NAME" --query 'KeyPairs[0].KeyName' --output text 2>/dev/null || true)"
KEYFILES=""
if [ -f "$VM_KEY" ] || [ -f "$VM_KEY.pub" ]; then KEYFILES="keypair-local/$VM_KEY_NAME (+ .pub)"; fi
ROLE="$(aws iam get-role --role-name "$VM_ROLE" --query Role.RoleName --output text 2>/dev/null || true)"
PROFILE="$(aws iam get-instance-profile --instance-profile-name "$VM_ROLE" --query InstanceProfile.InstanceProfileName --output text 2>/dev/null || true)"

vm_step "What goes"
n=0
show() { if [ -n "$2" ]; then vm_say "  $1: $2"; n=$((n + 1)); fi; }
show "instance (and its disk)" "${ID:+$ID ($(vm_state "$ID"))}"
show "auto-stop alarm" "$ALARM"
show "security group" "$SG"
show "key pair (in EC2)" "$KEY"
show "key files (your laptop)" "$KEYFILES"
show "instance profile" "$PROFILE"
show "IAM role" "$ROLE"
if [ "$n" = 0 ]; then
  ssh-keygen -R "$VM_NAME" > /dev/null 2>&1 || true
  echo "OK: nothing to delete."
  exit 0
fi

if [ "$YES" != 1 ]; then
  answer="$(vm_ask "Everything on the VM is lost. Type $VM_NAME to delete:")" || vm_stop "not a terminal: add --yes to delete (the list is above)."
  echo "  your answer: ${answer:-<empty>}"
  [ "$answer" = "$VM_NAME" ] || { echo "Nothing was deleted."; exit 1; }
fi

vm_step "Deleting"
if [ -n "$ALARM" ]; then aws cloudwatch delete-alarms --alarm-names "$VM_ALARM"; vm_say "  alarm deleted"; fi
if [ -n "$ID" ]; then
  aws ec2 terminate-instances --instance-ids "$ID" > /dev/null
  vm_say "  $ID terminating ..."
  aws ec2 wait instance-terminated --instance-ids "$ID"
  vm_say "  $ID terminated (its disk too)"
fi
if [ -n "$SG" ]; then
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do   # the network card of a just-terminated VM lets go after a moment
    if aws ec2 delete-security-group --group-id "$SG" > /dev/null 2>&1; then vm_say "  security group deleted"; break; fi
    [ "$i" -lt 12 ] || vm_stop "the security group $SG is still in use; run this script again in a minute"
    sleep 10
  done
fi
if [ -n "$KEY" ]; then aws ec2 delete-key-pair --key-name "$VM_KEY_NAME" > /dev/null; vm_say "  key pair $VM_KEY_NAME deleted in EC2"; fi
if [ -n "$KEYFILES" ]; then rm -f "$VM_KEY" "$VM_KEY.pub"; rmdir "$VM_KEY_DIR" 2>/dev/null || true; vm_say "  key files deleted (keypair-local/)"; fi
if [ -n "$PROFILE" ]; then
  if [ -n "$ROLE" ]; then aws iam remove-role-from-instance-profile --instance-profile-name "$VM_ROLE" --role-name "$VM_ROLE" 2>/dev/null || true; fi
  aws iam delete-instance-profile --instance-profile-name "$VM_ROLE"
  vm_say "  instance profile deleted"
fi
if [ -n "$ROLE" ]; then
  for p in $(aws iam list-attached-role-policies --role-name "$VM_ROLE" --query 'AttachedPolicies[].PolicyArn' --output text); do
    aws iam detach-role-policy --role-name "$VM_ROLE" --policy-arn "$p"
  done
  aws iam delete-role --role-name "$VM_ROLE"
  vm_say "  role deleted"
fi
ssh-keygen -R "$VM_NAME" > /dev/null 2>&1 || true
vm_say "  host key removed from ~/.ssh/known_hosts"

echo
echo "OK: $VM_NAME and its resources are deleted."
# The account check: orphan-sweep.sh where this section has it (MLOps 06 on), else one AWS CLI check
SWEEP="$(dirname "$(dirname "$0")")/orphan-sweep.sh"
if [ -f "$SWEEP" ]; then
  echo "Next: bash $SWEEP   (it must end with 0 resources found)"
else
  echo "Check: aws ec2 describe-instances --filters Name=tag:Name,Values=$VM_NAME Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text   (prints nothing)"
fi
