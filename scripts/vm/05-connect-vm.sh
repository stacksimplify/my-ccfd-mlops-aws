#!/usr/bin/env bash
# shellcheck disable=SC2016  # the --query strings are JMESPath: their `...` literals stay in single quotes
# 05-connect-vm.sh: opens a terminal on your EC2 VM. It finds the VM's IP itself, so you never paste it.
# Runs on your laptop. Read only in AWS. Options: --setup, --help
#
#   (no option)    an SSH session on the VM (type exit to come back). Ports 8001, 8080 and 7080 of the VM
#                  are forwarded to your laptop (VM_PORTS), so the API pages and the MLflow UI open in your
#                  browser while the session is open. Ports already busy on your laptop (for example in a second
#                  VM terminal: the first one forwards them) are not forwarded; one line names them.
#   --setup        after 03-create-vm.sh: checks that SSH is open to your current IP, waits for the first-boot
#                  install (up to 20 minutes) and prints the tool table (mlops-vm-check)
# Settings: vm-settings.sh (VM_PORTS: the forwarded ports).
# Exit codes: 0 OK, 1 Error or STOP; with a session, the exit code of ssh.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=vm-settings.sh
. "$(cd "$(dirname "$0")" && pwd)/vm-settings.sh"

MODE="ssh"
case "${1:-}" in
  --help|-h) vm_help "$0"; exit 0 ;;
  --setup) MODE="setup" ;;
  "") ;;
  *) vm_stop "unknown option \"$1\". Run: bash $0 --help" ;;
esac

if on_course_vm; then vm_stop "you are on your EC2 VM already. Run this on your laptop (type exit to go back to it)."; fi
[ -f "$VM_KEY" ] || vm_stop "no key file keypair-local/$VM_KEY_NAME: run 02-create-keypair.sh first"
ID="$(vm_instance_id)"
[ -n "$ID" ] || vm_stop "there is no $VM_NAME. Create it: bash $(dirname "$0")/03-create-vm.sh"
STATE="$(vm_state "$ID")"
[ "$STATE" = running ] || vm_stop "$VM_NAME is $STATE. Start it: bash $(dirname "$0")/06-stop-start-vm.sh start"
IP="$(vm_ip "$ID")"
OPTS="$(vm_ssh_opts)"

if [ "$MODE" = ssh ]; then
  FWD=""
  BUSY=""
  for port in $VM_PORTS; do
    if command -v lsof > /dev/null 2>&1 && lsof -nP -iTCP:"$port" -sTCP:LISTEN > /dev/null 2>&1; then
      BUSY="$BUSY $port"
    else
      FWD="$FWD -L $port:localhost:$port"
    fi
  done
  [ -z "$BUSY" ] || vm_say "Not forwarded, busy on your laptop (for example in your first VM terminal):$BUSY"
  echo "Connecting to $VM_NAME ($IP) as $VM_USER. Forwarded to your laptop:${FWD:- none}. Type exit to come back."
  # shellcheck disable=SC2086  # options and forwards are split on purpose
  exec ssh $OPTS -o ExitOnForwardFailure=no $FWD "$VM_USER@$IP"
fi

# --setup
echo "$(basename "$0") --setup   $VM_NAME $ID ($IP)   $(date -u +%F)"
# Is your current IP the one the security group allows? (a warning only)
ME="$(curl -fsS --max-time 10 https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || true)"
if [ -n "$ME" ]; then
  ALLOWED="$(aws ec2 describe-security-groups --group-ids "$(vm_sg_id)" \
              --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text 2>/dev/null || true)"
  case " $ALLOWED " in
    *" $ME/32 "*) ;;
    *) vm_say "WARN: SSH is open to $ALLOWED, but your IP is now $ME. Run: bash $(dirname "$0")/03-create-vm.sh" ;;
  esac
fi

vm_step "1. Waiting for the first-boot install (the VM's /var/log/mlops-vm-setup.log)"
vm_wait_ssh "$IP" || vm_stop "SSH does not answer from $IP. Your IP changed? Run 03-create-vm.sh again."
start=$(date +%s)
last=""
while :; do
  # shellcheck disable=SC2086
  line="$(ssh $OPTS -o BatchMode=yes "$VM_USER@$IP" 'sudo tail -n 1 /var/log/mlops-vm-setup.log 2>/dev/null; sudo grep "^== " /var/log/mlops-vm-setup.log 2>/dev/null | tail -n 1' 2>/dev/null || true)"
  case "$line" in
    *"MLOPS VM SETUP DONE"*) vm_say "  done after $(( ($(date +%s) - start) / 60 )) min of waiting"; break ;;
    *"MLOPS VM SETUP FAILED"*) vm_stop "the install failed: $(printf '%s' "$line" | head -1). On the VM: sudo tail -50 /var/log/mlops-vm-setup.log" ;;
  esac
  now="$(printf '%s' "$line" | grep '^== ' | tail -1 || true)"
  if [ -n "$now" ] && [ "$now" != "$last" ]; then vm_say "  $now"; last="$now"; fi
  [ $(( $(date +%s) - start )) -lt 1200 ] || vm_stop "still installing after 20 minutes. On the VM: sudo tail -50 /var/log/mlops-vm-setup.log"
  sleep 15
done

vm_step "2. The tools on the VM"
# shellcheck disable=SC2086
ssh $OPTS "$VM_USER@$IP" mlops-vm-check || vm_stop "a tool is missing or has the wrong version (the table above)"

echo
echo "OK: your EC2 VM is ready."
echo "Next: bash $0      (a terminal on the VM, ports $VM_PORTS forwarded)"
