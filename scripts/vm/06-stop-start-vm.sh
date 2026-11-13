#!/usr/bin/env bash
# 06-stop-start-vm.sh: stops your EC2 VM or starts it.
# Runs on your laptop. CHANGES AWS (the hourly bill stops or starts). Options: stop|start, --help
#
# With no word (or a typo) it changes nothing: it says to use stop or start.
# A stopped VM keeps its disk (your files, the conda env, Docker images) and bills only that disk (40 GB gp3: about
# 0.11 USD a day). At every start AWS gives it a NEW public IP; the VM scripts find it themselves, so nothing to
# change on your side. Containers do not start again by themselves.
# Exit codes: 0 OK, 1 Error or STOP.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=vm-settings.sh
. "$(cd "$(dirname "$0")" && pwd)/vm-settings.sh"
vm_masked "$0" "$@"
if on_course_vm; then vm_stop "you are on your EC2 VM. Run this on your laptop (type exit to go back to it)."; fi

case "${1:-}" in
  --help|-h) vm_help "$0"; exit 0 ;;
  stop|start) VERB="$1" ;;
  *) vm_stop "use stop or start: bash $0 stop   (or: bash $0 start)" ;;
esac

echo "$(basename "$0") $VERB   region $VM_REGION   $(date -u +%F)"
ID="$(vm_instance_id)"
[ -n "$ID" ] || vm_stop "there is no $VM_NAME. Create it: bash $(dirname "$0")/03-create-vm.sh"
STATE="$(vm_state "$ID")"

case "$VERB" in
  stop)
    if [ "$STATE" = stopped ]; then vm_say "$VM_NAME is already stopped."; exit 0; fi
    aws ec2 stop-instances --instance-ids "$ID" > /dev/null
    vm_say "$VM_NAME stopping ..."
    aws ec2 wait instance-stopped --instance-ids "$ID"
    vm_say "$VM_NAME stopped: only its disk bills now."
    echo "Next (when you come back): bash $0 start" ;;
  start)
    if [ "$STATE" = stopping ]; then vm_say "waiting until it has stopped ..."; aws ec2 wait instance-stopped --instance-ids "$ID"; STATE=stopped; fi
    if [ "$STATE" != running ]; then
      aws ec2 start-instances --instance-ids "$ID" > /dev/null
      vm_say "$VM_NAME starting ..."
      aws ec2 wait instance-running --instance-ids "$ID"
    fi
    IP="$(vm_ip "$ID")"
    vm_say "$VM_NAME running, public IP $IP"
    if [ ! -f "$VM_KEY" ]; then
      vm_say "WARN: no key file $VM_KEY, so SSH cannot log in from this computer."
      vm_say "  On the computer that made the key pair (Section 00-B Step-01-06), copy its keypair-local/ folder to the top of this clone."
    else
      vm_say "waiting until SSH answers (usually under 1 minute, at most about 4) ..."
      if vm_wait_ssh "$IP"; then vm_say "SSH answers"; else vm_say "WARN: SSH does not answer: your IP changed? Run 03-create-vm.sh again."; fi
    fi
    echo "Next: bash $(dirname "$0")/05-connect-vm.sh" ;;
esac
