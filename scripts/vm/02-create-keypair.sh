#!/usr/bin/env bash
# 02-create-keypair.sh: creates the SSH key pair you use to log in to your EC2 VM.
# Runs on your laptop. CHANGES AWS (free): it imports the PUBLIC key into EC2 as the key pair mlops-vm-key.
# Options: --help
#
# The 2 key files go to keypair-local/ at the top of your course clone:
#   keypair-local/mlops-vm-key       the private key (ed25519, no passphrase, mode 600): never leaves your laptop
#   keypair-local/mlops-vm-key.pub   the public key (the only part AWS gets)
# keypair-local/ is in .gitignore: it is never committed (it stops when git does not ignore it, or when the private
# key is not mode 600). Safe to run again: it creates only what is missing.
# Stopping the VM keeps the key; 09-destroy-vm.sh deletes the EC2 key pair and the 2 files.
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

if git -C "$REPO_ROOT" rev-parse --git-dir > /dev/null 2>&1 && ! git -C "$REPO_ROOT" check-ignore -q "keypair-local/$VM_KEY_NAME"; then
  vm_stop "keypair-local/ is NOT ignored by git in $REPO_ROOT. Add the line keypair-local/ to .gitignore first."
fi

vm_step "1. The key files"
NEW=0   # 1 when this run created a file or imported the key
mkdir -p "$VM_KEY_DIR"
chmod 700 "$VM_KEY_DIR"
if [ -f "$VM_KEY" ]; then
  vm_say "  keypair-local/$VM_KEY_NAME is there (kept)"
else
  NEW=1
  ssh-keygen -t ed25519 -N "" -C "$VM_KEY_NAME" -f "$VM_KEY" -q
  vm_say "  created keypair-local/$VM_KEY_NAME and keypair-local/$VM_KEY_NAME.pub"
fi
chmod 600 "$VM_KEY"
# ssh refuses a private key others can read. ls -l is the simple way to read the mode on macOS and Linux alike.
# shellcheck disable=SC2012
MODE="$(ls -l "$VM_KEY" | cut -c1-10)"
if [ "$MODE" != "-rw-------" ]; then
  vm_stop "the private key is not mode 600 ($MODE). On Windows keep the course clone inside WSL (~), not under /mnt/c."
fi
[ -f "$VM_KEY.pub" ] || ssh-keygen -y -f "$VM_KEY" > "$VM_KEY.pub"

vm_step "2. The key pair in EC2"
if aws ec2 describe-key-pairs --key-names "$VM_KEY_NAME" > /dev/null 2>&1; then
  vm_say "  $VM_KEY_NAME is in EC2 already (kept)"
else
  NEW=1
  aws ec2 import-key-pair --key-name "$VM_KEY_NAME" --public-key-material "fileb://$VM_KEY.pub" \
    --tag-specifications "ResourceType=key-pair,Tags=[{Key=Name,Value=$VM_KEY_NAME},{Key=course,Value=$COURSE_TAG},{Key=owner,Value=$TAG_OWNER},{Key=purpose,Value=course-vm},{Key=created,Value=$(date +%F)}]" > /dev/null
  vm_say "  imported $VM_KEY_NAME (the public key only)"
fi

echo
if [ "$NEW" = 0 ]; then echo "OK: nothing new was created: the key pair $VM_KEY_NAME is ready."; else echo "OK: the key pair $VM_KEY_NAME is ready."; fi
echo "Next: bash $(dirname "$0")/03-create-vm.sh"
