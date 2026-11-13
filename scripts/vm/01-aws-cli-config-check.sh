#!/usr/bin/env bash
# 01-aws-cli-config-check.sh: checks your laptop is ready for your EC2 VM. READ ONLY, free.
# Runs on your laptop. It uses your current AWS CLI identity (the default profile, or AWS_PROFILE if you set one).
# Options: --help
#
# Checks: the AWS CLI (version 2), the profile in use, the region us-east-1, and who you are (an IAM user, never the
# root user; the account shows only its last 4 digits).
# Exit codes: 0 OK, 1 FAIL (fix the FAIL lines, then run it again).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=vm-settings.sh
. "$(cd "$(dirname "$0")" && pwd)/vm-settings.sh"

case "${1:-}" in
  --help|-h) vm_help "$0"; exit 0 ;;
  "") ;;
  *) vm_stop "unknown option \"$1\". Run: bash $0 --help" ;;
esac

echo "$(basename "$0")   $(date -u +%F)"
STATUS=OK
fail() { echo "  FAIL: $*"; STATUS=FAIL; }

vm_step "1. The AWS CLI"
CLI="not installed"
if command -v aws > /dev/null 2>&1; then CLI="$(aws --version 2>&1 | awk '{print $1}')"; fi
echo "  AWS CLI:   ${CLI#aws-cli/}"
case "$CLI" in aws-cli/2.*) ;; *) fail "the course needs AWS CLI version 2 (Section 00-B Step-01-03)" ;; esac
[ "$STATUS" = OK ] || { echo; echo "Overall:   FAIL"; exit 1; }

vm_step "2. The profile and the region"
echo "  Profile:   ${AWS_PROFILE:-default}"
REGION="$(aws configure get region 2>/dev/null || true)"
echo "  Region:    ${REGION:-<not set>} (the VM scripts always use $VM_REGION)"
[ "$REGION" = "$VM_REGION" ] || fail "set the profile's region: aws configure set region $VM_REGION"

vm_step "3. Who you are"
if ! ARN="$(aws sts get-caller-identity --query Arn --output text 2>&1)"; then
  fail "no AWS identity: $(printf '%s' "$ARN" | sed -E "$MASK_SED" | grep -v '^[[:space:]]*$' | head -1)"
  echo "        Fix: run  aws configure  with a current access key of your IAM user (Section 00-B Step-01-03);"
  echo "        InvalidAccessKeyId means the CLI holds an old, deleted key"
  echo; echo "Overall:   FAIL"; exit 1
fi
ACCOUNT="$(printf '%s' "$ARN" | cut -d: -f5)"
echo "  Account:   ........${ACCOUNT: -4}"
echo "  Identity:  ${ARN##*:}"
case "$ARN" in
  *:root) fail "you are the root user: use an IAM user's access key, never the root user's (Section 00-B Step-01-02)" ;;
esac

echo
echo "Overall:   $STATUS"
[ "$STATUS" = OK ] || exit 1
echo "Next: bash $(dirname "$0")/02-create-keypair.sh"
