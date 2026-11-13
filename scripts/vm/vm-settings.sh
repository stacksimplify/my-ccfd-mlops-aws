# shellcheck shell=bash
# shellcheck disable=SC2034  # the values are used by the scripts that source this file
# vm-settings.sh: the course VM's names, sizes and pinned tool versions, plus the helpers the VM scripts share.
# Sourced by the VM scripts 01-06 and 09, never run on its own. Optional: your own values in my-vm-settings.sh next
# to this file (git-ignored), for example:  TAG_OWNER="<your name>"   or   VM_IDLE_MINUTES=90
# Works with the bash 3.2 that macOS ships.

# The VM
VM_NAME="mlops-vm"                    # the Name tag and the SSH host key alias
VM_TYPE="c7i.xlarge"                     # 4 vCPU, 8 GiB, x86, full speed always (SageMaker images are x86: native builds)
VM_DISK_GB=40                            # gp3, encrypted, deleted with the VM
VM_REGION="us-east-1"
VM_USER="ec2-user"
VM_KEY_NAME="mlops-vm-key"            # the EC2 key pair and the 2 key files (02-create-keypair.sh)
VM_SG="mlops-vm-sg"                   # security group: SSH 22 from your IP only
VM_ROLE="mlops-vm-role"               # the VM's IAM role and instance profile (no access keys on the VM)
VM_ALARM="mlops-vm-auto-stop"         # 04-enable-auto-stop.sh
VM_IDLE_MINUTES=120                      # auto-stop after this many idle minutes (2 hours; a multiple of 5)
VM_PORTS="8001 8080 7080"                # 05-connect-vm.sh forwards these to your laptop (the base API, the serving API or container, MLflow)
VM_AMI_PARAM="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"   # the latest AL2023
TAG_OWNER="student"
COURSE_TAG="mlops-on-aws"

# The pinned tool versions live in vm-setup.sh (the VM's user-data).

# Prices for the cost line (us-east-1, on-demand, USD)
PRICE_HOUR="0.1785"                      # c7i.xlarge
PRICE_GB_MONTH="0.08"                    # gp3
PRICE_IPV4_HOUR="0.005"                  # public IPv4 while running

VM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The key files live in the course clone, in keypair-local/ at its top (git-ignored: never committed), so every
# section folder's copy of these scripts finds the same key. Outside a git clone: next to the scripts.
REPO_ROOT="$(git -C "$VM_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$VM_DIR" && pwd))"
VM_KEY_DIR="$REPO_ROOT/keypair-local"
VM_KEY="$VM_KEY_DIR/$VM_KEY_NAME"        # the private key (ed25519); the public key is the same path plus .pub
if [ -f "$VM_DIR/my-vm-settings.sh" ]; then
  # shellcheck source=/dev/null  # your own file, optional
  . "$VM_DIR/my-vm-settings.sh"
  VM_KEY="$VM_KEY_DIR/$VM_KEY_NAME"
fi

export AWS_PAGER=""
export AWS_REGION="$VM_REGION"
MASK_SED='s/[0-9]{12}/<ACCOUNT_ID>/g'

# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------
vm_say()  { printf '%s\n' "$*"; }
vm_step() { printf '\n== %s\n' "$*"; }
vm_stop() { printf 'STOP: %s\n' "$*"; exit 1; }

# Call first:  vm_masked "$0" "$@"   -> the script runs again with stdout and stderr through the account ID mask
vm_masked() {
  if [ -z "${_VM_MASKED:-}" ]; then
    local script="$1" unbuffered=""
    shift
    if sed -u -n '' < /dev/null > /dev/null 2>&1; then unbuffered="-u"; fi
    _VM_MASKED=1 bash "$script" "$@" 2>&1 | (trap '' INT; exec sed $unbuffered -E "$MASK_SED")
    exit "${PIPESTATUS[0]}"
  fi
}

# vm_ask "<question>": asks on the terminal itself (/dev/tty), outside the account ID mask, so the question shows at
# once and the answer is read from the keyboard; prints the answer. Returns 1 when there is no terminal.
vm_ask() {
  local answer
  { : < /dev/tty; } 2> /dev/null || return 1
  printf '%s ' "$1" > /dev/tty
  IFS= read -r answer < /dev/tty || answer=""
  printf '%s' "$answer"
}

# True when this script runs on your EC2 VM: vm-setup.sh writes /etc/mlops-course-vm there.
on_course_vm() { [ -f /etc/mlops-course-vm ]; }

# --help: prints the header comment of the calling script
vm_help() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$1"; }

# The AWS CLI is there, the region is us-east-1, and you are not the root user
vm_check_identity() {
  local arn
  command -v aws > /dev/null 2>&1 || vm_stop "the AWS CLI is not installed on this laptop (Section 00-B Step-01-03 installs it)."
  arn="$(aws sts get-caller-identity --query Arn --output text 2>&1)" ||
    vm_stop "no AWS identity: $(printf '%s' "$arn" | grep -v '^[[:space:]]*$' | head -1)  (run: aws configure, Section 00-B Step-01-03)"
  vm_say "Identity: $arn   region $VM_REGION"
  case "$arn" in
    *:root) vm_stop "you are signed in as the root user. Use an IAM user's access key (Section 00-B Step-01-02)." ;;
  esac
}

# The VM's instance ID (any state except terminated), or empty
vm_instance_id() {
  aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$VM_NAME" "Name=tag:course,Values=$COURSE_TAG" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | awk '{print $1}'
}

vm_state() { aws ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].State.Name' --output text; }
vm_ip()    { aws ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }

# Your public IPv4 (AWS's checkip page answers with it). Called as MYIP="$(vm_my_ip)": a STOP goes to stderr, so it shows.
vm_my_ip() {
  local ip
  ip="$(curl -fsS --max-time 10 https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || true)"
  case "$ip" in
    *.*.*.*) printf '%s' "$ip" ;;
    *) vm_stop "could not find your public IP (https://checkip.amazonaws.com did not answer). Check the internet connection, then run this again." >&2 ;;
  esac
}

vm_sg_id() {
  aws ec2 describe-security-groups --filters "Name=group-name,Values=$VM_SG" "Name=tag:course,Values=$COURSE_TAG" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null | grep -v '^None$' || true
}

# The SSH options every script uses. HostKeyAlias: the host key is stored under the name mlops-vm, so a new
# public IP after a stop and start never brings a host key warning.
vm_ssh_opts() {
  printf '%s\n' -i "$VM_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    -o "HostKeyAlias=$VM_NAME" -o ConnectTimeout=10 -o ServerAliveInterval=30
}

# Wait until SSH answers (18 tries: at most about 4 minutes)
vm_wait_ssh() {
  local ip="$1" i opts
  opts="$(vm_ssh_opts)"
  for i in $(seq 1 18); do
    # shellcheck disable=SC2086  # the options are one per line, split on purpose
    if ssh $opts -o BatchMode=yes "$VM_USER@$ip" true > /dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}
