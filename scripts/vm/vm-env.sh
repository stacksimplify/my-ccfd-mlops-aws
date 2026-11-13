# shellcheck shell=bash disable=SC2155
# vm-env.sh: sets the 3 VM variables in THIS terminal. Load it (do not run it), from anywhere in your course clone:
#   source scripts/vm/vm-env.sh
# It looks every value up fresh (never stored: the IP changes at every start, the ID when the VM is made again):
#   MLOPS_KEY     the private key file, keypair-local/mlops-vm-key at the top of your course clone
#   MLOPS_VM_ID   the VM's instance ID (running or stopped)
#   MLOPS_VM_IP   the VM's public IP (only while it runs)
# A missing value prints a WARN line with the step that fixes it. Works in bash and zsh.

export MLOPS_KEY="$(git rev-parse --show-toplevel 2>/dev/null)/keypair-local/mlops-vm-key"
export MLOPS_VM_ID="$(aws ec2 describe-instances --filters Name=tag:Name,Values=mlops-vm Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)"
export MLOPS_VM_IP="$(aws ec2 describe-instances --filters Name=tag:Name,Values=mlops-vm Name=instance-state-name,Values=running --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)"

echo "MLOPS_KEY=$MLOPS_KEY"
echo "MLOPS_VM_ID=$MLOPS_VM_ID"
echo "MLOPS_VM_IP=$MLOPS_VM_IP"
[ -f "$MLOPS_KEY" ] || echo "WARN: no key file $MLOPS_KEY: source this inside your course clone; still missing? Section 00-B Step-01-06"
case "$MLOPS_VM_ID" in
  i-*)
    case "$MLOPS_VM_IP" in
      ""|None) echo "WARN: no public IP: the VM is stopped. Start it (Section 00-B Step-03), then source this file again" ;;
    esac ;;
  *) echo "WARN: your EC2 VM was not found: create it (Section 00-B Step-01-07)" ;;
esac
