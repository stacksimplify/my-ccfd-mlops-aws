#!/usr/bin/env bash
# shellcheck disable=SC2016  # the --query strings are JMESPath: their `...` literals stay in single quotes
# 03-create-vm.sh: creates your EC2 VM (Amazon Linux 2023, x86) with SSH open to your IP only.
# Runs on your laptop, with the AWS CLI profile of Section 00-B Step-01-03. CHANGES AWS (billable): it shows the cost and
# asks first. Run again: when the VM exists it only updates the SSH rule to your current IP. Options: --help
#
# It creates (each tagged Name, course=mlops-on-aws, owner, purpose=course-vm, created):
#   (it uses the key pair mlops-vm-key of 02-create-keypair.sh) the security group mlops-vm-sg (SSH 22 from your IP /32, nothing else), the IAM role
#   and instance profile mlops-vm-role (AdministratorAccess: the AWS CLI on the VM needs no access key), and
#   the instance (c7i.xlarge, 40 GB gp3 encrypted, IMDSv2 only) whose first boot runs vm-setup.sh (about 2 minutes).
# Settings: vm-settings.sh.
# Exit codes: 0 OK, 1 Error or STOP (or you answered no).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=vm-settings.sh
. "$(cd "$(dirname "$0")" && pwd)/vm-settings.sh"
vm_masked "$0" "$@"

YES=0
for arg in "$@"; do
  case "$arg" in
    --yes) YES=1 ;;
    --help|-h) vm_help "$0"; exit 0 ;;
    *) vm_stop "unknown option \"$arg\". Run: bash $0 --help" ;;
  esac
done
# On the VM, "your IP" would be the VM's own IP, and the SSH rule would shut your laptop out
if on_course_vm; then vm_stop "you are on your EC2 VM. Run this on your laptop (type exit to go back to it)."; fi

echo "$(basename "$0")   region $VM_REGION   $(date -u +%F)"
vm_check_identity
MYIP="$(vm_my_ip)"
CREATED="$(date +%F)"
tags() {   # tags <resource type> -> a --tag-specifications value
  printf 'ResourceType=%s,Tags=[{Key=Name,Value=%s},{Key=course,Value=%s},{Key=owner,Value=%s},{Key=purpose,Value=course-vm},{Key=created,Value=%s}]' \
    "$1" "$VM_NAME" "$COURSE_TAG" "$TAG_OWNER" "$CREATED"
}

# Allow SSH from this IP only (removes any other SSH source)
open_ssh_to_me() {
  local sg="$1" old
  for old in $(aws ec2 describe-security-groups --group-ids "$sg" \
                 --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text); do
    if [ "$old" != "$MYIP/32" ]; then
      aws ec2 revoke-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$old" > /dev/null
      vm_say "  SSH from $old: removed"
    fi
  done
  if ! aws ec2 describe-security-groups --group-ids "$sg" \
         --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text | grep -qw "$MYIP/32"; then
    aws ec2 authorize-security-group-ingress --group-id "$sg" \
      --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MYIP/32,Description=mlops-ssh-from-my-ip}]" > /dev/null
  fi
  vm_say "  SSH (22) open to $MYIP/32 only"
}

# 1. The VM exists already? Then only the SSH rule changes.
ID="$(vm_instance_id)"
if [ -n "$ID" ]; then
  vm_step "$VM_NAME exists ($ID, $(vm_state "$ID")): only the SSH rule is updated to your IP"
  open_ssh_to_me "$(vm_sg_id)"
  echo
  echo "OK: nothing new was created."
  echo "Next: bash $(dirname "$0")/05-connect-vm.sh --setup     (stopped? bash $(dirname "$0")/06-stop-start-vm.sh start)"
  exit 0
fi

# 2. What it needs: the default VPC, a subnet in a zone that has c7i.xlarge, the vCPU quota, the AMI
vm_step "1. Checks before anything is created"
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
[ "$VPC" != "None" ] || vm_stop "this region has no default VPC. Create one: aws ec2 create-default-vpc"
ZONES="$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
          --filters "Name=instance-type,Values=$VM_TYPE" --query 'InstanceTypeOfferings[].Location' --output text)"
SUBNET=""
for z in $ZONES; do
  SUBNET="$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" "Name=availability-zone,Values=$z" \
             "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)"
  if [ "$SUBNET" != "None" ] && [ -n "$SUBNET" ]; then ZONE="$z"; break; fi
done
[ -n "$SUBNET" ] && [ "$SUBNET" != "None" ] || vm_stop "no default subnet in a zone that offers $VM_TYPE"
vm_say "  default VPC $VPC, subnet $SUBNET ($ZONE)"
QUOTA="$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A \
          --query 'Quota.Value' --output text 2>/dev/null || echo unknown)"
vm_say "  vCPU quota (Running On-Demand Standard instances): $QUOTA; this VM needs 4 vCPU"
case "$QUOTA" in
  unknown) vm_say "  WARN: could not read the quota; continuing" ;;
  *) if [ "${QUOTA%.*}" -lt 4 ]; then vm_stop "your vCPU quota is $QUOTA. Ask for 8 in Service Quotas (EC2, L-1216C47A)."; fi ;;
esac
AMI="$(aws ssm get-parameter --name "$VM_AMI_PARAM" --query Parameter.Value --output text)"
AMI_NAME="$(aws ec2 describe-images --image-ids "$AMI" --query 'Images[0].Name' --output text)"
vm_say "  AMI $AMI ($AMI_NAME, the latest Amazon Linux 2023 for x86_64)"

# 3. The cost, then the question
vm_step "2. The cost (us-east-1, on-demand)"
vm_say "  $VM_TYPE running:  $PRICE_HOUR USD/hour + public IPv4 $PRICE_IPV4_HOUR USD/hour"
vm_say "  $VM_DISK_GB GB gp3 disk:   $(awk -v g="$VM_DISK_GB" -v p="$PRICE_GB_MONTH" 'BEGIN { printf "%.2f", g * p }') USD/month, also while the VM is stopped"
vm_say "  Stop it when you stop (06-stop-start-vm.sh stop); 04-enable-auto-stop.sh stops it after $VM_IDLE_MINUTES idle minutes."
if [ "$YES" != 1 ]; then
  answer="$(vm_ask "Create $VM_NAME now? [y/N]")" || vm_stop "not a terminal: add --yes to create (the cost is above)."
  echo "  your answer: ${answer:-<empty>}"
  case "$answer" in y|Y|yes|YES) ;; *) echo "Nothing was created."; exit 1 ;; esac
fi

# 4. The key pair (02-create-keypair.sh made it)
vm_step "3. The SSH key pair $VM_KEY_NAME"
[ -f "$VM_KEY" ] || vm_stop "no key file keypair-local/$VM_KEY_NAME. Run: bash $(dirname "$0")/02-create-keypair.sh"
aws ec2 describe-key-pairs --key-names "$VM_KEY_NAME" > /dev/null 2>&1 \
  || vm_stop "the key pair $VM_KEY_NAME is not in EC2. Run: bash $(dirname "$0")/02-create-keypair.sh"
vm_say "  keypair-local/$VM_KEY_NAME and the EC2 key pair $VM_KEY_NAME: OK"

# 5. The security group
vm_step "4. The security group $VM_SG"
SG="$(vm_sg_id)"
if [ -z "$SG" ]; then
  SG="$(aws ec2 create-security-group --group-name "$VM_SG" --description "MLOps course VM: SSH from the student IP only" \
         --vpc-id "$VPC" --tag-specifications "$(tags security-group)" --query GroupId --output text)"
  vm_say "  created $SG"
fi
open_ssh_to_me "$SG"

# 6. The IAM role and instance profile
vm_step "5. The VM role $VM_ROLE (the AWS CLI on the VM uses it: no access key on the VM)"
if ! aws iam get-role --role-name "$VM_ROLE" > /dev/null 2>&1; then
  aws iam create-role --role-name "$VM_ROLE" --description "MLOps course VM" \
    --assume-role-policy-document "file://$VM_DIR/ec2-trust-policy.json" \
    --tags "Key=course,Value=$COURSE_TAG" "Key=owner,Value=$TAG_OWNER" "Key=purpose,Value=course-vm" "Key=created,Value=$CREATED" > /dev/null
  aws iam attach-role-policy --role-name "$VM_ROLE" --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
  vm_say "  created the role (AdministratorAccess)"
fi
if ! aws iam get-instance-profile --instance-profile-name "$VM_ROLE" > /dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$VM_ROLE" \
    --tags "Key=course,Value=$COURSE_TAG" "Key=owner,Value=$TAG_OWNER" > /dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$VM_ROLE" --role-name "$VM_ROLE"
  vm_say "  created the instance profile"
fi

# 7. The user-data: the requirements, the pinned versions and vm-setup.sh, written to /opt/mlops at the first boot
vm_step "6. The instance"
ID=""
for attempt in 1 2 3 4 5 6 7 8; do   # a new instance profile takes a few seconds to be usable
  if ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$VM_TYPE" --key-name "$VM_KEY_NAME" \
            --security-group-ids "$SG" --subnet-id "$SUBNET" --associate-public-ip-address \
            --iam-instance-profile "Name=$VM_ROLE" \
            --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$VM_DISK_GB,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}" \
            --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled" \
            --tag-specifications "$(tags instance)" "$(tags volume)" \
            --user-data "file://$VM_DIR/vm-setup.sh" --query 'Instances[0].InstanceId' --output text 2>&1)"; then
    break
  fi
  case "$ID" in
    *"Invalid IAM Instance Profile"*|*InvalidParameterValue*) vm_say "  the instance profile is not ready yet; again in 10 s ($attempt/8)"; ID=""; sleep 10 ;;
    *) vm_stop "run-instances: $ID" ;;
  esac
done
[ -n "$ID" ] || vm_stop "run-instances did not accept the instance profile after 8 tries; run this script again"
vm_say "  $ID starting ..."
aws ec2 wait instance-running --instance-ids "$ID"
IP="$(vm_ip "$ID")"
vm_say "  $ID running, public IP $IP"
if vm_wait_ssh "$IP"; then vm_say "  SSH answers"; else vm_say "  WARN: SSH does not answer yet (05-connect-vm.sh --setup waits longer)"; fi

echo
echo "OK: $VM_NAME is up. Its first boot installs the toolset now (about 2 minutes)."
echo "Next: bash $(dirname "$0")/04-enable-auto-stop.sh"
echo "      bash $(dirname "$0")/05-connect-vm.sh --setup"
