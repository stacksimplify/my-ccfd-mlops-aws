#!/usr/bin/env bash
# vm-setup.sh: installs the course toolset on the VM. Runs ON THE VM as root, never on your laptop.
# It is the VM's user-data: run-instances --user-data file://scripts/vm/vm-setup.sh (Section 00-B Step-01-07, Option 1)
# or 03-create-vm.sh (Option 2) hands it to the VM, so it runs once by itself at the first boot (about 2 minutes).
# It keeps a copy in /opt/mlops/; safe to run again by hand on the VM:  sudo bash /opt/mlops/vm-setup.sh
#
# It installs (the versions are pinned just below): Docker Engine + the buildx plugin, Terraform, the GitHub CLI gh,
# git, jq, lsof, rsync, tmux, unzip; Miniconda in /home/ec2-user/miniconda3 with the empty env mlops-aws-env (Python
# only; the course packages come with pip in Section 00-B Step-01-11); and /usr/local/bin/mlops-vm-check (the tool
# table). It sets the AWS CLI default region (us-east-1, output json) for ec2-user, turns off Terraform's update check
# in every shell (CHECKPOINT_DISABLE=1: the course pins Terraform, so no "out of date" notice), and writes the file
# /etc/mlops-course-vm: the course scripts read it to know they run on your EC2 VM.
# The AWS CLI v2 comes with Amazon Linux 2023. Log: /var/log/mlops-vm-setup.log; its last line says DONE or FAILED.

# The pinned versions (the newest release at least 2 weeks old on 2026-10-22, gh on 2026-10-25; Terraform: its line)
BUILDX_VERSION="v0.38.0"
MINICONDA_FILE="Miniconda3-py314_26.7.1-1-Linux-x86_64.sh"
MINICONDA_SHA256="e8b25b92b262499141c5bd57a98d3c008024185fa951494b9cd9b6d94e72338b"
PYTHON_VERSION="3.14.8"
TERRAFORM_VERSION="1.15.6"   # the version the course's Terraform layers were tested with (each needs ~> 1.15.0)
TERRAFORM_SHA256="a7150d3b0e1b5c466ad42e8c499954a3c54645f8b56b385fa025d34f7e88faa9"   # terraform_1.15.6_linux_amd64.zip
GH_VERSION="2.102.0"
GH_SHA256="eba164e2b9be93d210690010436ff4fb7834221dd902b4fbac37a61d0124520e"          # gh_2.102.0_linux_amd64.rpm
MLOPS_REGION="us-east-1"        # the AWS CLI default region for ec2-user: aws, Terraform and boto3 all read it

set -euo pipefail

LOG=/var/log/mlops-vm-setup.log
exec > >(tee -a "$LOG") 2>&1
STEP="start"
trap 'echo "MLOPS VM SETUP FAILED at: $STEP ($(date -u +%FT%TZ)). Fix, then run: sudo bash /opt/mlops/vm-setup.sh"' ERR

step() { STEP="$1"; echo; echo "== $(date -u +%T) $1"; }

step "guard: Amazon Linux 2023, x86_64, root"
# shellcheck source=/dev/null  # the VM's own file
. /etc/os-release
[ "${ID:-}" = amzn ] && [ "${VERSION_ID:-}" = 2023 ] || { echo "not Amazon Linux 2023"; false; }
[ "$(uname -m)" = x86_64 ] || { echo "not x86_64"; false; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; false; }
U=ec2-user
H=/home/$U

step "the marker file /etc/mlops-course-vm"
echo "mlops-vm" > /etc/mlops-course-vm
chmod 644 /etc/mlops-course-vm

step "a copy of this script and the pinned versions in /opt/mlops"
mkdir -p /opt/mlops
cp "$0" /opt/mlops/vm-setup.sh 2>/dev/null || true
cat > /opt/mlops/vm-versions.env <<EOF
BUILDX_VERSION="$BUILDX_VERSION"
PYTHON_VERSION="$PYTHON_VERSION"
TERRAFORM_VERSION="$TERRAFORM_VERSION"
GH_VERSION="$GH_VERSION"
MLOPS_REGION="$MLOPS_REGION"
EOF

step "system packages (dnf)"
dnf install -y -q docker git jq lsof rsync tmux unzip

step "Docker: service on at boot, $U in the docker group"
systemctl enable --now docker
usermod -aG docker "$U"

step "Docker plugin: buildx $BUILDX_VERSION"
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-buildx \
  "https://github.com/docker/buildx/releases/download/$BUILDX_VERSION/buildx-$BUILDX_VERSION.linux-amd64"
chmod 755 /usr/local/lib/docker/cli-plugins/docker-buildx

step "Terraform $TERRAFORM_VERSION (checksum checked)"
TF_ZIP="terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
curl -fsSL -o "/tmp/$TF_ZIP" "https://releases.hashicorp.com/terraform/$TERRAFORM_VERSION/$TF_ZIP"
(cd /tmp && echo "$TERRAFORM_SHA256  $TF_ZIP" | sha256sum -c -)
unzip -o -q "/tmp/$TF_ZIP" terraform -d /usr/local/bin
rm -f "/tmp/$TF_ZIP"

step "no Terraform update check: CHECKPOINT_DISABLE=1 in every shell (/etc/profile.d/mlops-course.sh)"
cat > /etc/profile.d/mlops-course.sh <<'EOF'
# The course pins Terraform 1.15.6: no update check, so no "Your version of Terraform is out of date" notice
export CHECKPOINT_DISABLE=1
EOF
chmod 644 /etc/profile.d/mlops-course.sh

step "GitHub CLI gh $GH_VERSION (checksum checked)"
GH_RPM="gh_${GH_VERSION}_linux_amd64.rpm"
curl -fsSL -o "/tmp/$GH_RPM" "https://github.com/cli/cli/releases/download/v$GH_VERSION/$GH_RPM"
(cd /tmp && echo "$GH_SHA256  $GH_RPM" | sha256sum -c -)
dnf install -y -q "/tmp/$GH_RPM"
rm -f "/tmp/$GH_RPM"

step "the AWS CLI default region for $U: $MLOPS_REGION (output json)"
sudo -u "$U" -H aws configure set region "$MLOPS_REGION"
sudo -u "$U" -H aws configure set output json

step "Miniconda ($MINICONDA_FILE, checksum checked) and the empty env mlops-aws-env (Python $PYTHON_VERSION)"
sudo -u "$U" -H bash -s <<EOF
set -euo pipefail
cd "$H"
if [ ! -x "$H/miniconda3/bin/conda" ]; then
  curl -fsSLO "https://repo.anaconda.com/miniconda/$MINICONDA_FILE"
  echo "$MINICONDA_SHA256  $MINICONDA_FILE" | sha256sum -c -
  bash "$MINICONDA_FILE" -b -p "$H/miniconda3"
  rm -f "$MINICONDA_FILE"
  "$H/miniconda3/bin/conda" init bash
fi
if [ ! -x "$H/miniconda3/envs/mlops-aws-env/bin/python" ]; then
  "$H/miniconda3/bin/conda" create -y -q -n mlops-aws-env python=$PYTHON_VERSION --override-channels -c conda-forge
fi
EOF

step "mlops-vm-check"
cat > /usr/local/bin/mlops-vm-check <<'CHECK'
#!/usr/bin/env bash
# mlops-vm-check: your EC2 VM's tool table (05-connect-vm.sh --setup runs it). Exit 0 = all OK.
. /opt/mlops/vm-versions.env
E=/home/ec2-user/miniconda3/envs/mlops-aws-env/bin
ok=0; n=0
row() {   # row <tool> <expected or -> <found>
  local want="$1" exp="$2" got="$3" res=OK
  n=$((n + 1))
  if [ -z "$got" ]; then res=MISSING; elif [ "$exp" != "-" ] && [ "$got" != "$exp" ]; then res=WRONG; fi
  [ "$res" = OK ] && ok=$((ok + 1))
  printf '%-12s %-12s %-14s %s\n' "$want" "$exp" "${got:--}" "$res"
}
printf '%-12s %-12s %-14s %s\n' Tool Expected "On the VM" Check
row "AWS CLI"   -                     "$(aws --version 2>/dev/null | awk '{print $1}' | cut -d/ -f2)"
row "AWS region" "$MLOPS_REGION"         "$(aws configure get region 2>/dev/null)"
row Docker      -                     "$(docker version --format '{{.Server.Version}}' 2>/dev/null || sudo docker version --format '{{.Server.Version}}' 2>/dev/null)"
row buildx      "$BUILDX_VERSION"     "$(docker buildx version 2>/dev/null | awk '{print $2}')"
row Terraform   "$TERRAFORM_VERSION"  "$(CHECKPOINT_DISABLE=1 terraform version 2>/dev/null | sed -n '1s/^Terraform v//p')"
row git         -                     "$(git --version 2>/dev/null | awk '{print $3}')"
row gh          "$GH_VERSION"         "$(gh --version 2>/dev/null | awk 'NR == 1 {print $3}')"
row jq          -                     "$(jq --version 2>/dev/null | sed 's/^jq-//')"
row lsof        -                     "$(lsof -v 2>&1 | awk '/revision/ {print $2; exit}')"
row tmux        -                     "$(tmux -V 2>/dev/null | awk '{print $2}')"
row conda       -                     "$(/home/ec2-user/miniconda3/bin/conda --version 2>/dev/null | awk '{print $2}')"
row Python      "$PYTHON_VERSION"     "$($E/python --version 2>/dev/null | awk '{print $2}')"
echo
if [ "$ok" = "$n" ]; then echo "Verdict: OK, your EC2 VM is ready ($ok of $n)"; exit 0; fi
echo "Verdict: NOT READY ($ok of $n OK). Run on the VM: sudo bash /opt/mlops/vm-setup.sh"; exit 1
CHECK
chmod 755 /usr/local/bin/mlops-vm-check

trap - ERR
echo
echo "MLOPS VM SETUP DONE $(date -u +%FT%TZ)"
