# infra/00-state-bucket: the S3 bucket that holds the Terraform state of every other layer (01 to 06).
# This layer keeps its OWN state in this folder on your EC2 VM (local state, terraform.tfstate, never committed),
# because the bucket it makes does not exist before the first apply. The course keeps it on your EC2 VM until
# MLOps 16, so the VM is deleted only after the full exit.
#
#   terraform init
#   terraform plan
#   terraform apply
#
# It writes infra/backend.hcl (never committed); every other layer then runs:
#   terraform init -backend-config=<path to infra/backend.hcl>   (../backend.hcl from a layer folder next to it)
terraform {
  required_version = "~> 1.15.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.68.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.9.0"
    }
  }
}
