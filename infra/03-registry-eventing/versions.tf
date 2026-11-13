# infra/03-registry-eventing: the alerts topic and the rule that reports every model registry change to it.
# State in S3 (the bucket of infra/00-state-bucket), under the key below; the other backend settings come from
# the infra/backend.hcl that infra/00-state-bucket wrote (../backend.hcl when this folder sits next to
# 00-state-bucket, as in the full project; else the path to that file, for example in your MLOps 06 folder):
#
#   terraform init -backend-config=<path to infra/backend.hcl>
#   terraform plan
#   terraform apply
terraform {
  required_version = "~> 1.15.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.68.0"
    }
  }

  backend "s3" {
    key = "mlops/03-registry-eventing/terraform.tfstate"
  }
}
