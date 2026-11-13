# Who and where: the account ID is read at plan time and never typed into any file.
data "aws_caller_identity" "current" {}

locals {
  region     = "us-east-1"
  account_id = data.aws_caller_identity.current.account_id

  # From the course names: infra/01-foundation made the bucket and the execution role
  mlflow_bucket_name  = "ccfd-mlops-mlflow-${var.suffix}"
  mlflow_bucket_arn   = "arn:aws:s3:::${local.mlflow_bucket_name}"
  execution_role_name = "ccfd-sagemaker-execution-role"

  app_name      = "ccfd-mlflow"
  app_role_name = "ccfd-mlflow-app-role"
}

# Read-only lookups of what infra/01-foundation made. If 01 is missing or var.suffix is wrong, terraform plan
# stops here with "not found", before anything is made (the App alone takes about 5 minutes to create).
data "aws_s3_bucket" "mlflow" {
  bucket = local.mlflow_bucket_name
}

data "aws_iam_role" "execution" {
  name = local.execution_role_name
}
