# Who and where: the account ID is read at plan time and never typed into any file.
data "aws_caller_identity" "current" {}

locals {
  region     = "us-east-1"
  account_id = data.aws_caller_identity.current.account_id

  # The 3 course buckets: ccfd-mlops-<purpose>-<suffix> (the state bucket is the 4th, in infra/00-state-bucket)
  bucket_purposes = toset(["dvc", "mlflow", "jobs"])
  bucket_names    = { for p in local.bucket_purposes : p => "ccfd-mlops-${p}-${var.suffix}" }

  # The 2 image repositories: one image for serving, one for training and processing jobs
  ecr_repositories = toset(["ccfd-serving", "ccfd-training"])

  execution_role_name = "ccfd-sagemaker-execution-role"
}
