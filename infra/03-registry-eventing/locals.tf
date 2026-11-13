# Who and where: the account ID is read at plan time and never typed into any file.
data "aws_caller_identity" "current" {}

locals {
  region     = "us-east-1"
  account_id = data.aws_caller_identity.current.account_id

  topic_name          = "ccfd-mlops-alerts"
  rule_name           = "ccfd-registry-state-change"
  model_package_group = "ccfd-fraud-detector"
}
