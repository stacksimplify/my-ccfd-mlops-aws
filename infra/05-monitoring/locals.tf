# Who and where: the account ID is read at plan time and never typed into any file.
data "aws_caller_identity" "current" {}

locals {
  region     = "us-east-1"
  account_id = data.aws_caller_identity.current.account_id

  # The alerts topic of infra/03-registry-eventing
  alerts_topic_arn = "arn:aws:sns:${local.region}:${local.account_id}:ccfd-mlops-alerts"

  # The drift job publishes these 3 metrics with no dimensions, one value per month, timestamped when it runs
  namespace      = "MLOps/CCFD"
  dashboard_name = "ccfd-mlops-dashboard"
}
