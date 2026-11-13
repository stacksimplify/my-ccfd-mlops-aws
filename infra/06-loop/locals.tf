# Who and where: the account ID is read at plan time and never typed into any file.
data "aws_caller_identity" "current" {}

locals {
  region     = "us-east-1"
  account_id = data.aws_caller_identity.current.account_id

  rule_name    = "ccfd-netbenefit-drop-starts-retrain"
  role_name    = "ccfd-loop-events-role"
  alarm_name   = "ccfd-netbenefit-drop"
  pipeline_arn = "arn:aws:sagemaker:${local.region}:${local.account_id}:pipeline/${var.pipeline_name}"

  # The pipeline parameters this rule passes. An empty value is left out, so the pipeline uses its own default.
  # The gate's champion is always the newest Approved package.
  pipeline_parameters = {
    for name, value in {
      InputDataS3Uri      = var.input_data_s3_uri
      DataVersion         = var.data_version
      TrainingConfigS3Uri = var.training_config_s3_uri
      GateHoldoutS3Uri    = var.gate_holdout_s3_uri
    } : name => value if value != ""
  }
}
