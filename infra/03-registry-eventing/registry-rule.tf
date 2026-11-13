# Every change in the model registry group ccfd-fraud-detector (a new version, Approved, Rejected) is an
# EventBridge event "SageMaker Model Package State Change". This rule sends it to the alerts topic as one line.
resource "aws_cloudwatch_event_rule" "registry_state_change" {
  name        = local.rule_name
  description = "Model registry changes in ${local.model_package_group} to the ${local.topic_name} topic."

  event_pattern = jsonencode({
    source        = ["aws.sagemaker"]
    "detail-type" = ["SageMaker Model Package State Change"]
    detail = {
      ModelPackageGroupName = [local.model_package_group]
    }
  })
}

resource "aws_cloudwatch_event_target" "registry_state_change_to_sns" {
  rule      = aws_cloudwatch_event_rule.registry_state_change.name
  target_id = "alerts-topic"
  arn       = aws_sns_topic.alerts.arn

  input_transformer {
    input_paths = {
      group   = "$.detail.ModelPackageGroupName"
      version = "$.detail.ModelPackageVersion"
      status  = "$.detail.ModelApprovalStatus"
      package = "$.detail.ModelPackageArn"
    }
    input_template = "\"Model registry: <group> version <version> is now <status> (<package>).\""
  }

  depends_on = [aws_sns_topic_policy.alerts]
}
