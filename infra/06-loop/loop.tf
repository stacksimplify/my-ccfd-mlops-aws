# Closing the loop: when the Net Benefit alarm (infra/05-monitoring) goes to ALARM, EventBridge starts the
# retraining pipeline with the parameters below. The pipeline itself is made by m13_01_start_pipeline.py (boto3),
# so it must exist before this layer is applied.

resource "aws_cloudwatch_event_rule" "netbenefit_drop_starts_retrain" {
  name        = local.rule_name
  description = "${local.alarm_name} in ALARM starts the pipeline ${var.pipeline_name}."

  event_pattern = jsonencode({
    source        = ["aws.cloudwatch"]
    "detail-type" = ["CloudWatch Alarm State Change"]
    detail = {
      alarmName = [local.alarm_name]
      state = {
        value = ["ALARM"]
      }
    }
  })
}

# The role EventBridge uses to start the pipeline: it may start that one pipeline and do nothing else.
data "aws_iam_policy_document" "trust_events" {
  statement {
    sid     = "EventBridgeAssumesTheRole"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "loop" {
  name               = local.role_name
  description        = "EventBridge starts the ccfd retraining pipeline when the Net Benefit alarm fires."
  assume_role_policy = data.aws_iam_policy_document.trust_events.json
}

data "aws_iam_policy_document" "loop" {
  statement {
    sid       = "StartThePipelineOnly"
    actions   = ["sagemaker:StartPipelineExecution"]
    resources = [local.pipeline_arn]
  }
}

resource "aws_iam_role_policy" "loop" {
  name   = "${local.role_name}-policy"
  role   = aws_iam_role.loop.id
  policy = data.aws_iam_policy_document.loop.json
}

resource "aws_cloudwatch_event_target" "start_pipeline" {
  rule      = aws_cloudwatch_event_rule.netbenefit_drop_starts_retrain.name
  target_id = "retraining-pipeline"
  arn       = local.pipeline_arn
  role_arn  = aws_iam_role.loop.arn

  sagemaker_pipeline_target {
    dynamic "pipeline_parameter_list" {
      for_each = local.pipeline_parameters
      content {
        name  = pipeline_parameter_list.key
        value = pipeline_parameter_list.value
      }
    }
  }

  # Checked when the target is created or changed, never on terraform destroy (which needs none of these values)
  lifecycle {
    precondition {
      condition     = var.input_data_s3_uri != ""
      error_message = "Set input_data_s3_uri (the S3 URI of the retrain data file) to apply this layer."
    }
    precondition {
      condition     = var.data_version != ""
      error_message = "Set data_version (the label of the retrain data, for example its DVC md5) to apply this layer."
    }
    precondition {
      condition     = var.gate_holdout_s3_uri != ""
      error_message = "Set gate_holdout_s3_uri (the S3 URI of the labeled month the gate scores on) to apply this layer."
    }
  }

  depends_on = [aws_iam_role_policy.loop]
}
