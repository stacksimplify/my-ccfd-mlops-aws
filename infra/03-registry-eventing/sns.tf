# The one alerts topic of the course. The registry rule below and the 2 monitoring alarms (infra/05-monitoring)
# publish to it; an email subscription turns each message into an email.
resource "aws_sns_topic" "alerts" {
  name = local.topic_name
}

# Who may publish: this account, the registry rule below (EventBridge), and the course alarms (CloudWatch).
data "aws_iam_policy_document" "alerts_topic" {
  statement {
    sid = "AccountOwnsTheTopic"
    actions = [
      "sns:GetTopicAttributes",
      "sns:SetTopicAttributes",
      "sns:AddPermission",
      "sns:RemovePermission",
      "sns:DeleteTopic",
      "sns:Subscribe",
      "sns:ListSubscriptionsByTopic",
      "sns:Publish",
    ]
    resources = [aws_sns_topic.alerts.arn]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${local.account_id}:root"]
    }
  }

  statement {
    sid       = "RegistryRulePublishes"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alerts.arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [aws_cloudwatch_event_rule.registry_state_change.arn]
    }
  }

  statement {
    sid       = "CourseAlarmsPublish"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alerts.arn]

    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudwatch:${local.region}:${local.account_id}:alarm:ccfd-*"]
    }
  }
}

resource "aws_sns_topic_policy" "alerts" {
  arn    = aws_sns_topic.alerts.arn
  policy = data.aws_iam_policy_document.alerts_topic.json
}

# The email subscription, only when an address is given (var.alert_email; scripts/create-env.sh passes ALERT_EMAIL).
# AWS sends a confirmation email first: only you click its link. terraform destroy removes the subscription
# with the topic.
resource "aws_sns_topic_subscription" "alert_email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
