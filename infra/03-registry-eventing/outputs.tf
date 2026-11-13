# Read with: terraform output -raw <name>   (scripts/create-env.sh writes them into .env.mlops)

output "alerts_topic_arn" {
  description = "The alerts topic ccfd-mlops-alerts (infra/05-monitoring alarms publish to it too)."
  value       = aws_sns_topic.alerts.arn
}

output "registry_rule_name" {
  description = "The EventBridge rule for model registry changes."
  value       = aws_cloudwatch_event_rule.registry_state_change.name
}

output "alert_email_subscribed" {
  description = "true when an email subscription was made (it stays PendingConfirmation until its link is clicked)."
  value       = length(aws_sns_topic_subscription.alert_email) > 0
}
