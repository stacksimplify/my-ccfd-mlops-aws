# Read with: terraform output -raw <name>

output "loop_rule_name" {
  description = "The EventBridge rule: ccfd-netbenefit-drop in ALARM starts the pipeline."
  value       = aws_cloudwatch_event_rule.netbenefit_drop_starts_retrain.name
}

output "loop_role_arn" {
  description = "The role EventBridge uses; it may only start the retraining pipeline."
  value       = aws_iam_role.loop.arn
}

output "pipeline_parameters" {
  description = "The parameters the rule passes to the pipeline (empty ones are left out)."
  value       = local.pipeline_parameters
}
