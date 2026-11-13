# Read with: terraform output -raw <name>

output "dashboard_name" {
  description = "The CloudWatch dashboard (console: CloudWatch, Dashboards)."
  value       = aws_cloudwatch_dashboard.mlops.dashboard_name
}

output "maxpsi_alarm_name" {
  description = "The drift alarm (MaxPSI at or above 0.10)."
  value       = aws_cloudwatch_metric_alarm.maxpsi_high.alarm_name
}

output "netbenefit_drop_alarm_name" {
  description = "The business alarm (NetBenefitDropPct at or above 25); infra/06-loop listens to it."
  value       = aws_cloudwatch_metric_alarm.netbenefit_drop.alarm_name
}
