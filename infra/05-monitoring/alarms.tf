# The 2 monitoring alarms. Each one looks at the highest value of its metric in a 5-minute window and notifies
# the alerts topic when it crosses the line. With no data (for example no labels yet) it stays INSUFFICIENT_DATA.

# Input drift: the largest PSI over all features in a month. 0.10 or more = at least a moderate shift.
resource "aws_cloudwatch_metric_alarm" "maxpsi_high" {
  alarm_name          = "ccfd-maxpsi-high"
  alarm_description   = "Input drift: MaxPSI at or above 0.10 in ${local.namespace} (a moderate or larger shift in at least one feature)."
  namespace           = local.namespace
  metric_name         = "MaxPSI"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 0.10
  treat_missing_data  = "missing"
  alarm_actions       = [local.alerts_topic_arn]
}

# Business impact: how far Net Benefit fell, in percent (the drift job computes it). 25 or more = retrain.
# infra/06-loop starts the retraining pipeline when this alarm goes to ALARM.
resource "aws_cloudwatch_metric_alarm" "netbenefit_drop" {
  alarm_name          = "ccfd-netbenefit-drop"
  alarm_description   = "Business impact: NetBenefitDropPct at or above 25 in ${local.namespace} (Net Benefit fell by a quarter or more)."
  namespace           = local.namespace
  metric_name         = "NetBenefitDropPct"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 25
  treat_missing_data  = "missing"
  alarm_actions       = [local.alerts_topic_arn]
}
