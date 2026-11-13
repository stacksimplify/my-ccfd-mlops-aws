# One dashboard for the monitoring story: drift (MaxPSI), money (NetBenefit, NetBenefitDropPct) and the 2 alarms.
# The 2 red lines are the alarm thresholds.
# A CloudWatch dashboard cannot carry tags (the only resource in these layers that cannot), so the course tag
# never finds it: scripts/orphan-sweep.sh finds it by its ccfd- name instead.
resource "aws_cloudwatch_dashboard" "mlops" {
  dashboard_name = local.dashboard_name

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Input drift: MaxPSI (alarm at 0.10)"
          region  = local.region
          view    = "timeSeries"
          stat    = "Maximum"
          period  = 300
          metrics = [[local.namespace, "MaxPSI"]]
          annotations = {
            horizontal = [{ label = "ccfd-maxpsi-high", value = 0.10, color = "#d62728" }]
          }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Net Benefit drop in percent (alarm at 25)"
          region  = local.region
          view    = "timeSeries"
          stat    = "Maximum"
          period  = 300
          metrics = [[local.namespace, "NetBenefitDropPct"]]
          annotations = {
            horizontal = [{ label = "ccfd-netbenefit-drop", value = 25, color = "#d62728" }]
          }
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Net Benefit in USD"
          region  = local.region
          view    = "timeSeries"
          stat    = "Maximum"
          period  = 300
          metrics = [[local.namespace, "NetBenefit"]]
        }
      },
      {
        type   = "alarm"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title = "Monitoring alarms"
          alarms = [
            aws_cloudwatch_metric_alarm.maxpsi_high.arn,
            aws_cloudwatch_metric_alarm.netbenefit_drop.arn,
          ]
        }
      },
    ]
  })
}
