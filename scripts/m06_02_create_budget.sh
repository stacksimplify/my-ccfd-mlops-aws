#!/usr/bin/env bash
# m06_02_create_budget.sh: Section 00-B Step-01-04. A 10 USD monthly cost budget for the whole account, with email alerts
# at 80% and 100% of the actual cost and 100% of the forecast. Credits are left out, so the alert fires on real
# spend even when credits pay the bill. Budgets are free (the first 2 action-free budgets of an account).
#
#   export ALERT_EMAIL=<your address>
#   bash scripts/m06_02_create_budget.sh            create it (skipped when ccfd-mlops-monthly already exists)
#
# The address is read from ALERT_EMAIL and never written into a file. AWS emails the alerts; no link to click.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="ccfd-mlops-monthly"

[[ -n "${ALERT_EMAIL:-}" ]] || { echo "STOP: set your alert address first: export ALERT_EMAIL=<your address>"; exit 1; }
NOTIFICATIONS="$(sed "s/ALERT_EMAIL/${ALERT_EMAIL}/g" "$DIR/notifications.json")"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
if aws budgets describe-budget --account-id "$ACCOUNT" --budget-name "$NAME" >/dev/null 2>&1; then
  echo "Budget $NAME already exists: nothing changed."
else
  aws budgets create-budget --account-id "$ACCOUNT" --budget "file://$DIR/budget-10usd.json" \
    --notifications-with-subscribers "$NOTIFICATIONS"
  echo "Budget $NAME created: 10 USD a month, alerts at 80% and 100% actual, 100% forecasted."
fi
aws budgets describe-budget --account-id "$ACCOUNT" --budget-name "$NAME" \
  --query 'Budget.[BudgetName,BudgetLimit.Amount,BudgetLimit.Unit,BudgetType,TimeUnit]' --output text
