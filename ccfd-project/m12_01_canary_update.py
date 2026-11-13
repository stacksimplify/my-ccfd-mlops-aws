"""
MLOps 12 Step-01 and Step-02: update ccfd-prod with deployment guardrails: a canary of 1 instance takes some
traffic, bakes 300 s, then the rest moves and bakes 120 s; if either of the 2 rollout alarms fires, SageMaker rolls
back to the old fleet by itself.

    python m12_01_canary_update.py --config ccfd-prod-config-v1-canary
"""

import argparse
import time

import boto3
from botocore.exceptions import ClientError

import m12_common as m12
from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Canary update of ccfd-prod with auto-rollback.")
parser.add_argument("--config", required=True)
args = parser.parse_args()

if m12.already_serves(args.config):
    raise SystemExit(0)
before, start = m12.current_config(), time.time()
try:
    endpoints.update(m12.EP, args.config, deployment_config=endpoints.canary_deployment_config(), log=m12.log,
                     timeout_seconds=3600)
except (ClientError, RuntimeError) as error:   # a rolled-back update ends with an error status
    m12.say(f"update ended: {type(error).__name__}: {config.mask(str(error))[:160]}")
now = m12.current_config()
result = "SUCCEEDED" if now == args.config else f"ROLLED BACK (still {now})"
m12.say(f"canary to {args.config} ended after {time.time() - start:.0f} s: {result}; before: {before}")
cw = boto3.client("cloudwatch", region_name=config.REGION)
for item in cw.describe_alarm_history(AlarmName="ccfd-prod-rollout-netbenefit", HistoryItemType="StateUpdate",
                                      MaxRecords=3)["AlarmHistoryItems"]:
    m12.say(f"alarm history: {item['Timestamp']:%H:%M:%S} {item['HistorySummary']}")
