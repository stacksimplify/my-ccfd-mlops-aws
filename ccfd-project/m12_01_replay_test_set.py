"""
MLOps 12 Step-01 and Step-02: the labeled probe replay. Real payments get their fraud labels weeks later, so during
a release a team sends a fixed labeled set through the endpoint and alarms on its result. Every 3 seconds this sends
the probe set (the gate holdout of MLOps 09: 2,000 rows, seed 301, no labels sent) to ccfd-prod as one JSON Lines
request, scores the answers with the labels kept here, prints the model and its Net Benefit the first time each
answer is seen, and publishes NetBenefit (namespace MLOps/CCFD, EndpointName=ccfd-prod, Source=TestSetReplay): the
metric the alarm ccfd-prod-rollout-netbenefit watches. When --minutes is over it says so: with no data the alarm
cannot fire, so start it again before a canary.

    python m12_01_replay_test_set.py [--minutes 30]       (Ctrl+C to stop)
"""

import argparse
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

import m12_common as m12
from mlops_utils import config

parser = argparse.ArgumentParser(description="Replay the labeled probe set against ccfd-prod.")
parser.add_argument("--minutes", type=float, default=30)
args = parser.parse_args()

cw, seen, end = boto3.client("cloudwatch", region_name=config.REGION), set(), time.time() + 60 * args.minutes
try:
    while time.time() < end:
        try:
            _, model, net_benefit = m12.score()
            cw.put_metric_data(Namespace="MLOps/CCFD", MetricData=[{
                "MetricName": "NetBenefit", "Value": net_benefit, "Unit": "None",
                "Dimensions": [{"Name": "EndpointName", "Value": m12.EP},
                               {"Name": "Source", "Value": "TestSetReplay"}]}])
            if (model, net_benefit) not in seen:    # the same model name can give another number (another model file)
                m12.say(f"replay: {m12.EP} {model} ${net_benefit:,}")
                seen.add((model, net_benefit))
        except (ClientError, BotoCoreError) as error:
            m12.say(f"replay call failed: {type(error).__name__}")
        time.sleep(3)
    m12.say(f"replay ended after {args.minutes:g} minutes: start it again before a canary (no data = no alarm)")
except KeyboardInterrupt:
    m12.say("replay stopped (Ctrl+C)")
