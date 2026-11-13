"""
MLOps 12 Step-02: the rollback drill. Before a real release depends on the rollout alarms, a team tests them once
with a model it KNOWS is worse. Ours is package 2 of ccfd-fraud-detector: the data v2 retrain, Rejected in MLOps 09
($19,925 vs the champion's $36,975 on the gate holdout; its training did not converge). Our deploy path deploys only
Approved packages, so the drill builds the bad release by hand: a PLAIN model from the package's image and model data
URL, and its config ccfd-prod-config-drill (AllTraffic, 2 instances). Step-02-02 then sends it through the canary.

    python m12_02_create_weaker_config.py --package-version 2
"""

import argparse

import m12_common as m12
from mlops_utils import config, endpoints

MODEL, CONFIG = "ccfd-prod-model-drill", "ccfd-prod-config-drill"

parser = argparse.ArgumentParser(description="The drill model (a plain model of the Rejected package) and its config.")
parser.add_argument("--package-version", default="2")
args = parser.parse_args()
if not m12.model_exists(MODEL):
    endpoints.create_plain_model(MODEL, endpoints.package_arn(args.package_version), tags=config.load().aws_tags(),
                                 log=m12.log)
m12.make_config(CONFIG, [m12.variant("AllTraffic", MODEL)])
