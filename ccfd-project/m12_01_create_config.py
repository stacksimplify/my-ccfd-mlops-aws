"""
MLOps 12 Step-01: a new endpoint config for the canary: an Approved package on 2 instances, variant AllTraffic
(a canary keeps the variant name).

    python m12_01_create_config.py --package-version 1 --name ccfd-prod-config-v1-canary
"""

import argparse

import m12_common as m12
from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Create a canary target config for ccfd-prod.")
parser.add_argument("--package-version", required=True)
parser.add_argument("--name", required=True)
args = parser.parse_args()
model = f"ccfd-prod-model-v{args.package_version}"
if not m12.model_exists(model):
    endpoints.create_model_from_package(model, endpoints.package_arn(args.package_version),
                                        tags=config.load().aws_tags(), log=m12.log)
m12.make_config(args.name, [m12.variant("AllTraffic", model)])
