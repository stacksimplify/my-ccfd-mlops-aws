"""
MLOps 12 Step-01: the production endpoint ccfd-prod: the champion package on 2 instances (a canary needs 2+).

    python m12_01_deploy_prod.py --package-version 1
"""

import argparse

import m12_common as m12
from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Deploy the champion package to ccfd-prod (2 instances).")
parser.add_argument("--package-version", required=True)
args = parser.parse_args()
endpoints.deploy_package(m12.EP, args.package_version, instance_count=2, tags=config.load().aws_tags(), log=m12.log)
m12.say(f"{m12.EP}: InService on {m12.current_config()}")
