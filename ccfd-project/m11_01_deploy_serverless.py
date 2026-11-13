"""
MLOps 11 Step-01: the same Approved package on a serverless endpoint, ccfd-serverless.

    python m11_01_deploy_serverless.py --package-version 1

No instance type or count: ServerlessConfig 2048 MB, MaxConcurrency 1. SageMaker starts compute per call, scales to
0 when idle and bills per millisecond. It runs our image as its own non-root user on a read-only disk (MLOps 02).
The same names as the commands of the README (Option 1): what exists is kept, so a re-run (or a run after the
commands) only waits until ccfd-serverless is InService.
"""

import argparse

import boto3
from botocore.exceptions import ClientError

from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Deploy an Approved package to ccfd-serverless.")
parser.add_argument("--package-version", required=True)
args = parser.parse_args()

say = lambda text: print(config.mask(str(text)), flush=True)
sm, tags = boto3.client("sagemaker", region_name=config.REGION), config.load().aws_tags()
model, cfg = f"ccfd-serverless-model-v{args.package_version}", f"ccfd-serverless-config-v{args.package_version}"
try:
    sm.describe_model(ModelName=model)
except ClientError:
    endpoints.create_model_from_package(model, endpoints.package_arn(args.package_version), tags=tags, log=say)
try:
    sm.describe_endpoint_config(EndpointConfigName=cfg)
except ClientError:
    sm.create_endpoint_config(EndpointConfigName=cfg, Tags=tags, ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": model,
        "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 1}}])
    say(f"endpoint config {cfg}: serverless, 2048 MB, MaxConcurrency 1")
if endpoints.endpoint_exists("ccfd-serverless"):
    say("endpoint ccfd-serverless exists: waiting for InService")
    endpoints.wait_for_endpoint("ccfd-serverless", log=say)
else:
    endpoints.deploy("ccfd-serverless", cfg, tags=tags, log=say)
