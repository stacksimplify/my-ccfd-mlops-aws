"""
MLOps 10 Step-05: update ccfd-rt to another endpoint config (blue/green: the new fleet starts, traffic moves, the
old fleet stops), or roll back to a config you kept.

    python m10_05_update_endpoint.py --package-version 1 --instances 2    a new config ccfd-rt-config-v1-x2
    python m10_05_update_endpoint.py --config ccfd-rt-config-v1          roll back to the kept config

Configs never change and are kept, so a rollback is one update call. It prints which package serves at the end.
"""

import argparse

import boto3

from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Update or roll back ccfd-rt.")
parser.add_argument("--package-version")
parser.add_argument("--instances", type=int, default=1)
parser.add_argument("--config")
args = parser.parse_args()

say = lambda text: print(config.mask(str(text)), flush=True)
sm = boto3.client("sagemaker", region_name=config.REGION)
target = args.config
if target is None:
    model, base = endpoints.model_names("ccfd-rt", args.package_version)
    target = f"{base}-x{args.instances}" if args.instances != 1 else base
    if not endpoints._find("DescribeModel", sm.describe_model, "Could not find model", ModelName=model):
        endpoints.create_model_from_package(model, endpoints.package_arn(args.package_version),
                                            tags=config.load().aws_tags(), log=say)
    if not endpoints._find("DescribeEndpointConfig", sm.describe_endpoint_config,
                           "Could not find endpoint configuration", EndpointConfigName=target):
        endpoints.create_endpoint_config(target, model, instance_count=args.instances, tags=config.load().aws_tags(),
                                         log=say)
endpoints.update("ccfd-rt", target, log=say)
serving = endpoints.describe("ccfd-rt")
cfg = sm.describe_endpoint_config(EndpointConfigName=serving["EndpointConfigName"])["ProductionVariants"][0]
package = sm.describe_model(ModelName=cfg["ModelName"])["PrimaryContainer"].get("ModelPackageName", "?")
say(f"ccfd-rt: {serving['EndpointStatus']}; config {serving['EndpointConfigName']} (x{cfg['InitialInstanceCount']}); "
    f"package version {package.rsplit('/', 1)[-1]}")
