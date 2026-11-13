"""
MLOps 10 Step-02: deploy one Approved package of ccfd-fraud-detector to the real-time endpoint ccfd-rt.

    python m10_02_deploy_realtime.py --package-version 1

It makes the model ccfd-rt-model-v<N> FROM THE PACKAGE (no image or S3 path typed: the package holds both; only
Approved packages deploy), the config ccfd-rt-config-v<N> (variant AllTraffic, ml.m5.large x 1) and the endpoint
ccfd-rt, all with the course tags, then waits for InService. About 0.115 USD per hour until you delete it
(bash ../scripts/m10_05_delete_realtime.sh).
"""

import argparse

from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Deploy an Approved package to ccfd-rt.")
parser.add_argument("--package-version", required=True)
args = parser.parse_args()

say = lambda text: print(config.mask(str(text)), flush=True)
say(f"Approved versions of {endpoints.GROUP}: {endpoints.approved_versions()}")
endpoints.deploy_package("ccfd-rt", args.package_version, tags=config.load().aws_tags(), log=say)
described = endpoints.describe("ccfd-rt")
say(f"ccfd-rt: {described['EndpointStatus']}, config {described['EndpointConfigName']}, "
    f"instances {described['ProductionVariants'][0]['CurrentInstanceCount']}")
