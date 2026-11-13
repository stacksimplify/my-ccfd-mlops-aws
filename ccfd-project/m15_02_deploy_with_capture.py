"""
MLOps 15: ccfd-prod with data capture (every request and answer saved to S3; 100 percent sampling).

    python m15_02_deploy_with_capture.py [--package-version 1]

DataCaptureConfig: EnableCapture true, InitialSamplingPercentage 100, s3://<jobs bucket>/data-capture/, CaptureOptions
Input + Output, JsonContentTypes application/json and application/jsonlines (so the data is stored as JSON text).
"""
import argparse

from mlops_utils import endpoints

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("--package-version", type=int, default=None,
                    help="default: the original champion (the oldest Approved version)")
args = parser.parse_args()
version = args.package_version or endpoints.approved_versions()[0]
endpoints.deploy_package(endpoints.PROD_ENDPOINT, version, instance_count=2,
                         data_capture=endpoints.data_capture_config(), log=endpoints._say)
print(f"ccfd-prod serves package version {version} with data capture on")
