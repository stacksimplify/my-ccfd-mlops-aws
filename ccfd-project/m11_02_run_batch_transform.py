"""
MLOps 11 Step-02: score month_<N> offline with Batch Transform: no endpoint; an instance starts, scores the file
and stops.

    python m11_02_run_batch_transform.py --month 1 --package-version 1

Model ccfd-batch-model-v<N> from the Approved package; ml.m5.large x 1; application/jsonlines, SplitType Line,
BatchStrategy MultiRecord, MaxPayloadInMB 6, AssembleWith Line: one output line per input line, in order.
"""

import argparse
import time
from datetime import UTC, datetime

import boto3
from botocore.exceptions import ClientError

from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Run a Batch Transform job on one month.")
parser.add_argument("--month", type=int, default=1)
parser.add_argument("--package-version", required=True)
args = parser.parse_args()

say = lambda text: print(config.mask(str(text)), flush=True)
sm, env = boto3.client("sagemaker", region_name=config.REGION), config.load()
bucket, model = env.get("MLOPS_JOBS_BUCKET"), f"ccfd-batch-model-v{args.package_version}"
try:
    sm.describe_model(ModelName=model)
except ClientError:
    endpoints.create_model_from_package(model, endpoints.package_arn(args.package_version), tags=env.aws_tags(), log=say)
job = f"ccfd-batch-month-{args.month}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
sm.create_transform_job(
    TransformJobName=job, ModelName=model, Tags=env.aws_tags(), BatchStrategy="MultiRecord", MaxPayloadInMB=6,
    TransformInput={"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                                    "S3Uri": f"s3://{bucket}/batch/month_{args.month}/input/"}},
                    "ContentType": "application/jsonlines", "SplitType": "Line"},
    TransformOutput={"S3OutputPath": f"s3://{bucket}/batch/month_{args.month}/output/",
                     "Accept": "application/jsonlines", "AssembleWith": "Line"},
    TransformResources={"InstanceType": "ml.m5.large", "InstanceCount": 1})
say(f"transform job {job} started: ml.m5.large x 1")
while True:
    described = sm.describe_transform_job(TransformJobName=job)
    if described["TransformJobStatus"] in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(30)
seconds = (described["TransformEndTime"] - described["TransformStartTime"]).total_seconds()
say(f"transform job {job}: {described['TransformJobStatus']} in {seconds:.0f} s "
    f"{described.get('FailureReason', '')}".rstrip())
