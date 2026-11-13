"""
MLOps 09 Step-01: register a trained model as a deployable package in the SageMaker group ccfd-fraud-detector.

    python m09_01_register_package.py --model-tar s3://.../output/model.tar.gz --training-job <job> \
        --processing-job <job> --dvc-md5 <md5> --description "<text>"

The group is made with boto3 when it is missing. The package = our serving image (MLOPS_SERVING_IMAGE_URI) +
the model.tar.gz + its metrics (evaluation.json, uploaded next to it as ModelMetrics) + PendingManualApproval.
Lineage goes into the package metadata. When the training job logged an MLflow run, that run is registered as a
version of ccfd-fraud-model too, and its number is stored in the package (two registries, one story).
"""

import argparse
import json
import subprocess

import boto3
from botocore.exceptions import ClientError

import m09_common as m09
from mlops_utils import config, mlflow_layer

parser = argparse.ArgumentParser(description="Register a model.tar.gz as a ccfd-fraud-detector package.")
parser.add_argument("--model-tar", required=True)
parser.add_argument("--description", required=True)
parser.add_argument("--training-job", default="none")
parser.add_argument("--processing-job", default="none")
parser.add_argument("--dvc-md5", default="none")
args = parser.parse_args()

env = config.load()
image = env.get("MLOPS_SERVING_IMAGE_URI")
if not image:
    raise SystemExit("STOP: .env.mlops has no MLOPS_SERVING_IMAGE_URI: push the serving image (MLOps 06 Step-05), then "
                     "bash ../scripts/create-env.sh --write-env-only")
sm = m09.sm()
try:
    sm.describe_model_package_group(ModelPackageGroupName=m09.GROUP)
except ClientError:
    sm.create_model_package_group(ModelPackageGroupName=m09.GROUP, Tags=env.aws_tags(),
                                  ModelPackageGroupDescription="Fraud model packages (Net Benefit decides).")
    m09.say(f"Group {m09.GROUP}: created")

_, evaluation = m09.tar_files(args.model_tar)
net_benefit = int(evaluation["business_metrics"]["net_benefit"])
metrics_uri = args.model_tar.rsplit("/", 1)[0] + "/evaluation.json"
bucket, key = metrics_uri.removeprefix("s3://").split("/", 1)
boto3.client("s3", region_name=config.REGION).put_object(Bucket=bucket, Key=key, Body=json.dumps(evaluation).encode())

repository, tag = image.rsplit(":", 1)
digest = boto3.client("ecr", region_name=config.REGION).describe_images(
    repositoryName=repository.rsplit("/", 1)[1], imageIds=[{"imageTag": tag}])["imageDetails"][0]["imageDigest"]
git_sha = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True,
                         check=False).stdout.strip() or "not-a-git-repo"
metadata = {"net_benefit": str(net_benefit), "training_job": args.training_job, "processing_job": args.processing_job,
            "dvc_md5": args.dvc_md5, "image_digest": digest, "git_sha": git_sha}

mlflow_version = None
if args.training_job != "none" and env.get("MLOPS_MLFLOW_APP_ARN"):
    mlflow_layer.connect(env.get("MLOPS_MLFLOW_APP_ARN"))
    run = mlflow_layer.latest_run("training_job", args.training_job)
    if run is not None:
        mlflow_version = mlflow_layer.register(run.info.run_id, {"training_job": args.training_job}).version
        metadata.update({"mlflow_run_id": run.info.run_id, "mlflow_version": str(mlflow_version)})

answer = sm.create_model_package(
    ModelPackageGroupName=m09.GROUP, ModelPackageDescription=args.description,
    InferenceSpecification={
        "Containers": [{"Image": image, "ModelDataUrl": args.model_tar}],
        "SupportedContentTypes": ["application/json", "application/jsonlines"],
        "SupportedResponseMIMETypes": ["application/json", "application/jsonlines"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
        "SupportedTransformInstanceTypes": ["ml.m5.large"]},
    ModelMetrics={"ModelQuality": {"Statistics": {"ContentType": "application/json", "S3Uri": metrics_uri}}},
    ModelApprovalStatus="PendingManualApproval", CustomerMetadataProperties=metadata)
arn = answer["ModelPackageArn"]
described = sm.describe_model_package(ModelPackageName=arn)
# One fact per line, each with its label
m09.say(f"Registered package {m09.GROUP}/{arn.rsplit('/', 1)[1]}")
m09.say(f"  Package status:   {described['ModelPackageStatus']}")
m09.say(f"  Approval status:  {described['ModelApprovalStatus']}")
m09.say(f"  Net Benefit:      {net_benefit:,}")
m09.say(f"  MLflow model:     " + (f"{mlflow_layer.MODEL_NAME}, version {mlflow_version}" if mlflow_version else "none"))
