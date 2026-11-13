"""
MLOps 13 Step-04: follow one pipeline execution back to its data and code.

    python m13_04_follow_lineage.py                       the newest execution of ccfd-retraining-pipeline
    python m13_04_follow_lineage.py --execution-arn <arn>

execution -> its parameters (DataVersion, InputDataS3Uri = the DVC object) -> every step and its job -> the MLflow run
of the training job -> the gate decision -> the registered package, if the gate promoted it. Read only.
"""

import argparse
import json

import boto3
from botocore.exceptions import ClientError

from mlops_utils import config, mlflow_layer

PIPELINE = "ccfd-retraining-pipeline"
say = lambda text: print(config.mask(str(text)), flush=True)

parser = argparse.ArgumentParser(description="Lineage of a pipeline execution.")
parser.add_argument("--execution-arn")
args = parser.parse_args()

sm = boto3.client("sagemaker", region_name=config.REGION)
arn = args.execution_arn or sm.list_pipeline_executions(
    PipelineName=PIPELINE, SortBy="CreationTime", SortOrder="Descending", MaxResults=1)[
    "PipelineExecutionSummaries"][0]["PipelineExecutionArn"]
execution = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
params = {p["Name"]: p["Value"] for p in sm.list_pipeline_parameters_for_execution(PipelineExecutionArn=arn)["PipelineParameters"]}
say(f"execution {arn.rsplit('/', 1)[-1]}: {execution['PipelineExecutionStatus']}")
data_uri = params.get("InputDataS3Uri", "")
say(f"  -> data       {params.get('DataVersion')}: md5 {''.join(data_uri.rsplit('/', 2)[-2:])} ({data_uri})")
say(f"  -> code       git {params.get('GitSha', '?')}, images {params.get('TrainingImageUri', '?').rsplit(':', 1)[-1]}")
steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn, SortOrder="Ascending")["PipelineExecutionSteps"]
training_job = None
for step in steps:
    meta, name = step.get("Metadata", {}), step["StepName"]
    detail = ""
    if "ProcessingJob" in meta:
        detail = "processing job " + meta["ProcessingJob"]["Arn"].rsplit("/", 1)[-1]
    elif "TrainingJob" in meta:
        training_job = meta["TrainingJob"]["Arn"].rsplit("/", 1)[-1]
        detail = "training job " + training_job
    elif "RegisterModel" in meta:
        detail = "package " + meta["RegisterModel"]["Arn"].rsplit("/", 2)[-2] + "/" + meta["RegisterModel"]["Arn"].rsplit("/", 1)[-1]
    elif "Fail" in meta:
        detail = "fail: " + meta["Fail"].get("ErrorMessage", "")
    elif "Condition" in meta:
        detail = "condition: " + str(meta["Condition"].get("Outcome"))
    cached = " (cache hit)" if step.get("CacheHitResult") else ""
    say(f"  -> step       {name}: {step['StepStatus']}{cached} {detail}".rstrip())
if training_job and not config.load().get("MLOPS_MLFLOW_APP_ARN"):
    say("  -> MLflow run (no MLOPS_MLFLOW_APP_ARN in .env.mlops: run bash ../scripts/create-env.sh --write-env-only)")
elif training_job:
    mlflow_layer.connect(config.load().get("MLOPS_MLFLOW_APP_ARN"))
    run = mlflow_layer.latest_run("training_job", training_job)
    # The run's own number is on its own 20% test split of that data version; only the gate line compares models
    value = f"{int(run.data.metrics['net_benefit']):,}" if run else "?"
    say(f"  -> MLflow run {run.info.run_id if run else 'none'} (net_benefit {value} on its own 20% test split, "
        "not the gate holdout)")
bucket = config.load().get("MLOPS_JOBS_BUCKET")
try:
    key = f"pipeline/{arn.rsplit('/', 1)[-1]}/gate/gate_decision.json"
    decision = json.loads(boto3.client("s3", region_name=config.REGION).get_object(Bucket=bucket, Key=key)["Body"].read())
    say(f"  -> gate       {decision['decision']}: {decision['reason']}")
except ClientError:
    say("  -> gate       (gate_decision.json not found under pipeline/<execution>/gate/)")
