"""
The one job helper: every course script that starts a SageMaker training or processing job
calls this file, so the job settings live in one place.

  training_job_request()     the exact CreateTrainingJob request of every course training job
  start_training_job()       sends it once
  wait_for_training_job()    waits; prints Training and Billable seconds and the cost
  training_cost_usd()        Billable seconds x 0.115 USD / 3600 (ml.m5.large, the on-demand rate as the upper bound)
  processing_job_request(), start_processing_job(), wait_for_processing_job()
                             ml.t3.medium x 1, MaxRuntimeInSeconds 3600, the course tags
  s3_channel(), processing_input(), processing_output()
                             the small dicts the requests are made of

The training job, in one line: managed spot ml.m5.large, MaxRuntime 3600 s < MaxWait 7200 s, 5 GB volume,
stage "production" (the base project's fixed YAML in channel config, p5_03 only), Profiler off, the course tags.

Rule 27: SageMaker Debugger and Profiler are closed to new customers, so every training job sends
ProfilerConfig {DisableProfiler: True} and none of the Debugger or TensorBoard fields (NEVER_IN_TRAINING).

No 12-digit account ID reaches the terminal from this file: every message it prints, and every AWS error it
raises (AwsCallFailed), goes through config.mask(), and the original botocore error, whose text can hold an ARN,
is left out of the printed traceback. The dicts it returns hold ARNs with the account ID: mask them before you
print them.

    from mlops_utils import jobs
    jobs.start_training_job(
        "ccfd-train-...", image_uri, role_arn,
        fe_s3_uri=fe_prefix,           # the S3 prefix that holds the data checks job's output fe (one file)
        config_s3_uri="s3://ccfd-mlops-jobs-<random6>/config/p5_02_best_config.yaml",   # TrainingConfigS3Uri
        output_s3_uri=output_prefix)   # an S3 prefix in the jobs bucket for model.tar.gz
    summary = jobs.wait_for_training_job("ccfd-train-...")
    print(summary["billable_seconds"], summary["cost_usd"])
The fe and output prefixes get their names with the pipeline; until then the caller picks them.
"""

import time
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from mlops_utils import config

# ---------------------------------------------------------------------------
# The fixed job settings and their prices
# ---------------------------------------------------------------------------
TRAINING_INSTANCE = "ml.m5.large"
TRAINING_PRICE_PER_HOUR = 0.115            # USD, ml.m5.large training, us-east-1 (on-demand: the upper bound for spot)
TRAINING_VOLUME_GB = 5
TRAINING_MAX_RUNTIME = 3600                # seconds of training
TRAINING_MAX_WAIT = 7200                   # seconds of training + waiting for spot capacity (must be > MaxRuntime)
TRAINING_STAGE = "production"              # every SageMaker job; stage gridsearch runs only on the laptop
PROFILER_OFF = {"DisableProfiler": True}   # in every training job

PROCESSING_INSTANCE = "ml.t3.medium"
PROCESSING_PRICE_PER_HOUR = 0.05           # USD, ml.t3.medium processing, us-east-1
PROCESSING_VOLUME_GB = 5
PROCESSING_MAX_RUNTIME = 3600

# Fields a course training request never holds (no checkpoints: a short spot job restarts instead)
NEVER_IN_TRAINING = ("CheckpointConfig", "DebugHookConfig", "DebugRuleConfigurations",
                     "ProfilerRuleConfigurations", "TensorBoardOutputConfig")

FINISHED = ("Completed", "Failed", "Stopped")


class AwsCallFailed(RuntimeError):
    """
    A SageMaker call failed. The message names the call, the error code and SageMaker's text, with the account ID
    masked. .code is the error code (for example ResourceLimitExceeded), .operation the API call.
    """

    def __init__(self, message: str, code: str | None = None, operation: str | None = None):
        super().__init__(message)
        self.code = code
        self.operation = operation


def _client(sm):
    return sm if sm is not None else boto3.client("sagemaker", region_name=config.REGION)


def _tags(tags):
    """The course tags as boto3 wants them; the default comes from mlops_utils/config.py (.env.mlops)."""
    return tags if tags is not None else config.load().aws_tags()


def _error_parts(error: Exception) -> tuple[str, str]:
    """The error code and SageMaker's text of a botocore error (not masked yet)."""
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        return details.get("Code") or "Unknown", details.get("Message") or str(error)
    return type(error).__name__, str(error)


def _call(operation: str, function, **request):
    """
    One SageMaker call. An AWS error comes back as AwsCallFailed with the code and the masked text; "from None"
    keeps the original error, whose text can hold the account ID, out of the printed traceback.
    """
    try:
        return function(**request)
    except (ClientError, BotoCoreError) as error:
        code, text = _error_parts(error)
        raise AwsCallFailed(config.mask(f"{operation} failed ({code}): {text}"), code, operation) from None


# ---------------------------------------------------------------------------
# 1. The small parts of a request
# ---------------------------------------------------------------------------
def s3_channel(name: str, s3_uri: str) -> dict:
    """One training input channel: every object under the S3 prefix, copied to /opt/ml/input/data/<name>/."""
    return {
        "ChannelName": name,
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": s3_uri,
            "S3DataDistributionType": "FullyReplicated",
        }},
    }


def processing_input(name: str, s3_uri: str, local_path: str) -> dict:
    """One processing input: the S3 prefix is copied to local_path (under /opt/ml/processing/input/)."""
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": s3_uri,
            "LocalPath": local_path,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
            "S3CompressionType": "None",
        },
    }


def processing_output(name: str, s3_uri: str, local_path: str) -> dict:
    """One processing output: what the job writes to local_path is uploaded to the S3 prefix when it ends."""
    return {"OutputName": name, "S3Output": {"S3Uri": s3_uri, "LocalPath": local_path, "S3UploadMode": "EndOfJob"}}


def training_cost_usd(billable_seconds: float, price_per_hour: float = TRAINING_PRICE_PER_HOUR) -> float:
    """Cost of a training job: BillableTimeInSeconds x 0.115 / 3600 (managed spot bills fewer seconds)."""
    return (billable_seconds or 0) * price_per_hour / 3600


# ---------------------------------------------------------------------------
# 2. Training jobs
# ---------------------------------------------------------------------------
def training_job_request(name: str, training_image_uri: str, execution_role_arn: str, fe_s3_uri: str,
                         config_s3_uri: str, output_s3_uri: str, tags: list | None = None,
                         tracking_uri: str | None = None) -> dict:
    """
    The CreateTrainingJob request of every course training job, always the same:
      channel fe      the FE1 data (one file, any name), channel config  the base project's fixed YAML (one file)
      stage           production (p5_03 only), managed spot ml.m5.large, Profiler off, the course tags
    tracking_uri: the MLflow App ARN, only when the job logs to MLflow itself (proved in a real run).
                  It is the one addition to that request: Environment {"MLFLOW_TRACKING_URI": <the App ARN>}.
    """
    request = {
        "TrainingJobName": name,
        "AlgorithmSpecification": {"TrainingImage": training_image_uri, "TrainingInputMode": "File"},
        "RoleArn": execution_role_arn,
        "HyperParameters": {"stage": TRAINING_STAGE},
        "InputDataConfig": [s3_channel("fe", fe_s3_uri), s3_channel("config", config_s3_uri)],
        "OutputDataConfig": {"S3OutputPath": output_s3_uri},
        "ResourceConfig": {"InstanceType": TRAINING_INSTANCE, "InstanceCount": 1,
                           "VolumeSizeInGB": TRAINING_VOLUME_GB},
        "EnableManagedSpotTraining": True,
        "StoppingCondition": {"MaxRuntimeInSeconds": TRAINING_MAX_RUNTIME,
                              "MaxWaitTimeInSeconds": TRAINING_MAX_WAIT},
        "ProfilerConfig": dict(PROFILER_OFF),
        "Tags": _tags(tags),
    }
    if tracking_uri:
        # A job cannot reach the laptop's MLflow server: only the App ARN makes sense here
        config.check_app_arn(tracking_uri, "tracking_uri")
        request["Environment"] = {config.TRACKING_URI_VARIABLE: tracking_uri}
    return request


def start_training_job(name: str, training_image_uri: str, execution_role_arn: str, fe_s3_uri: str,
                       config_s3_uri: str, output_s3_uri: str, *, tags: list | None = None,
                       tracking_uri: str | None = None, sm=None, log=print) -> dict:
    """
    Start one training job with training_job_request(), sent once. An AWS error raises AwsCallFailed with
    SageMaker's code and masked text. Returns the create_training_job answer (its ARN holds the account ID).
    """
    request = training_job_request(name, training_image_uri, execution_role_arn, fe_s3_uri, config_s3_uri,
                                   output_s3_uri, tags=tags, tracking_uri=tracking_uri)
    response = _call("CreateTrainingJob", _client(sm).create_training_job, **request)
    log(f"training job {name} started: {TRAINING_INSTANCE}, managed spot")
    return response


def training_summary(description: dict) -> dict:
    """The numbers the course records for a training job (seconds, cost, where the model went)."""
    billable = description.get("BillableTimeInSeconds") or 0
    training = description.get("TrainingTimeInSeconds") or 0
    return {
        "name": description["TrainingJobName"],
        "status": description["TrainingJobStatus"],
        "secondary_status": description.get("SecondaryStatus"),
        "failure_reason": description.get("FailureReason"),
        "profiling_status": description.get("ProfilingStatus"),
        "training_seconds": training,
        "billable_seconds": billable,
        "spot_saving_pct": round(100 * (1 - billable / training)) if training else None,
        "cost_usd": training_cost_usd(billable),
        "model_artifact": description.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
    }


def wait_for_training_job(name: str, *, sm=None, poll_seconds: int = 30, sleep=time.sleep, log=print,
                          check: bool = True) -> dict:
    """
    Wait until the job is Completed, Failed or Stopped (one describe call every poll_seconds), printing each
    status change. Then read BillableTimeInSeconds and return training_summary().
    check=True raises RuntimeError when the job did not complete.
    """
    sm = _client(sm)
    last = None
    while True:
        description = _call("DescribeTrainingJob", sm.describe_training_job, TrainingJobName=name)
        now = (description["TrainingJobStatus"], description.get("SecondaryStatus"))
        if now != last:
            log(f"  {datetime.now():%H:%M:%S} {now[0]} (step: {now[1]})")
            last = now
        if now[0] in FINISHED:
            break
        sleep(poll_seconds)
    summary = training_summary(description)
    log(config.mask(
        f"training job {name}: {summary['status']}; Training {summary['training_seconds']} s, "
        f"Billable {summary['billable_seconds']} s; cost = {summary['billable_seconds']} x "
        f"{TRAINING_PRICE_PER_HOUR} / 3600 = {summary['cost_usd']:.4f} USD"))
    if check and summary["status"] != "Completed":
        raise RuntimeError(config.mask(f"Training job {name} ended {summary['status']}: "
                                       f"{summary['failure_reason']}"))
    return summary


# ---------------------------------------------------------------------------
# 3. Processing jobs (the data checks of MLOps 08 and the gate of MLOps 13 run in the same training image)
# ---------------------------------------------------------------------------
def processing_job_request(name: str, image_uri: str, execution_role_arn: str, entrypoint: list[str],
                           inputs: list[dict], outputs: list[dict], *, arguments: list[str] | None = None,
                           environment: dict | None = None, tags: list | None = None) -> dict:
    """
    The CreateProcessingJob request: ml.t3.medium x 1 (5 GB), MaxRuntimeInSeconds 3600, the course tags.
    entrypoint  for example ["python3", "/opt/program/m08_02_data_checks.py"] (the image's own files)
    inputs      processing_input() dicts; outputs: processing_output() dicts
    """
    app = {"ImageUri": image_uri, "ContainerEntrypoint": list(entrypoint)}
    if arguments:
        app["ContainerArguments"] = list(arguments)
    request = {
        "ProcessingJobName": name,
        "ProcessingResources": {"ClusterConfig": {"InstanceType": PROCESSING_INSTANCE, "InstanceCount": 1,
                                                  "VolumeSizeInGB": PROCESSING_VOLUME_GB}},
        "StoppingCondition": {"MaxRuntimeInSeconds": PROCESSING_MAX_RUNTIME},
        "AppSpecification": app,
        "RoleArn": execution_role_arn,
        "ProcessingInputs": list(inputs),
        "ProcessingOutputConfig": {"Outputs": list(outputs)},
        "Tags": _tags(tags),
    }
    if environment:
        request["Environment"] = dict(environment)
    return request


def start_processing_job(name: str, image_uri: str, execution_role_arn: str, entrypoint: list[str],
                         inputs: list[dict], outputs: list[dict], *, arguments: list[str] | None = None,
                         environment: dict | None = None, tags: list | None = None, sm=None, log=print) -> dict:
    """Start one processing job with processing_job_request(). Returns the create_processing_job answer."""
    request = processing_job_request(name, image_uri, execution_role_arn, entrypoint, inputs, outputs,
                                     arguments=arguments, environment=environment, tags=tags)
    response = _call("CreateProcessingJob", _client(sm).create_processing_job, **request)
    log(f"processing job {name} started: {PROCESSING_INSTANCE} x 1, {' '.join(entrypoint)}")
    return response


def wait_for_processing_job(name: str, *, sm=None, poll_seconds: int = 30, sleep=time.sleep, log=print,
                            check: bool = True) -> dict:
    """
    Wait until the processing job is Completed, Failed or Stopped, printing each status change.
    Returns its status, the run time in seconds and the cost estimate (seconds x 0.05 / 3600).
    check=True raises RuntimeError when the job did not complete.
    """
    sm = _client(sm)
    last = None
    while True:
        description = _call("DescribeProcessingJob", sm.describe_processing_job, ProcessingJobName=name)
        status = description["ProcessingJobStatus"]
        if status != last:
            log(f"  {datetime.now():%H:%M:%S} {status}")
            last = status
        if status in FINISHED:
            break
        sleep(poll_seconds)
    start, end = description.get("ProcessingStartTime"), description.get("ProcessingEndTime")
    seconds = int((end - start).total_seconds()) if start and end else 0
    summary = {
        "name": name,
        "status": status,
        "failure_reason": description.get("FailureReason"),
        "exit_message": description.get("ExitMessage"),
        "seconds": seconds,
        "cost_usd": seconds * PROCESSING_PRICE_PER_HOUR / 3600,
    }
    # One fact per line, each with its label
    log(config.mask(f"processing job {name}: {status}"))
    log(f"  run time: {seconds} s")
    log(f"  cost:     about {seconds} x {PROCESSING_PRICE_PER_HOUR} / 3600 = {summary['cost_usd']:.4f} USD")
    if check and status != "Completed":
        raise RuntimeError(config.mask(f"Processing job {name} ended {status}: {summary['failure_reason']}"))
    return summary
