"""
Tests for mlops_utils/jobs.py, the one job helper.

boto3 never reaches AWS here: every SageMaker call goes to botocore's Stubber, which checks the exact request
and returns a made-up answer. Account IDs in the made-up ARNs are the placeholder <ACCOUNT_ID>.

What is proved:
  - every training job sends exactly the one fixed request (spot ml.m5.large, 3600 < 7200 s, 5 GB,
    stage production, channels fe + config, ProfilerConfig {DisableProfiler: True}, the course tags), once
  - an AWS error raises AwsCallFailed with SageMaker's code and text, the account ID masked, also in the
    printed traceback, and no second request is sent
  - BillableTimeInSeconds is read back and cost = Billable x 0.115 / 3600
  - processing jobs: ml.t3.medium x 1, MaxRuntimeInSeconds 3600, the course tags
"""

import traceback
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from botocore.stub import Stubber

from mlops_utils import config, jobs

NAME = "ccfd-train-test-1"
IMAGE = "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/ccfd-training:sha-abc1234"
ROLE = "arn:aws:iam::<ACCOUNT_ID>:role/ccfd-sagemaker-execution-role"
# CONFIG_URI is the fixed config key (TrainingConfigS3Uri). FE_URI and OUTPUT_URI are made-up prefixes: the names table
# names them with the pipeline
FE_URI = "s3://ccfd-mlops-jobs-a1b2c3/fe/"
CONFIG_URI = "s3://ccfd-mlops-jobs-a1b2c3/config/p5_02_best_config.yaml"
OUTPUT_URI = "s3://ccfd-mlops-jobs-a1b2c3/training/"
TAGS = [{"Key": "course", "Value": "mlops-on-aws"}, {"Key": "owner", "Value": "test-owner"},
        {"Key": "purpose", "Value": "a0-build"}, {"Key": "created", "Value": "2026-10-12"}]
JOB_ARN = f"arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:training-job/{NAME}"
ACCOUNT = "".join(str(digit % 10) for digit in range(1, 13))       # a made-up 12-digit number, built at run time

# Written out by hand: the request every course training job must send
BLOCK_B = {
    "TrainingJobName": NAME,
    "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File"},
    "RoleArn": ROLE,
    "HyperParameters": {"stage": "production"},
    "InputDataConfig": [
        {"ChannelName": "fe", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": FE_URI, "S3DataDistributionType": "FullyReplicated"}}},
        {"ChannelName": "config", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": CONFIG_URI, "S3DataDistributionType": "FullyReplicated"}}},
    ],
    "OutputDataConfig": {"S3OutputPath": OUTPUT_URI},
    "ResourceConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 5},
    "EnableManagedSpotTraining": True,
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600, "MaxWaitTimeInSeconds": 7200},
    "ProfilerConfig": {"DisableProfiler": True},
    "Tags": TAGS,
}


@pytest.fixture
def sm():
    """A SageMaker client with made-up keys (nothing is ever sent) and a counter of the API calls it makes."""
    client = boto3.client("sagemaker", region_name="us-east-1",
                          aws_access_key_id="testing", aws_secret_access_key="testing")
    client.calls = []
    client.meta.events.register("provide-client-params.sagemaker.*",
                                lambda params, model, **_: client.calls.append(model.name))
    with Stubber(client) as stubber:
        client.stubber = stubber
        yield client
        stubber.assert_no_pending_responses()


def described(**changes):
    """A describe_training_job answer with the fields SageMaker always sends, plus changes."""
    answer = {
        "TrainingJobName": NAME,
        "TrainingJobArn": JOB_ARN,
        "ModelArtifacts": {"S3ModelArtifacts": f"{OUTPUT_URI}{NAME}/output/model.tar.gz"},
        "TrainingJobStatus": "InProgress",
        "SecondaryStatus": "Starting",
        "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        "ResourceConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 5},
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600, "MaxWaitTimeInSeconds": 7200},
        "CreationTime": datetime(2026, 10, 12, 12, 0, tzinfo=UTC),
        "ProfilingStatus": "Disabled",
    }
    answer.update(changes)
    return {key: value for key, value in answer.items() if value is not None}


def start(sm, **kwargs):
    options = {"log": lambda message: None, **kwargs}
    return jobs.start_training_job(NAME, IMAGE, ROLE, FE_URI, CONFIG_URI, OUTPUT_URI, tags=TAGS, sm=sm, **options)


def printed(error: BaseException) -> str:
    """The full traceback as Python prints it, chained errors included."""
    return "".join(traceback.format_exception(error))


# ---------------------------------------------------------------------------
# 1. The request: always the same one
# ---------------------------------------------------------------------------
def test_request_is_exactly_block_b():
    request = jobs.training_job_request(NAME, IMAGE, ROLE, FE_URI, CONFIG_URI, OUTPUT_URI, tags=TAGS)
    assert request == BLOCK_B
    assert request["ProfilerConfig"] == {"DisableProfiler": True}      # rule 27: Profiler off in every job
    assert not set(request) & set(jobs.NEVER_IN_TRAINING)              # and no Debugger or TensorBoard field
    assert "Environment" not in request
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] < request["StoppingCondition"]["MaxWaitTimeInSeconds"]


def test_tracking_uri_adds_only_the_environment():
    """a job that logs to MLflow gets the App ARN as MLFLOW_TRACKING_URI; nothing else changes."""
    app_arn = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:mlflow-app/app-abc123"
    request = jobs.training_job_request(NAME, IMAGE, ROLE, FE_URI, CONFIG_URI, OUTPUT_URI, tags=TAGS,
                                        tracking_uri=app_arn)
    assert request.pop("Environment") == {"MLFLOW_TRACKING_URI": app_arn}
    assert request == BLOCK_B


def test_tracking_uri_must_be_the_app_arn():
    """A job cannot reach the laptop's MLflow server; the error never shows the value."""
    with pytest.raises(ValueError, match="not an MLflow App ARN"):
        jobs.training_job_request(NAME, IMAGE, ROLE, FE_URI, CONFIG_URI, OUTPUT_URI, tags=TAGS,
                                  tracking_uri="http://127.0.0.1:7080")


def test_default_tags_come_from_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                      # no .env.mlops here or above it
    for name, value in (("MLOPS_OWNER", "test-owner"), ("MLOPS_PURPOSE", "a0-build"), ("MLOPS_CREATED", "2026-10-12")):
        monkeypatch.setenv(name, value)
    request = jobs.training_job_request(NAME, IMAGE, ROLE, FE_URI, CONFIG_URI, OUTPUT_URI)
    assert request["Tags"] == TAGS
    monkeypatch.delenv("MLOPS_OWNER")
    with pytest.raises(ValueError, match="MLOPS_OWNER"):
        jobs.processing_job_request("p", IMAGE, ROLE, ["python3", "x.py"], [], [])


# ---------------------------------------------------------------------------
# 2. start_training_job(): one request
# ---------------------------------------------------------------------------
def test_start_sends_block_b_once_and_prints_the_start_line(sm):
    sm.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, BLOCK_B)
    logged = []
    response = start(sm, log=logged.append)
    assert response["TrainingJobArn"] == JOB_ARN
    assert sm.calls == ["CreateTrainingJob"]
    assert logged == [f"training job {NAME} started: ml.m5.large, managed spot"]


def test_a_create_error_is_aws_call_failed_masked_and_not_retried(sm):
    """AccessDenied names the caller's ARN; the error and the whole printed traceback are masked."""
    sm.stubber.add_client_error("create_training_job", service_error_code="AccessDeniedException",
                                service_message=f"User: arn:aws:iam::{ACCOUNT}:user/mlops-test-user is not authorized "
                                                "to perform: sagemaker:CreateTrainingJob",
                                http_status_code=400, expected_params=BLOCK_B)
    with pytest.raises(jobs.AwsCallFailed) as caught:
        start(sm)
    assert caught.value.code == "AccessDeniedException" and caught.value.operation == "CreateTrainingJob"
    assert str(caught.value).startswith("CreateTrainingJob failed (AccessDeniedException): User: ")
    assert "arn:aws:iam::<ACCOUNT_ID>:user/mlops-test-user" in str(caught.value)
    assert ACCOUNT not in printed(caught.value)
    assert sm.calls == ["CreateTrainingJob"]          # never a second request


def test_other_errors_raise_aws_call_failed_with_the_aws_text(sm):
    sm.stubber.add_client_error("create_training_job", service_error_code="ResourceLimitExceeded",
                                service_message=f"The spot training limit of account {ACCOUNT} is 0.",
                                expected_params=BLOCK_B)
    with pytest.raises(jobs.AwsCallFailed) as caught:
        start(sm)
    assert caught.value.code == "ResourceLimitExceeded"
    assert str(caught.value) == ("CreateTrainingJob failed (ResourceLimitExceeded): "
                                 "The spot training limit of account <ACCOUNT_ID> is 0.")
    assert ACCOUNT not in printed(caught.value)
    assert sm.calls == ["CreateTrainingJob"]


# ---------------------------------------------------------------------------
# 3. wait_for_training_job(): Billable seconds and the cost
# ---------------------------------------------------------------------------
def test_cost_is_billable_seconds_times_the_hourly_price():
    assert jobs.training_cost_usd(400) == pytest.approx(400 * 0.115 / 3600)
    assert jobs.training_cost_usd(3600) == pytest.approx(0.115)
    assert jobs.training_cost_usd(None) == 0


def test_wait_reads_billable_time_and_cost(sm):
    """A real test job: Training 1,038 s, Billable 400 s (spot 61% off), 0.0128 USD."""
    sm.stubber.add_response("describe_training_job", described(), {"TrainingJobName": NAME})
    sm.stubber.add_response("describe_training_job",
                            described(TrainingJobStatus="Completed", SecondaryStatus="Completed",
                                      TrainingTimeInSeconds=1038, BillableTimeInSeconds=400),
                            {"TrainingJobName": NAME})
    printed = []
    summary = jobs.wait_for_training_job(NAME, sm=sm, sleep=lambda seconds: None, log=printed.append)
    assert summary["status"] == "Completed" and summary["profiling_status"] == "Disabled"
    assert sm.calls == ["DescribeTrainingJob", "DescribeTrainingJob"]
    assert (summary["training_seconds"], summary["billable_seconds"], summary["spot_saving_pct"]) == (1038, 400, 61)
    assert summary["cost_usd"] == pytest.approx(400 * 0.115 / 3600)
    assert "cost = 400 x 0.115 / 3600 = 0.0128 USD" in printed[-1]


def test_wait_failed_job_raises_with_the_reason(sm):
    sm.stubber.add_response("describe_training_job",
                            described(TrainingJobStatus="Failed", SecondaryStatus="Failed",
                                      FailureReason="AlgorithmError: Channel fe must hold exactly 1 file"),
                            {"TrainingJobName": NAME})
    with pytest.raises(RuntimeError, match="Channel fe must hold exactly 1 file"):
        jobs.wait_for_training_job(NAME, sm=sm, sleep=lambda seconds: None, log=lambda message: None)


# ---------------------------------------------------------------------------
# 4. Processing jobs
# ---------------------------------------------------------------------------
PROCESSING_NAME = "ccfd-data-checks-test-1"
ENTRYPOINT = ["python3", "/opt/program/m08_02_data_checks.py"]
# m08_02's folders: input raw (the DVC object), outputs fe (channel fe of the training job) and report
INPUTS = [jobs.processing_input("raw", "s3://ccfd-mlops-dvc-a1b2c3/dvc/files/md5/ab/cdef",
                                "/opt/ml/processing/input/raw")]
OUTPUTS = [jobs.processing_output("fe", "s3://ccfd-mlops-jobs-a1b2c3/fe/", "/opt/ml/processing/output/fe"),
           jobs.processing_output("report", "s3://ccfd-mlops-jobs-a1b2c3/report/",
                                  "/opt/ml/processing/output/report")]
PROCESSING_REQUEST = {
    "ProcessingJobName": PROCESSING_NAME,
    "ProcessingResources": {"ClusterConfig": {"InstanceType": "ml.t3.medium", "InstanceCount": 1,
                                              "VolumeSizeInGB": 5}},
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    "AppSpecification": {"ImageUri": IMAGE, "ContainerEntrypoint": ENTRYPOINT},
    "RoleArn": ROLE,
    "ProcessingInputs": [{"InputName": "raw", "S3Input": {
        "S3Uri": "s3://ccfd-mlops-dvc-a1b2c3/dvc/files/md5/ab/cdef", "LocalPath": "/opt/ml/processing/input/raw",
        "S3DataType": "S3Prefix", "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated",
        "S3CompressionType": "None"}}],
    "ProcessingOutputConfig": {"Outputs": [
        {"OutputName": "fe", "S3Output": {"S3Uri": "s3://ccfd-mlops-jobs-a1b2c3/fe/",
                                          "LocalPath": "/opt/ml/processing/output/fe", "S3UploadMode": "EndOfJob"}},
        {"OutputName": "report", "S3Output": {"S3Uri": "s3://ccfd-mlops-jobs-a1b2c3/report/",
                                              "LocalPath": "/opt/ml/processing/output/report",
                                              "S3UploadMode": "EndOfJob"}}]},
    "Tags": TAGS,
}


def test_processing_request_is_t3_medium_3600_s_and_n01_tags():
    assert jobs.processing_job_request(PROCESSING_NAME, IMAGE, ROLE, ENTRYPOINT, INPUTS, OUTPUTS,
                                       tags=TAGS) == PROCESSING_REQUEST


def test_start_processing_job_sends_that_request(sm):
    arn = f"arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:processing-job/{PROCESSING_NAME}"
    request = {**PROCESSING_REQUEST, "Environment": {"STAGE_NOTE": "x"},
               "AppSpecification": {**PROCESSING_REQUEST["AppSpecification"], "ContainerArguments": ["--a", "1"]}}
    sm.stubber.add_response("create_processing_job", {"ProcessingJobArn": arn}, request)
    jobs.start_processing_job(PROCESSING_NAME, IMAGE, ROLE, ENTRYPOINT, INPUTS, OUTPUTS, arguments=["--a", "1"],
                              environment={"STAGE_NOTE": "x"}, tags=TAGS, sm=sm, log=lambda message: None)
    assert sm.calls == ["CreateProcessingJob"]


def test_wait_for_processing_job_gives_seconds_and_cost(sm):
    start_time = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
    base = {"ProcessingJobName": PROCESSING_NAME, "ProcessingResources": PROCESSING_REQUEST["ProcessingResources"],
            "AppSpecification": PROCESSING_REQUEST["AppSpecification"], "CreationTime": start_time,
            "ProcessingJobArn": f"arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:processing-job/{PROCESSING_NAME}"}
    sm.stubber.add_response("describe_processing_job", {**base, "ProcessingJobStatus": "InProgress"},
                            {"ProcessingJobName": PROCESSING_NAME})
    sm.stubber.add_response("describe_processing_job",
                            {**base, "ProcessingJobStatus": "Completed", "ProcessingStartTime": start_time,
                             "ProcessingEndTime": start_time + timedelta(seconds=150)},
                            {"ProcessingJobName": PROCESSING_NAME})
    summary = jobs.wait_for_processing_job(PROCESSING_NAME, sm=sm, sleep=lambda seconds: None,
                                           log=lambda message: None)
    assert (summary["status"], summary["seconds"]) == ("Completed", 150)
    assert summary["cost_usd"] == pytest.approx(150 * 0.05 / 3600)


def test_the_sample_request_matches_m08_02_folders():
    """The sample above is the contract of m08_02: it reads input/raw and writes output/fe and output/report."""
    import m08_02_data_checks as checks

    assert INPUTS[0]["S3Input"]["LocalPath"] == str(checks.PROCESSING_DIR / "input" / checks.INPUT_NAME)
    assert [output["S3Output"]["LocalPath"] for output in OUTPUTS] == [
        str(checks.PROCESSING_DIR / "output" / "fe"), str(checks.PROCESSING_DIR / "output" / "report")]


def test_masking_is_the_config_rule():
    assert config.mask(f"role in {ACCOUNT}") == "role in <ACCOUNT_ID>"
