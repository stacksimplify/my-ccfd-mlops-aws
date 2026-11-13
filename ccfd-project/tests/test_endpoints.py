"""
Tests for mlops_utils/endpoints.py, the endpoint helper.

boto3 never reaches AWS here: every call goes to botocore's Stubber, which checks the exact request and returns a
made-up answer. Account IDs in the made-up ARNs are the placeholder <ACCOUNT_ID>, or a 12-digit number built at
run time to prove the masking.

What is proved:
  - invoke() calls once when the first call works, retries exactly once after a failure, and raises after the
    second failure (2 calls, never 3); botocore's own retries are off; errors are masked;
    a dict or a list body goes as JSON, any other type is refused before a call
  - the 2 rollout alarms are exactly the ones of MLOps 12 (names, metrics, statistics, thresholds, missing
    data not breaching) and carry the course tags; delete sends only the names that exist
  - only an Approved package becomes a package model; a plain model uses Image + ModelDataUrl, never
    ModelPackageName; newest Approved and the version list read the registry the right way
    (newest Approved follows an empty page that has a NextToken)
  - the command line prints its ERROR line on stderr (a script that captures stdout still shows it)
  - delete() removes autoscaling, then the endpoint, then its configs and models; with the endpoint gone already,
    it still deletes the configs and models left behind
  - every create call in endpoints.py carries Tags
"""

import ast
import io
import traceback
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from mlops_utils import endpoints
from mlops_utils.jobs import AwsCallFailed

TAGS = [{"Key": "course", "Value": "mlops-on-aws"}, {"Key": "owner", "Value": "test-owner"},
        {"Key": "purpose", "Value": "a0-build"}, {"Key": "created", "Value": "2026-10-12"}]
ROLE = "arn:aws:iam::<ACCOUNT_ID>:role/ccfd-sagemaker-execution-role"
ACCOUNT = "".join(str((digit * 3) % 10) for digit in range(1, 13))   # a made-up 12-digit number, built at run time
GROUP_ARN = "arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>:model-package/ccfd-fraud-detector"
IMAGE = "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/ccfd-serving:sha-abc1234"
MODEL_DATA = "s3://ccfd-mlops-jobs-a1b2c3/seed/data-v1/training/job/output/model.tar.gz"
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
NO_LOG = {"log": lambda message: None}
SM_ARN = "arn:aws:sagemaker:us-east-1:<ACCOUNT_ID>"            # the start of every made-up SageMaker ARN


def client(service):
    """A client with made-up keys (nothing is ever sent), a Stubber and a list of the API calls it makes."""
    made = boto3.client(service, region_name="us-east-1", aws_access_key_id="testing",
                        aws_secret_access_key="testing")
    made.calls = []
    made.meta.events.register(f"provide-client-params.{made.meta.service_model.endpoint_prefix}.*",
                              lambda params, model, **_: made.calls.append(model.name))
    return made


@pytest.fixture
def sm():
    made = client("sagemaker")
    with Stubber(made) as stubber:
        made.stubber = stubber
        yield made
        stubber.assert_no_pending_responses()


@pytest.fixture
def cw():
    made = client("cloudwatch")
    with Stubber(made) as stubber:
        made.stubber = stubber
        yield made
        stubber.assert_no_pending_responses()


@pytest.fixture
def aas():
    made = client("application-autoscaling")
    with Stubber(made) as stubber:
        made.stubber = stubber
        yield made
        stubber.assert_no_pending_responses()


@pytest.fixture
def runtime():
    made = client("sagemaker-runtime")
    with Stubber(made) as stubber:
        made.stubber = stubber
        yield made
        stubber.assert_no_pending_responses()


def answer_body(text: str) -> dict:
    data = text.encode("utf-8")
    return {"Body": StreamingBody(io.BytesIO(data), len(data)), "ContentType": "application/json",
            "InvokedProductionVariant": "AllTraffic"}


def package(version: int, status: str = "Approved") -> dict:
    return {"ModelPackageGroupName": "ccfd-fraud-detector", "ModelPackageVersion": version,
            "ModelPackageArn": f"{GROUP_ARN}/{version}", "CreationTime": NOW, "ModelPackageStatus": "Completed",
            "ModelApprovalStatus": status}


def described_package(version: int, status: str) -> dict:
    return {"ModelPackageName": "ccfd-fraud-detector", "ModelPackageGroupName": "ccfd-fraud-detector",
            "ModelPackageVersion": version, "ModelPackageArn": f"{GROUP_ARN}/{version}", "CreationTime": NOW,
            "ModelPackageStatus": "Completed", "ModelApprovalStatus": status,
            "ModelPackageStatusDetails": {"ValidationStatuses": [], "ImageScanStatuses": []},
            "InferenceSpecification": {"Containers": [{"Image": IMAGE, "ModelDataUrl": MODEL_DATA}],
                                       "SupportedContentTypes": ["application/json"],
                                       "SupportedResponseMIMETypes": ["application/json"]}}


def described_endpoint(name: str, config_name: str, status: str = "InService") -> dict:
    return {"EndpointName": name, "EndpointArn": f"{SM_ARN}:endpoint/{name}",
            "EndpointConfigName": config_name, "EndpointStatus": status, "CreationTime": NOW,
            "LastModifiedTime": NOW}


def not_found(stubbed, name):
    stubbed.stubber.add_client_error("describe_endpoint", service_error_code="ValidationException",
                                     service_message=f"Could not find endpoint \"{name}\".",
                                     expected_params={"EndpointName": name})


def printed(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))


# ---------------------------------------------------------------------------
# 1. invoke(): exactly one retry
# ---------------------------------------------------------------------------
REQUEST = {"EndpointName": "ccfd-serverless", "Body": '{"x": 1}', "ContentType": "application/json",
           "Accept": "application/json"}


def test_invoke_first_call_works_one_call(runtime):
    runtime.stubber.add_response("invoke_endpoint", answer_body('{"prediction": "FRAUD"}'), REQUEST)
    result = endpoints.invoke("ccfd-serverless", '{"x": 1}', runtime=runtime, sleep=lambda s: None, **NO_LOG)
    assert result == {"body": '{"prediction": "FRAUD"}', "variant": "AllTraffic", "attempts": 1}
    assert runtime.calls == ["InvokeEndpoint"]


def test_invoke_retries_exactly_once_after_a_failure(runtime):
    messages, waits = [], []
    runtime.stubber.add_client_error("invoke_endpoint", service_error_code="ModelError", http_status_code=424,
                                     service_message=f"cold start arn:aws:sagemaker:us-east-1:{ACCOUNT}:endpoint/x",
                                     expected_params=REQUEST)
    runtime.stubber.add_response("invoke_endpoint", answer_body("[]"), REQUEST)
    result = endpoints.invoke("ccfd-serverless", '{"x": 1}', runtime=runtime, sleep=waits.append,
                              log=messages.append)
    assert result["attempts"] == 2 and result["body"] == "[]"
    assert runtime.calls == ["InvokeEndpoint", "InvokeEndpoint"]
    assert waits == [endpoints.RETRY_WAIT_SECONDS]
    assert "ModelError" in messages[0] and ACCOUNT not in messages[0] and "<ACCOUNT_ID>" in messages[0]


def test_invoke_raises_after_the_second_failure_never_a_third_call(runtime):
    for _ in range(2):
        runtime.stubber.add_client_error("invoke_endpoint", service_error_code="ServiceUnavailable",
                                         http_status_code=503,
                                         service_message=f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:endpoint/x busy",
                                         expected_params=REQUEST)
    with pytest.raises(AwsCallFailed) as raised:
        endpoints.invoke("ccfd-serverless", '{"x": 1}', runtime=runtime, sleep=lambda s: None, **NO_LOG)
    assert runtime.calls == ["InvokeEndpoint", "InvokeEndpoint"]
    assert raised.value.code == "ServiceUnavailable" and "failed twice" in str(raised.value)
    assert ACCOUNT not in printed(raised.value)


def test_invoke_passes_target_variant_and_inference_id(runtime):
    expected = {**REQUEST, "EndpointName": "ccfd-prod", "TargetVariant": "challenger",
                "InferenceId": "v1:month_1:000001"}
    runtime.stubber.add_response("invoke_endpoint", answer_body("[]"), expected)
    endpoints.invoke("ccfd-prod", '{"x": 1}', target_variant="challenger", inference_id="v1:month_1:000001",
                     runtime=runtime, **NO_LOG)


def test_invoke_sends_a_dict_or_list_as_json(runtime):
    """str() of a dict is a Python repr ({'x': 1}), which the model refuses; a dict or a list goes as JSON."""
    runtime.stubber.add_response("invoke_endpoint", answer_body("[]"), REQUEST)
    runtime.stubber.add_response("invoke_endpoint", answer_body("[]"), {**REQUEST, "Body": '[{"x": 1}]'})
    endpoints.invoke("ccfd-serverless", {"x": 1}, runtime=runtime, **NO_LOG)
    endpoints.invoke("ccfd-serverless", [{"x": 1}], runtime=runtime, **NO_LOG)
    assert runtime.calls == ["InvokeEndpoint", "InvokeEndpoint"]


@pytest.mark.parametrize("body", [42, 1.5, None, ("x",)])
def test_invoke_refuses_any_other_body_type_without_a_call(runtime, body):
    with pytest.raises(TypeError, match="body must be bytes, str, a dict or a list"):
        endpoints.invoke("ccfd-serverless", body, runtime=runtime, **NO_LOG)
    assert runtime.calls == []


def test_runtime_client_has_no_botocore_retries():
    """Else 'one retry' would hide up to 4 botocore retries behind each call."""
    made = endpoints.runtime_client()
    assert made.meta.config.retries["total_max_attempts"] == 1


# ---------------------------------------------------------------------------
# 2. The 2 rollout alarms (MLOps 12 Step-01 and Notes)
# ---------------------------------------------------------------------------
ON_VARIANT = [{"Name": "EndpointName", "Value": "ccfd-prod"}, {"Name": "VariantName", "Value": "AllTraffic"}]
ALARM_NAMES = ["ccfd-prod-rollout-5xx", "ccfd-prod-rollout-netbenefit"]


def test_rollout_alarm_requests_are_the_mlops_12_alarms():
    requests = {r["AlarmName"]: r for r in endpoints.rollout_alarm_requests(tags=TAGS)}
    assert list(requests) == ALARM_NAMES == list(endpoints.ROLLOUT_ALARMS)
    five = requests["ccfd-prod-rollout-5xx"]
    assert (five["Namespace"], five["MetricName"], five["Dimensions"], five["Statistic"], five["Period"],
            five["EvaluationPeriods"], five["Threshold"], five["ComparisonOperator"]) == (
        "AWS/SageMaker", "Invocation5XXErrors", ON_VARIANT, "Sum", 60, 1, 0, "GreaterThanThreshold")
    money = requests["ccfd-prod-rollout-netbenefit"]
    assert (money["Namespace"], money["MetricName"], money["Dimensions"], money["Statistic"], money["Period"],
            money["Threshold"], money["ComparisonOperator"]) == (
        "MLOps/CCFD", "NetBenefit", [{"Name": "EndpointName", "Value": "ccfd-prod"},
                                  {"Name": "Source", "Value": "TestSetReplay"}],
        "Minimum", 60, 33000, "LessThanThreshold")
    assert 19925 < money["Threshold"] < 36975                       # on the probe set: the drill model rolls back, the champion passes
    for request in requests.values():
        assert request["TreatMissingData"] == "notBreaching" and request["Tags"] == TAGS
        assert "AlarmActions" not in request


def test_create_rollout_alarms_sends_two_tagged_puts(cw):
    for request in endpoints.rollout_alarm_requests(tags=TAGS):
        cw.stubber.add_response("put_metric_alarm", {}, request)
    names = endpoints.create_rollout_alarms(tags=TAGS, cw=cw, **NO_LOG)
    assert names == ALARM_NAMES
    assert cw.calls == ["PutMetricAlarm"] * 2


def alarm(name):
    return {"AlarmName": name, "StateValue": "OK"}


def test_delete_rollout_alarms_sends_only_the_names_that_exist(cw):
    cw.stubber.add_response("describe_alarms", {"MetricAlarms": [alarm("ccfd-prod-rollout-netbenefit")]},
                            {"AlarmNames": ALARM_NAMES})
    cw.stubber.add_response("delete_alarms", {}, {"AlarmNames": ["ccfd-prod-rollout-netbenefit"]})
    assert endpoints.delete_rollout_alarms(cw=cw, **NO_LOG) == ["ccfd-prod-rollout-netbenefit"]


def test_delete_rollout_alarms_none_calls_no_delete(cw):
    messages = []
    cw.stubber.add_response("describe_alarms", {"MetricAlarms": []}, {"AlarmNames": ALARM_NAMES})
    assert endpoints.delete_rollout_alarms(cw=cw, log=messages.append) == []
    assert "DeleteAlarms" not in cw.calls
    assert messages == ["rollout alarms: deleted none"]


def test_canary_config_uses_the_two_alarms():
    deployment = endpoints.canary_deployment_config()
    routing = deployment["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]
    assert routing == {"Type": "CANARY", "CanarySize": {"Type": "INSTANCE_COUNT", "Value": 1},
                       "WaitIntervalInSeconds": 300}
    assert deployment["BlueGreenUpdatePolicy"]["TerminationWaitInSeconds"] == 120
    assert [a["AlarmName"] for a in deployment["AutoRollbackConfiguration"]["Alarms"]] == ALARM_NAMES


# ---------------------------------------------------------------------------
# 3. Packages and models
# ---------------------------------------------------------------------------
def test_newest_approved_asks_for_one_approved_newest_first(sm):
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [package(2)]},
                            {"ModelPackageGroupName": "ccfd-fraud-detector", "ModelApprovalStatus": "Approved",
                             "SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 1})
    assert endpoints.newest_approved(sm=sm)["ModelPackageVersion"] == 2


def test_newest_approved_follows_an_empty_page_with_a_next_token(sm):
    """A filtered page may be empty with a NextToken (the newest version is Rejected): it is not 'no champion'."""
    base = {"ModelPackageGroupName": "ccfd-fraud-detector", "ModelApprovalStatus": "Approved",
            "SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 1}
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [], "NextToken": "t1"}, base)
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [], "NextToken": "t2"},
                            {**base, "NextToken": "t1"})
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [package(2)]},
                            {**base, "NextToken": "t2"})
    assert endpoints.newest_approved(sm=sm)["ModelPackageVersion"] == 2
    assert sm.calls == ["ListModelPackages"] * 3


def test_newest_approved_is_none_only_after_the_last_page(sm):
    base = {"ModelPackageGroupName": "ccfd-fraud-detector", "ModelApprovalStatus": "Approved",
            "SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 1}
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [], "NextToken": "t1"}, base)
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": []}, {**base, "NextToken": "t1"})
    assert endpoints.newest_approved(sm=sm) is None


def test_approved_versions_oldest_first_over_pages(sm):
    base = {"ModelPackageGroupName": "ccfd-fraud-detector", "SortBy": "CreationTime", "SortOrder": "Ascending",
            "ModelApprovalStatus": "Approved"}
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [package(1)], "NextToken": "t"}, base)
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [package(2)]},
                            {**base, "NextToken": "t"})
    assert endpoints.approved_versions(sm=sm) == [1, 2]


def test_a_package_model_needs_an_approved_package(sm):
    sm.stubber.add_response("describe_model_package", described_package(3, "PendingManualApproval"),
                            {"ModelPackageName": f"{GROUP_ARN}/3"})
    with pytest.raises(endpoints.PackageNotApproved, match="PendingManualApproval"):
        endpoints.create_model_from_package("ccfd-rt-model-v3", f"{GROUP_ARN}/3", role_arn=ROLE, tags=TAGS, sm=sm,
                                            **NO_LOG)
    assert "CreateModel" not in sm.calls


def test_an_approved_package_model_points_at_the_package(sm):
    sm.stubber.add_response("describe_model_package", described_package(1, "Approved"),
                            {"ModelPackageName": f"{GROUP_ARN}/1"})
    sm.stubber.add_response("create_model", {"ModelArn": f"{SM_ARN}:model/m"},
                            {"ModelName": "ccfd-rt-model-v1", "ExecutionRoleArn": ROLE,
                             "PrimaryContainer": {"ModelPackageName": f"{GROUP_ARN}/1"}, "Tags": TAGS})
    endpoints.create_model_from_package("ccfd-rt-model-v1", f"{GROUP_ARN}/1", role_arn=ROLE, tags=TAGS, sm=sm,
                                        **NO_LOG)


@pytest.mark.parametrize("status", ["PendingManualApproval", "Rejected", "Approved"])
def test_a_plain_model_uses_image_and_model_data_never_the_package(sm, status):
    sm.stubber.add_response("describe_model_package", described_package(3, status),
                            {"ModelPackageName": f"{GROUP_ARN}/3"})
    sm.stubber.add_response("create_model", {"ModelArn": f"{SM_ARN}:model/m"},
                            {"ModelName": "ccfd-staging-model-v3", "ExecutionRoleArn": ROLE,
                             "PrimaryContainer": {"Image": IMAGE, "ModelDataUrl": MODEL_DATA}, "Tags": TAGS})
    endpoints.create_plain_model("ccfd-staging-model-v3", f"{GROUP_ARN}/3", role_arn=ROLE, tags=TAGS, sm=sm,
                                 **NO_LOG)


def test_model_and_config_names_follow_v6():
    assert endpoints.model_names("ccfd-rt", 1) == ("ccfd-rt-model-v1", "ccfd-rt-config-v1")
    assert endpoints.model_names("ccfd-prod", "2") == ("ccfd-prod-model-v2", "ccfd-prod-config-v2")


def test_endpoint_config_with_the_step16_data_capture(sm):
    capture = endpoints.data_capture_config("s3://ccfd-mlops-jobs-a1b2c3/data-capture/")
    assert capture == {"EnableCapture": True, "InitialSamplingPercentage": 100,
                       "DestinationS3Uri": "s3://ccfd-mlops-jobs-a1b2c3/data-capture/",
                       "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                       "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json",
                                                                         "application/jsonlines"]}}
    sm.stubber.add_response("create_endpoint_config", {"EndpointConfigArn": f"{SM_ARN}:endpoint-config/c"},
                            {"EndpointConfigName": "ccfd-prod-config-v1",
                             "ProductionVariants": [{"VariantName": "AllTraffic", "ModelName": "ccfd-prod-model-v1",
                                                     "InstanceType": "ml.m5.large", "InitialInstanceCount": 2}],
                             "Tags": TAGS, "DataCaptureConfig": capture})
    endpoints.create_endpoint_config("ccfd-prod-config-v1", "ccfd-prod-model-v1", instance_count=2,
                                     data_capture=capture, tags=TAGS, sm=sm, **NO_LOG)


def test_data_capture_default_is_the_jobs_bucket(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                      # no .env.mlops here or above it
    monkeypatch.setenv("MLOPS_SUFFIX", "a1b2c3")
    monkeypatch.delenv("MLOPS_JOBS_BUCKET", raising=False)
    assert endpoints.data_capture_config()["DestinationS3Uri"] == "s3://ccfd-mlops-jobs-a1b2c3/data-capture/"


# ---------------------------------------------------------------------------
# 4. Deploy one version, then delete (MLOps 10 order)
# ---------------------------------------------------------------------------
def test_deploy_package_makes_model_config_and_endpoint(sm):
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [package(1), package(2)]},
                            {"ModelPackageGroupName": "ccfd-fraud-detector", "SortBy": "CreationTime",
                             "SortOrder": "Ascending"})
    sm.stubber.add_client_error("describe_model", service_error_code="ValidationException",
                                service_message="Could not find model \"ccfd-prod-model-v1\".",
                                expected_params={"ModelName": "ccfd-prod-model-v1"})
    sm.stubber.add_response("describe_model_package", described_package(1, "Approved"),
                            {"ModelPackageName": f"{GROUP_ARN}/1"})
    sm.stubber.add_response("create_model", {"ModelArn": f"{SM_ARN}:model/m"},
                            {"ModelName": "ccfd-prod-model-v1", "ExecutionRoleArn": ROLE,
                             "PrimaryContainer": {"ModelPackageName": f"{GROUP_ARN}/1"}, "Tags": TAGS})
    sm.stubber.add_client_error("describe_endpoint_config", service_error_code="ValidationException",
                                service_message="Could not find endpoint configuration \"ccfd-prod-config-v1\".",
                                expected_params={"EndpointConfigName": "ccfd-prod-config-v1"})
    sm.stubber.add_response("create_endpoint_config", {"EndpointConfigArn": f"{SM_ARN}:endpoint-config/c"},
                            {"EndpointConfigName": "ccfd-prod-config-v1",
                             "ProductionVariants": [{"VariantName": "AllTraffic", "ModelName": "ccfd-prod-model-v1",
                                                     "InstanceType": "ml.m5.large", "InitialInstanceCount": 2}],
                             "Tags": TAGS})
    not_found(sm, "ccfd-prod")
    sm.stubber.add_response("create_endpoint", {"EndpointArn": f"{SM_ARN}:endpoint/ccfd-prod"},
                            {"EndpointName": "ccfd-prod", "EndpointConfigName": "ccfd-prod-config-v1", "Tags": TAGS})
    sm.stubber.add_response("describe_endpoint", described_endpoint("ccfd-prod", "ccfd-prod-config-v1", "Creating"),
                            {"EndpointName": "ccfd-prod"})
    sm.stubber.add_response("describe_endpoint", described_endpoint("ccfd-prod", "ccfd-prod-config-v1"),
                            {"EndpointName": "ccfd-prod"})
    result = endpoints.deploy_package("ccfd-prod", 1, instance_count=2, role_arn=ROLE, tags=TAGS, sm=sm,
                                      sleep=lambda s: None, **NO_LOG)
    assert result["EndpointStatus"] == "InService"


def test_deploy_package_reuses_an_existing_model_and_config_and_updates(sm):
    """A kept model and config are used again (exact names, never a NameContains match); the endpoint is updated."""
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [package(1), package(2)]},
                            {"ModelPackageGroupName": "ccfd-fraud-detector", "SortBy": "CreationTime",
                             "SortOrder": "Ascending"})
    sm.stubber.add_response("describe_model", {"ModelName": "ccfd-rt-model-v1", "ModelArn": f"{SM_ARN}:model/v1",
                                               "CreationTime": NOW}, {"ModelName": "ccfd-rt-model-v1"})
    sm.stubber.add_response("describe_endpoint_config",
                            {"EndpointConfigName": "ccfd-rt-config-v1", "EndpointConfigArn": f"{SM_ARN}:ec/v1",
                             "CreationTime": NOW,
                             "ProductionVariants": [{"VariantName": "AllTraffic", "InitialInstanceCount": 1}]},
                            {"EndpointConfigName": "ccfd-rt-config-v1"})
    sm.stubber.add_response("describe_endpoint", described_endpoint("ccfd-rt", "ccfd-rt-config-v2"),
                            {"EndpointName": "ccfd-rt"})
    sm.stubber.add_response("update_endpoint", {"EndpointArn": f"{SM_ARN}:endpoint/ccfd-rt"},
                            {"EndpointName": "ccfd-rt", "EndpointConfigName": "ccfd-rt-config-v1"})
    sm.stubber.add_response("describe_endpoint", described_endpoint("ccfd-rt", "ccfd-rt-config-v1"),
                            {"EndpointName": "ccfd-rt"})
    endpoints.deploy_package("ccfd-rt", 1, role_arn=ROLE, tags=TAGS, sm=sm, sleep=lambda s: None, **NO_LOG)
    assert "CreateModel" not in sm.calls and "CreateEndpointConfig" not in sm.calls
    assert sm.calls.count("UpdateEndpoint") == 1


def test_delete_order_autoscaling_endpoint_configs_models(sm, aas):
    order = []
    sm.meta.events.register("provide-client-params.sagemaker.*", lambda model, **_: order.append(model.name))
    aas.meta.events.register("provide-client-params.application-autoscaling.*",
                             lambda model, **_: order.append(model.name))
    sm.stubber.add_response("describe_endpoint", described_endpoint("ccfd-rt", "ccfd-rt-config-v2"),
                            {"EndpointName": "ccfd-rt"})
    sm.stubber.add_response("describe_endpoint_config",
                            {"EndpointConfigName": "ccfd-rt-config-v2", "EndpointConfigArn": f"{SM_ARN}:endpoint-config/c",
                             "CreationTime": NOW,
                             "ProductionVariants": [{"VariantName": "AllTraffic", "ModelName": "ccfd-rt-model-v2"}]},
                            {"EndpointConfigName": "ccfd-rt-config-v2"})
    target = {"ServiceNamespace": "sagemaker", "ResourceId": "endpoint/ccfd-rt/variant/AllTraffic",
              "ScalableDimension": "sagemaker:variant:DesiredInstanceCount", "MinCapacity": 1, "MaxCapacity": 2,
              "RoleARN": "arn:aws:iam::<ACCOUNT_ID>:role/autoscaling", "CreationTime": NOW}
    other = {**target, "ResourceId": "endpoint/someone-else/variant/AllTraffic"}
    aas.stubber.add_response("describe_scalable_targets", {"ScalableTargets": [target, other]},
                             {"ServiceNamespace": "sagemaker"})
    aas.stubber.add_response("deregister_scalable_target", {},
                             {"ServiceNamespace": "sagemaker", "ResourceId": "endpoint/ccfd-rt/variant/AllTraffic",
                              "ScalableDimension": "sagemaker:variant:DesiredInstanceCount"})
    sm.stubber.add_response("delete_endpoint", {}, {"EndpointName": "ccfd-rt"})
    not_found(sm, "ccfd-rt")
    sm.stubber.add_response("list_endpoint_configs",
                            {"EndpointConfigs": [{"EndpointConfigName": "ccfd-rt-config-v1", "EndpointConfigArn": f"{SM_ARN}:ec/1",
                                                  "CreationTime": NOW}], "NextToken": "page2"},
                            {"NameContains": "ccfd-rt-"})
    sm.stubber.add_response("list_endpoint_configs",          # a second page: delete() reads every page
                            {"EndpointConfigs": [{"EndpointConfigName": "ccfd-rt-config-v2", "EndpointConfigArn": f"{SM_ARN}:ec/2",
                                                  "CreationTime": NOW}]},
                            {"NameContains": "ccfd-rt-", "NextToken": "page2"})
    sm.stubber.add_response("list_models",
                            {"Models": [{"ModelName": "ccfd-rt-model-v1", "ModelArn": f"{SM_ARN}:model/1", "CreationTime": NOW}]},
                            {"NameContains": "ccfd-rt-"})
    for name in ("ccfd-rt-config-v1", "ccfd-rt-config-v2"):
        sm.stubber.add_response("delete_endpoint_config", {}, {"EndpointConfigName": name})
    for name in ("ccfd-rt-model-v1", "ccfd-rt-model-v2"):
        sm.stubber.add_response("delete_model", {}, {"ModelName": name})
    gone = endpoints.delete("ccfd-rt", sm=sm, aas=aas, sleep=lambda s: None, **NO_LOG)
    assert gone == {"scalable_targets": ["endpoint/ccfd-rt/variant/AllTraffic"], "endpoint": ["ccfd-rt"],
                    "configs": ["ccfd-rt-config-v1", "ccfd-rt-config-v2"],
                    "models": ["ccfd-rt-model-v1", "ccfd-rt-model-v2"]}
    assert order.index("DeregisterScalableTarget") < order.index("DeleteEndpoint") < order.index("DeleteEndpointConfig")
    assert order.index("DeleteEndpointConfig") < order.index("DeleteModel")


def test_delete_with_the_endpoint_gone_still_deletes_its_leftover_configs(sm, aas):
    not_found(sm, "ccfd-staging")
    aas.stubber.add_response("describe_scalable_targets", {"ScalableTargets": []}, {"ServiceNamespace": "sagemaker"})
    sm.stubber.add_response("list_endpoint_configs",
                            {"EndpointConfigs": [{"EndpointConfigName": "ccfd-staging-config-v3",
                                                  "EndpointConfigArn": f"{SM_ARN}:ec/3", "CreationTime": NOW}]},
                            {"NameContains": "ccfd-staging-"})
    sm.stubber.add_response("list_models", {"Models": []}, {"NameContains": "ccfd-staging-"})
    sm.stubber.add_response("delete_endpoint_config", {}, {"EndpointConfigName": "ccfd-staging-config-v3"})
    gone = endpoints.delete("ccfd-staging", sm=sm, aas=aas, **NO_LOG)
    assert gone == {"scalable_targets": [], "endpoint": [], "configs": ["ccfd-staging-config-v3"], "models": []}
    assert "DeleteEndpoint" not in sm.calls and aas.calls == ["DescribeScalableTargets"]


# ---------------------------------------------------------------------------
# 5. every create call in endpoints.py carries Tags
# ---------------------------------------------------------------------------
def test_every_create_call_in_endpoints_py_carries_tags():
    tree = ast.parse(Path(endpoints.__file__).read_text(encoding="utf-8"))
    creates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Attribute) and argument.attr.startswith(("create_", "put_metric_alarm")):
                creates.append((argument.attr, node))
    assert {name for name, _ in creates} >= {"create_model", "create_endpoint_config", "create_endpoint",
                                            "put_metric_alarm"}
    for name, node in creates:
        keywords = {k.arg for k in node.keywords}
        if name == "create_endpoint_config":   # its request dict holds Tags (checked by the stubbed tests above)
            continue
        assert "Tags" in keywords or None in keywords, f"{name} without Tags"


# ---------------------------------------------------------------------------
# 6. The command line the ops scripts call (versions are printed, never ARNs)
# ---------------------------------------------------------------------------
def test_cli_prints_versions_never_arns(monkeypatch, capsys):
    monkeypatch.setattr(endpoints, "approved_versions", lambda: [1, 2])
    assert endpoints.main(["approved-versions"]) == 0
    assert capsys.readouterr().out == "1\n2\n"


def test_cli_delete_commands_and_errors_are_masked(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(endpoints, "delete_rollout_alarms", lambda **kwargs: seen.setdefault("alarms", kwargs))
    monkeypatch.setattr(endpoints, "delete", lambda name, **kwargs: seen.setdefault("delete", (name, kwargs)))
    assert endpoints.main(["delete-rollout-alarms"]) == 0
    assert endpoints.main(["delete", "--endpoint", "ccfd-rt"]) == 0
    assert seen["alarms"] == {"log": endpoints._say} and seen["delete"] == ("ccfd-rt", {"log": endpoints._say})

    def refuse(**kwargs):
        raise AwsCallFailed(f"PutMetricAlarm failed: arn:aws:iam::{ACCOUNT}:user/x", "AccessDenied", "PutMetricAlarm")
    monkeypatch.setattr(endpoints, "create_rollout_alarms", refuse)
    capsys.readouterr()
    assert endpoints.main(["create-rollout-alarms"]) == 1
    shown = capsys.readouterr()
    assert ACCOUNT not in shown.out + shown.err
    assert "ERROR: PutMetricAlarm failed" in shown.err and "<ACCOUNT_ID>" in shown.err


def test_cli_error_goes_to_stderr_so_a_captured_stdout_still_shows_it(monkeypatch, capsys):
    """catch-up-part3.sh runs VERSIONS="$(python -m mlops_utils.endpoints approved-versions)": stdout is captured."""
    def refuse(**kwargs):
        raise AwsCallFailed("ListModelPackages failed (EndpointConnectionError): could not connect", "x", "y")
    monkeypatch.setattr(endpoints, "approved_versions", refuse)
    assert endpoints.main(["approved-versions"]) == 1
    shown = capsys.readouterr()
    assert shown.out == ""
    assert shown.err.startswith("ERROR: ListModelPackages failed (EndpointConnectionError)")


def test_deploy_package_never_reuses_a_config_with_other_settings(sm):
    """Configs are immutable: ccfd-prod-config-v1 without capture is not used when capture is asked for."""
    sm.stubber.add_response("list_model_packages", {"ModelPackageSummaryList": [package(1)]},
                            {"ModelPackageGroupName": "ccfd-fraud-detector", "SortBy": "CreationTime",
                             "SortOrder": "Ascending"})
    sm.stubber.add_response("describe_model", {"ModelName": "ccfd-prod-model-v1", "ModelArn": f"{SM_ARN}:model/v1",
                                               "CreationTime": NOW}, {"ModelName": "ccfd-prod-model-v1"})
    sm.stubber.add_response("describe_endpoint_config",
                            {"EndpointConfigName": "ccfd-prod-config-v1", "EndpointConfigArn": f"{SM_ARN}:ec/v1",
                             "CreationTime": NOW,
                             "ProductionVariants": [{"VariantName": "AllTraffic", "InitialInstanceCount": 2}]},
                            {"EndpointConfigName": "ccfd-prod-config-v1"})
    capture = endpoints.data_capture_config("s3://ccfd-mlops-jobs-a1b2c3/data-capture/")
    with pytest.raises(ValueError, match="data capture off, not 2 and on"):
        endpoints.deploy_package("ccfd-prod", 1, instance_count=2, data_capture=capture, role_arn=ROLE, tags=TAGS,
                                 sm=sm, **NO_LOG)
    assert not [call for call in sm.calls if call.startswith(("Create", "Update"))]
