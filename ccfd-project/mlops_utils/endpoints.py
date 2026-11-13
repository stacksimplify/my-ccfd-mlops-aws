"""
The endpoint helper: every course script that deploys, updates, calls or deletes a SageMaker
endpoint uses this file, so the endpoint settings live in one place.

  Packages of ccfd-fraud-detector
    approved_versions()          the Approved package versions, oldest first (the first one is the ORIGINAL champion)
    newest_approved()            the newest Approved package (what "the champion" means in the SageMaker registry)
    package_arn()                the ARN of one package version (ARNs are read, never built)
  Models and configs (names as MLOps 10 and 12: <endpoint>-model-v<N>, <endpoint>-config-v<N>)
    create_model_from_package()  an Approved package -> a model; SageMaker refuses any other status
    create_plain_model()         ANY package -> a plain model from its Image + ModelDataUrl (staging in MLOps 14, the
                                 rollback drill of MLOps 12 with the Rejected data v2 retrain): never ModelPackageName
    create_endpoint_config()     ml.m5.large x N, variant AllTraffic, optional data capture (data_capture_config())
  Endpoints
    deploy(), update(), delete(), wait_for_endpoint(), wait_for_endpoint_deleted(), endpoint_exists()
    deploy_package()             model + config + endpoint (created, or updated when it exists) for one version
  Calls
    invoke()                     invoke_endpoint; on the first failure it retries exactly once, then raises
  The 2 rollout alarms of MLOps 12 Step-01, on ccfd-prod
    ccfd-prod-rollout-5xx         Invocation5XXErrors Sum > 0 over 60 s
    ccfd-prod-rollout-netbenefit  NetBenefit (MLOps/CCFD, Source=TestSetReplay) Minimum < 33,000 over 60 s
    create_rollout_alarms(), delete_rollout_alarms(), rollout_alarm_requests(), canary_deployment_config()
    Missing data never breaches: an alarm only fires on real numbers.

Every create call carries the course tags (course, owner, purpose, created from .env.mlops, mlops_utils/config.py).
Every message this file prints, and every AWS error it raises (jobs.AwsCallFailed), has the 12-digit account ID
masked. The describe answers it returns hold ARNs: mask them before you print them.

From a script (run from the ccfd-project folder; the output names versions, never ARNs; an error prints one
masked "ERROR: ..." line on stderr and exits 1):
    python -m mlops_utils.endpoints approved-versions
    python -m mlops_utils.endpoints deploy-package --endpoint ccfd-prod --version 1 --instances 2 [--capture]
    python -m mlops_utils.endpoints create-rollout-alarms
    python -m mlops_utils.endpoints delete --endpoint ccfd-prod
    python -m mlops_utils.endpoints delete-rollout-alarms
"""

import argparse
import json
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from mlops_utils import config
from mlops_utils.jobs import AwsCallFailed

# ---------------------------------------------------------------------------
# Fixed names and settings (12 and 15)
# ---------------------------------------------------------------------------
GROUP = config.MODEL_PACKAGE_GROUP            # ccfd-fraud-detector
REALTIME_ENDPOINT = "ccfd-rt"                 # MLOps 10
SERVERLESS_ENDPOINT = "ccfd-serverless"       # MLOps 11
STAGING_ENDPOINT = "ccfd-staging"             # MLOps 14
PROD_ENDPOINT = "ccfd-prod"                   # MLOps 12 on: the production endpoint with the rollout alarms
VARIANT = "AllTraffic"
INSTANCE_TYPE = "ml.m5.large"                 # the only endpoint instance type of the course
CONTENT_TYPE = "application/json"

METRIC_NAMESPACE = "MLOps/CCFD"
NETBENEFIT_METRIC = "NetBenefit"
REPLAY_SOURCE = "TestSetReplay"               # MLOps 12: only the test-set replay feeds the rollout alarm
NETBENEFIT_FLOOR = 33000                      # about 10 percent under the champion's 36,975 on the MLOps 12 probe set
#                                               (the gate holdout of MLOps 09, seed 301): set from the champion alone
ROLLOUT_5XX = "ccfd-prod-rollout-5xx"
ROLLOUT_NETBENEFIT = "ccfd-prod-rollout-netbenefit"
ROLLOUT_ALARMS = (ROLLOUT_5XX, ROLLOUT_NETBENEFIT)

# The canary of MLOps 12 Step-01: 1 instance first, bake 300 s, final bake 120 s, roll back on any alarm
CANARY_INSTANCES = 1
CANARY_BAKE_SECONDS = 300
FINAL_BAKE_SECONDS = 120

# The data capture of MLOps 15: every request and answer, JSON and JSON Lines kept readable
CAPTURE_PREFIX = "data-capture/"              # under s3://<jobs bucket>/

POLL_SECONDS = 30
RETRY_WAIT_SECONDS = 2


class PackageNotApproved(ValueError):
    """create_model_from_package() was given a package that is not Approved (SageMaker would refuse it)."""


# ---------------------------------------------------------------------------
# Clients and small helpers
# ---------------------------------------------------------------------------
def _sm(sm=None):
    return sm if sm is not None else boto3.client("sagemaker", region_name=config.REGION)


def _cw(cw=None):
    return cw if cw is not None else boto3.client("cloudwatch", region_name=config.REGION)


def _aas(aas=None):
    return aas if aas is not None else boto3.client("application-autoscaling", region_name=config.REGION)


def runtime_client():
    """
    The sagemaker-runtime client invoke() uses: botocore's own retries are off (1 attempt in total), so
    "exactly one retry" in invoke() really means 2 calls at most.
    """
    return boto3.client("sagemaker-runtime", region_name=config.REGION,
                        config=Config(retries={"total_max_attempts": 1}))


def _tags(tags):
    """The course tags as boto3 wants them; the default comes from mlops_utils/config.py (.env.mlops)."""
    return tags if tags is not None else config.load().aws_tags()


def _role(role_arn):
    """The SageMaker execution role (MLOPS_EXECUTION_ROLE_ARN in .env.mlops, written by create-env.sh)."""
    role = role_arn or config.load().get("MLOPS_EXECUTION_ROLE_ARN")
    if not role:
        raise ValueError("MLOPS_EXECUTION_ROLE_ARN not set: run scripts/create-env.sh first (it writes .env.mlops).")
    return role


def _error_parts(error):
    if isinstance(error, ClientError):
        details = error.response.get("Error", {})
        return details.get("Code") or "Unknown", details.get("Message") or str(error)
    return type(error).__name__, str(error)


def _call(operation, function, **request):
    """One AWS call. An AWS error comes back as AwsCallFailed with the code and the masked text."""
    try:
        return function(**request)
    except (ClientError, BotoCoreError) as error:
        code, text = _error_parts(error)
        raise AwsCallFailed(config.mask(f"{operation} failed ({code}): {text}"), code, operation) from None


def _list_all(operation, function, key, **request):
    """Every item of a paged list call (SageMaker list calls give 10 items a page by default)."""
    found = []
    while True:
        page = _call(operation, function, **request)
        found += page.get(key, [])
        if not page.get("NextToken"):
            return found
        request["NextToken"] = page["NextToken"]


def _find(operation, function, missing_text, **request) -> dict | None:
    """The describe answer, or None when SageMaker says the object does not exist."""
    try:
        return function(**request)
    except ClientError as error:
        code, text = _error_parts(error)
        if code == "ValidationException" and missing_text in text:
            return None
        raise AwsCallFailed(config.mask(f"{operation} failed ({code}): {text}"), code, operation) from None


def model_names(endpoint: str, version) -> tuple[str, str]:
    """(model name, config name) for one package version on one endpoint: ccfd-rt-model-v1, ccfd-rt-config-v1."""
    return f"{endpoint}-model-v{version}", f"{endpoint}-config-v{version}"


# ---------------------------------------------------------------------------
# 1. Packages (read only)
# ---------------------------------------------------------------------------
def _packages(sm, group, **filters):
    request = {"ModelPackageGroupName": group, "SortBy": "CreationTime", "SortOrder": "Ascending", **filters}
    found = []
    while True:
        page = _call("ListModelPackages", sm.list_model_packages, **request)
        found += page.get("ModelPackageSummaryList", [])
        if not page.get("NextToken"):
            return found
        request["NextToken"] = page["NextToken"]


def approved_versions(group: str = GROUP, sm=None) -> list[int]:
    """The Approved versions of the group, oldest first: [1] after MLOps 09, [1, 4] after MLOps 16."""
    return [p["ModelPackageVersion"] for p in _packages(_sm(sm), group, ModelApprovalStatus="Approved")]


def newest_approved(group: str = GROUP, sm=None) -> dict | None:
    """
    The newest Approved package summary (ModelPackageArn, ModelPackageVersion), or None when there is none.
    The pages are followed until one holds a package: a page may come back empty with a NextToken when the
    newest versions are not Approved (after MLOps 14, version 3 is Rejected), so an empty first page never means
    "no Approved package".
    """
    sm = _sm(sm)
    request = {"ModelPackageGroupName": group, "ModelApprovalStatus": "Approved", "SortBy": "CreationTime",
               "SortOrder": "Descending", "MaxResults": 1}
    while True:
        page = _call("ListModelPackages", sm.list_model_packages, **request)
        found = page.get("ModelPackageSummaryList", [])
        if found:
            return found[0]
        if not page.get("NextToken"):
            return None
        request["NextToken"] = page["NextToken"]


def package_arn(version, group: str = GROUP, sm=None) -> str:
    """The ARN of package version <version> in the group (read from the registry)."""
    for summary in _packages(_sm(sm), group):
        if str(summary.get("ModelPackageVersion")) == str(version):
            return summary["ModelPackageArn"]
    raise LookupError(f"No version {version} in model package group {group}.")


# ---------------------------------------------------------------------------
# 2. Models and endpoint configs
# ---------------------------------------------------------------------------
def create_model_from_package(model_name: str, package: str, *, role_arn: str | None = None,
                              tags: list | None = None, sm=None, log=print) -> str:
    """
    A model that points at an Approved package (the image and the model data come from the package; nothing is
    typed). SageMaker refuses a package that is not Approved, so this checks first and raises
    PackageNotApproved without any create call.
    """
    sm = _sm(sm)
    described = _call("DescribeModelPackage", sm.describe_model_package, ModelPackageName=package)
    status = described.get("ModelApprovalStatus")
    if status != "Approved":
        raise PackageNotApproved(config.mask(
            f"Package version {described.get('ModelPackageVersion')} is {status}, not Approved: only an Approved "
            "package deploys. For a smoke test of a package that is not Approved, use create_plain_model()."))
    _call("CreateModel", sm.create_model, ModelName=model_name, ExecutionRoleArn=_role(role_arn),
          PrimaryContainer={"ModelPackageName": package}, Tags=_tags(tags))
    log(f"model {model_name} created from package version {described.get('ModelPackageVersion')} (Approved)")
    return model_name


def create_plain_model(model_name: str, package: str, *, role_arn: str | None = None, tags: list | None = None,
                       sm=None, log=print) -> str:
    """
    A plain model from a package's image and model data (InferenceSpecification.Containers[0]), whatever its
    approval status: ccfd-staging (MLOps 14) and the rollback drill of MLOps 12 (the Rejected data v2 retrain) use it.
    """
    sm = _sm(sm)
    described = _call("DescribeModelPackage", sm.describe_model_package, ModelPackageName=package)
    container = described["InferenceSpecification"]["Containers"][0]
    _call("CreateModel", sm.create_model, ModelName=model_name, ExecutionRoleArn=_role(role_arn),
          PrimaryContainer={"Image": container["Image"], "ModelDataUrl": container["ModelDataUrl"]},
          Tags=_tags(tags))
    log(f"model {model_name} created as a plain model from package version "
        f"{described.get('ModelPackageVersion')} ({described.get('ModelApprovalStatus')})")
    return model_name


def data_capture_config(destination_s3_uri: str | None = None, sampling_percentage: int = 100) -> dict:
    """
    The data capture of MLOps 15: requests and answers, 100 percent, to
    s3://<jobs bucket>/data-capture/; JSON and JSON Lines are stored as text, not BASE64.
    """
    if destination_s3_uri is None:
        jobs = config.load().bucket("jobs")
        if not jobs:
            raise ValueError("The jobs bucket is unknown: run scripts/create-env.sh first (MLOPS_SUFFIX in .env.mlops).")
        destination_s3_uri = f"s3://{jobs}/{CAPTURE_PREFIX}"
    return {
        "EnableCapture": True,
        "InitialSamplingPercentage": sampling_percentage,
        "DestinationS3Uri": destination_s3_uri,
        "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json", "application/jsonlines"]},
    }


def create_endpoint_config(config_name: str, model_name: str, *, instance_count: int = 1,
                           instance_type: str = INSTANCE_TYPE, variant: str = VARIANT,
                           data_capture: dict | None = None, tags: list | None = None, sm=None, log=print) -> str:
    """One production variant (AllTraffic, ml.m5.large x instance_count); data_capture from data_capture_config()."""
    request = {
        "EndpointConfigName": config_name,
        "ProductionVariants": [{"VariantName": variant, "ModelName": model_name, "InstanceType": instance_type,
                                "InitialInstanceCount": instance_count}],
        "Tags": _tags(tags),
    }
    if data_capture:
        request["DataCaptureConfig"] = data_capture
    _call("CreateEndpointConfig", _sm(sm).create_endpoint_config, **request)
    log(f"endpoint config {config_name}: {variant}, {instance_type} x {instance_count}"
        + (", data capture on" if data_capture else ""))
    return config_name


# ---------------------------------------------------------------------------
# 3. Endpoints
# ---------------------------------------------------------------------------
def describe(endpoint: str, sm=None) -> dict | None:
    """describe_endpoint, or None when the endpoint does not exist."""
    try:
        return _sm(sm).describe_endpoint(EndpointName=endpoint)
    except (ClientError, BotoCoreError) as error:
        code, text = _error_parts(error)
        if code == "ValidationException" and "Could not find endpoint" in text:
            return None
        raise AwsCallFailed(config.mask(f"DescribeEndpoint failed ({code}): {text}"), code,
                            "DescribeEndpoint") from None


def endpoint_exists(endpoint: str, sm=None) -> bool:
    return describe(endpoint, sm) is not None


def wait_for_endpoint(endpoint: str, *, sm=None, poll_seconds: int = POLL_SECONDS, timeout_seconds: int = 3600,
                      sleep=time.sleep, log=print, clock=time.monotonic) -> dict:
    """Poll describe_endpoint until InService (returned) or Failed (raises), printing each status change."""
    start, last = clock(), None
    while True:
        described = describe(endpoint, sm)
        if described is None:
            raise RuntimeError(f"Endpoint {endpoint} does not exist.")
        status = described["EndpointStatus"]
        if status != last:
            log(f"  {time.strftime('%H:%M:%S')} {endpoint}: {status}")
            last = status
        if status == "InService":
            return described
        if status == "Failed":
            raise RuntimeError(config.mask(f"Endpoint {endpoint} Failed: {described.get('FailureReason')}"))
        if clock() - start > timeout_seconds:
            raise TimeoutError(f"Endpoint {endpoint} is still {status} after {timeout_seconds} s.")
        sleep(poll_seconds)


def wait_for_endpoint_deleted(endpoint: str, *, sm=None, poll_seconds: int = 15, timeout_seconds: int = 1800,
                              sleep=time.sleep, log=print, clock=time.monotonic) -> None:
    """Poll until describe_endpoint says the endpoint does not exist any more."""
    start = clock()
    while describe(endpoint, sm) is not None:
        if clock() - start > timeout_seconds:
            raise TimeoutError(f"Endpoint {endpoint} still exists after {timeout_seconds} s.")
        sleep(poll_seconds)
    log(f"  {time.strftime('%H:%M:%S')} {endpoint}: deleted")


def deploy(endpoint: str, config_name: str, *, tags: list | None = None, sm=None, wait: bool = True,
           log=print, **wait_args) -> dict | None:
    """create_endpoint (tagged), then wait until InService (one ml.m5.large is 0.115 USD an hour)."""
    _call("CreateEndpoint", _sm(sm).create_endpoint, EndpointName=endpoint, EndpointConfigName=config_name,
          Tags=_tags(tags))
    log(f"endpoint {endpoint} creating with {config_name}")
    return wait_for_endpoint(endpoint, sm=sm, log=log, **wait_args) if wait else None


def update(endpoint: str, config_name: str, *, deployment_config: dict | None = None, sm=None, wait: bool = True,
           log=print, **wait_args) -> dict | None:
    """update_endpoint to another config (blue/green; with deployment_config a canary), then wait."""
    request = {"EndpointName": endpoint, "EndpointConfigName": config_name}
    if deployment_config:
        request["DeploymentConfig"] = deployment_config
    _call("UpdateEndpoint", _sm(sm).update_endpoint, **request)
    log(f"endpoint {endpoint} updating to {config_name}" + (" (canary)" if deployment_config else ""))
    return wait_for_endpoint(endpoint, sm=sm, log=log, **wait_args) if wait else None


def deploy_package(endpoint: str, version, *, instance_count: int = 1, plain: bool = False,
                   data_capture: dict | None = None, group: str = GROUP, role_arn: str | None = None,
                   tags: list | None = None, sm=None, log=print, **wait_args) -> dict | None:
    """
    One package version on one endpoint: model <endpoint>-model-v<N> and config <endpoint>-config-v<N> (made
    only when missing), then the endpoint is created, or updated when it serves another config.
    plain=True makes a plain model (create_plain_model) instead of a package model.
    """
    sm = _sm(sm)
    model_name, config_name = model_names(endpoint, version)
    arn = package_arn(version, group, sm)
    if _find("DescribeModel", sm.describe_model, "Could not find model", ModelName=model_name) is None:
        maker = create_plain_model if plain else create_model_from_package
        maker(model_name, arn, role_arn=role_arn, tags=tags, sm=sm, log=log)
    kept = _find("DescribeEndpointConfig", sm.describe_endpoint_config, "Could not find endpoint configuration",
                 EndpointConfigName=config_name)
    if kept is None:
        create_endpoint_config(config_name, model_name, instance_count=instance_count, data_capture=data_capture,
                               tags=tags, sm=sm, log=log)
    else:
        # A config never changes: a kept one is used again only when it has the settings asked for
        count = kept["ProductionVariants"][0].get("InitialInstanceCount")
        capture = bool(kept.get("DataCaptureConfig", {}).get("EnableCapture"))
        if count != instance_count or capture != bool(data_capture):
            raise ValueError(f"Endpoint config {config_name} exists with {count} instance(s) and data capture "
                             f"{'on' if capture else 'off'}, not {instance_count} and {'on' if data_capture else 'off'}. "
                             "A config cannot change: delete the endpoint and its configs first "
                             f"(python -m mlops_utils.endpoints delete --endpoint {endpoint}), then deploy again.")
    described = describe(endpoint, sm)
    if described is None:
        return deploy(endpoint, config_name, tags=tags, sm=sm, log=log, **wait_args)
    if described.get("EndpointConfigName") == config_name:
        log(f"endpoint {endpoint} already serves {config_name} ({described['EndpointStatus']})")
        return wait_for_endpoint(endpoint, sm=sm, log=log, **wait_args)
    return update(endpoint, config_name, sm=sm, log=log, **wait_args)


def delete(endpoint: str, *, sm=None, aas=None, log=print, **wait_args) -> dict:
    """
    Remove one endpoint and everything named after it (MLOps 10 Step-05 order):
      1. its autoscaling targets (they own the TargetTracking alarms), 2. the endpoint, then wait until it is gone,
      3. every endpoint config and 4. every model whose name starts with '<endpoint>-' (old configs kept for a
      rollback included), and the config and models the endpoint used.
    Returns the deleted names, by kind.
    """
    sm, aas = _sm(sm), _aas(aas)
    gone = {"scalable_targets": [], "endpoint": [], "configs": [], "models": []}
    described = describe(endpoint, sm)
    configs, models = set(), set()
    if described is not None:
        configs.add(described["EndpointConfigName"])
        try:
            used = sm.describe_endpoint_config(EndpointConfigName=described["EndpointConfigName"])
            models.update(v["ModelName"] for v in used.get("ProductionVariants", []) if v.get("ModelName"))
        except ClientError:
            pass                                    # the config is gone already; the prefix search below still runs
    targets = _list_all("DescribeScalableTargets", aas.describe_scalable_targets, "ScalableTargets",
                        ServiceNamespace="sagemaker")
    for target in targets:
        if target["ResourceId"].startswith(f"endpoint/{endpoint}/"):
            gone["scalable_targets"].append(target["ResourceId"])
            _call("DeregisterScalableTarget", aas.deregister_scalable_target, ServiceNamespace="sagemaker",
                  ResourceId=target["ResourceId"], ScalableDimension=target["ScalableDimension"])
    if described is not None:
        gone["endpoint"].append(endpoint)
        _call("DeleteEndpoint", sm.delete_endpoint, EndpointName=endpoint)
        wait_for_endpoint_deleted(endpoint, sm=sm, log=log, **wait_args)
    prefix = f"{endpoint}-"
    listed = _list_all("ListEndpointConfigs", sm.list_endpoint_configs, "EndpointConfigs", NameContains=prefix)
    configs.update(c["EndpointConfigName"] for c in listed if c["EndpointConfigName"].startswith(prefix))
    listed = _list_all("ListModels", sm.list_models, "Models", NameContains=prefix)
    models.update(m["ModelName"] for m in listed if m["ModelName"].startswith(prefix))
    for name in sorted(configs):
        gone["configs"].append(name)
        _call("DeleteEndpointConfig", sm.delete_endpoint_config, EndpointConfigName=name)
    for name in sorted(models):
        gone["models"].append(name)
        _call("DeleteModel", sm.delete_model, ModelName=name)
    # One fact per line, each with its label
    log(f"{endpoint}: deleted")
    log(f"  autoscaling targets: {len(gone['scalable_targets'])}")
    log(f"  endpoint: {', '.join(gone['endpoint']) or 'none'}")
    log(f"  configs:  {', '.join(gone['configs']) or 'none'}")
    log(f"  models:   {', '.join(gone['models']) or 'none'}")
    return gone


# ---------------------------------------------------------------------------
# 4. Calls: exactly one retry (a serverless cold start can fail the first call)
# ---------------------------------------------------------------------------
def invoke(endpoint: str, body, *, content_type: str = CONTENT_TYPE, accept: str = CONTENT_TYPE,
           target_variant: str | None = None, inference_id: str | None = None, runtime=None, sleep=time.sleep,
           retry_wait_seconds: float = RETRY_WAIT_SECONDS, log=print) -> dict:
    """
    invoke_endpoint once; if that call fails (any AWS error), wait retry_wait_seconds and call exactly once more;
    if the second call fails too, raise AwsCallFailed (masked). Every caller uses this function.
    body: bytes or str are sent as they are; a dict or a list is sent as JSON (json.dumps); anything else
    raises TypeError (str() of a dict is a Python repr, not JSON, and the model would refuse it).
    Returns {"body": the answer as text, "variant": the InvokedProductionVariant, "attempts": 1 or 2}.
    """
    if isinstance(body, (dict, list)):
        body = json.dumps(body)
    elif not isinstance(body, (bytes, bytearray, str)):
        raise TypeError(f"invoke() body must be bytes, str, a dict or a list, not {type(body).__name__}")
    runtime = runtime if runtime is not None else runtime_client()
    request = {"EndpointName": endpoint, "Body": body, "ContentType": content_type, "Accept": accept}
    if target_variant:
        request["TargetVariant"] = target_variant
    if inference_id:
        request["InferenceId"] = inference_id
    for attempt in (1, 2):
        try:
            answer = runtime.invoke_endpoint(**request)
        except (ClientError, BotoCoreError) as error:
            code, text = _error_parts(error)
            if attempt == 1:
                log(config.mask(f"invoke {endpoint}: first call failed ({code}): {text}; one retry in "
                                f"{retry_wait_seconds} s"))
                sleep(retry_wait_seconds)
                continue
            raise AwsCallFailed(config.mask(f"InvokeEndpoint {endpoint} failed twice ({code}): {text}"), code,
                                "InvokeEndpoint") from None
        return {"body": answer["Body"].read().decode("utf-8"), "variant": answer.get("InvokedProductionVariant"),
                "attempts": attempt}
    raise AssertionError("not reached")


# ---------------------------------------------------------------------------
# 5. The 2 rollout alarms (MLOps 12 Step-01 and its Notes)
# ---------------------------------------------------------------------------
def rollout_alarm_requests(endpoint: str = PROD_ENDPOINT, variant: str = VARIANT, tags: list | None = None) -> list:
    """The 2 put_metric_alarm requests. Missing data is not breaching, so a quiet endpoint never rolls back."""
    tags = _tags(tags)
    on_variant = [{"Name": "EndpointName", "Value": endpoint}, {"Name": "VariantName", "Value": variant}]
    common = {"Period": 60, "TreatMissingData": "notBreaching", "Tags": tags}
    return [
        {"AlarmName": ROLLOUT_5XX, "AlarmDescription": f"{endpoint}: any 5xx answer in 60 s rolls a deployment back.",
         "Namespace": "AWS/SageMaker", "MetricName": "Invocation5XXErrors", "Dimensions": on_variant,
         "Statistic": "Sum", "EvaluationPeriods": 1, "DatapointsToAlarm": 1, "Threshold": 0,
         "ComparisonOperator": "GreaterThanThreshold", **common},
        {"AlarmName": ROLLOUT_NETBENEFIT,
         "AlarmDescription": (f"{endpoint}: the probe-set replay Net Benefit below {NETBENEFIT_FLOOR} USD rolls a "
                              "deployment back (Net Benefit decides)."),
         "Namespace": METRIC_NAMESPACE, "MetricName": NETBENEFIT_METRIC,
         "Dimensions": [{"Name": "EndpointName", "Value": endpoint}, {"Name": "Source", "Value": REPLAY_SOURCE}],
         "Statistic": "Minimum", "EvaluationPeriods": 1, "DatapointsToAlarm": 1, "Threshold": NETBENEFIT_FLOOR,
         "ComparisonOperator": "LessThanThreshold", **common},
    ]


def create_rollout_alarms(*, endpoint: str = PROD_ENDPOINT, tags: list | None = None, cw=None, log=print) -> list:
    """Create (or overwrite) the 2 rollout alarms. They exist before any canary refers to them (MLOps 12 Notes)."""
    cw = _cw(cw)
    names = []
    for request in rollout_alarm_requests(endpoint, tags=tags):
        _call("PutMetricAlarm", cw.put_metric_alarm, **request)
        names.append(request["AlarmName"])
    log(f"rollout alarms created: {', '.join(names)}")
    return names


def delete_rollout_alarms(*, cw=None, log=print) -> list:
    """
    Delete the rollout alarms that exist (DeleteAlarms deletes nothing when one name is unknown, so only the
    existing names are sent). Returns the names deleted.
    """
    cw = _cw(cw)
    found = _call("DescribeAlarms", cw.describe_alarms, AlarmNames=list(ROLLOUT_ALARMS))
    names = [alarm["AlarmName"] for alarm in found.get("MetricAlarms", [])]
    if names:
        _call("DeleteAlarms", cw.delete_alarms, AlarmNames=names)
    log(f"rollout alarms: deleted {', '.join(names) if names else 'none'}")
    return names


def canary_deployment_config(canary_instances: int = CANARY_INSTANCES, bake_seconds: int = CANARY_BAKE_SECONDS,
                             final_bake_seconds: int = FINAL_BAKE_SECONDS) -> dict:
    """
    The DeploymentConfig of the MLOps 12 canary for update(): CANARY with 1 instance, bake 300 s
    (WaitIntervalInSeconds), final bake 120 s (TerminationWaitInSeconds: the old fleet stays that long), and
    auto-rollback when either of the 2 rollout alarms goes to ALARM.
    """
    return {
        "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": {
                "Type": "CANARY",
                "CanarySize": {"Type": "INSTANCE_COUNT", "Value": canary_instances},
                "WaitIntervalInSeconds": bake_seconds,
            },
            "TerminationWaitInSeconds": final_bake_seconds,
        },
        "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": name} for name in ROLLOUT_ALARMS]},
    }


# ---------------------------------------------------------------------------
# 6. Command line, for scripts/destroy-env.sh and the steps of MLOps 14 and 16
# ---------------------------------------------------------------------------
def _say(text):
    print(config.mask(text), flush=True)


def _say_error(text):
    """ERROR lines go to stderr: a script that captures stdout (VERSIONS="$(...)") still shows them."""
    print(config.mask(text), file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mlops_utils.endpoints",
                                     description="The endpoint helper (versions are printed, never ARNs).")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("approved-versions", help="the Approved versions of ccfd-fraud-detector, oldest first")
    deploy_cmd = commands.add_parser("deploy-package", help="one package version on one endpoint")
    deploy_cmd.add_argument("--endpoint", required=True)
    deploy_cmd.add_argument("--version", required=True)
    deploy_cmd.add_argument("--instances", type=int, default=1)
    deploy_cmd.add_argument("--plain", action="store_true", help="a plain model (Image + ModelDataUrl)")
    deploy_cmd.add_argument("--capture", action="store_true", help="data capture to s3://<jobs>/data-capture/")
    delete_cmd = commands.add_parser("delete", help="an endpoint with its autoscaling, configs and models")
    delete_cmd.add_argument("--endpoint", required=True)
    commands.add_parser("create-rollout-alarms", help="the 2 ccfd-prod rollout alarms")
    commands.add_parser("delete-rollout-alarms", help="the rollout alarms that exist")
    args = parser.parse_args(argv)
    try:
        if args.command == "approved-versions":
            for version in approved_versions():
                print(version)
        elif args.command == "deploy-package":
            capture = data_capture_config() if args.capture else None
            deploy_package(args.endpoint, args.version, instance_count=args.instances, plain=args.plain,
                           data_capture=capture, log=_say)
        elif args.command == "delete":
            delete(args.endpoint, log=_say)
        elif args.command == "create-rollout-alarms":
            create_rollout_alarms(log=_say)
        elif args.command == "delete-rollout-alarms":
            delete_rollout_alarms(log=_say)
    except (AwsCallFailed, LookupError, ValueError, RuntimeError, TimeoutError) as error:
        _say_error(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
