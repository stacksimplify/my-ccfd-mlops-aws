"""
MLOps 14: the small release helpers that cd.yml and rollback.yml call through the m14_0*_*.py scripts.

The two registries: the SageMaker Model Registry (ccfd-fraud-detector, written by boto3) decides what may ship;
the MLflow alias @champion (ccfd-fraud-model) follows it after a release. They are 2 separate writes, so a release can
be half done: m14_04_check_sync.py reports that.
"""

import boto3

from mlops_utils import config, endpoints, mlflow_layer

PROD = endpoints.PROD_ENDPOINT
STAGING = "ccfd-staging"                          # MLOps 14
PROD_INSTANCES = 2


def sm_client(sm=None):
    return sm or boto3.client("sagemaker", region_name=config.REGION)


def version_of(package_arn: str) -> int:
    """'.../model-package/ccfd-fraud-detector/7' -> 7."""
    return int(package_arn.rstrip("/").rsplit("/", 1)[1])


def package_from_args(args) -> str:
    """--package-version N (what the workflows pass: no account ID in a public log) or --package-arn ARN."""
    if getattr(args, "package_version", None):
        return endpoints.package_arn(int(args.package_version))
    return args.package_arn


def set_status(package_arn: str, status: str, reason: str, sm=None) -> None:
    """Approved or Rejected in the SageMaker registry (only after the GitHub Environment approval)."""
    sm_client(sm).update_model_package(ModelPackageArn=package_arn, ModelApprovalStatus=status,
                                       ApprovalDescription=reason[:1000])


def prod_version(sm=None) -> int | None:
    """The package version ccfd-prod serves (its model is ccfd-prod-model-v<N>), or None when there is no ccfd-prod."""
    described = endpoints.describe(PROD, sm=sm_client(sm))
    if not described:
        return None
    config_name = described["EndpointConfigName"]
    variants = sm_client(sm).describe_endpoint_config(EndpointConfigName=config_name)["ProductionVariants"]
    model = variants[0]["ModelName"]
    return int(model.rsplit("-v", 1)[1]) if "-v" in model else None


def deploy_to_prod(version: int, *, canary: bool = True, log=print, sm=None) -> int | None:
    """ccfd-prod on package version N: created (with the 2 rollout alarms) when missing, else a canary update
    (auto-rollback on the alarms). Returns the version ccfd-prod serves afterwards."""
    sm = sm_client(sm)
    if not endpoints.endpoint_exists(PROD, sm=sm):
        endpoints.deploy_package(PROD, version, instance_count=PROD_INSTANCES, sm=sm, log=log)
        endpoints.create_rollout_alarms(endpoint=PROD, log=log)
        return prod_version(sm)
    endpoints.create_rollout_alarms(endpoint=PROD, log=log)          # idempotent: put_metric_alarm
    model_name, config_name = endpoints.model_names(PROD, version)
    if endpoints._find("DescribeModel", sm.describe_model, "Could not find model", ModelName=model_name) is None:
        endpoints.create_model_from_package(model_name, endpoints.package_arn(version, sm=sm), sm=sm, log=log)
    if endpoints._find("DescribeEndpointConfig", sm.describe_endpoint_config, "Could not find endpoint configuration",
                       EndpointConfigName=config_name) is None:
        endpoints.create_endpoint_config(config_name, model_name, instance_count=PROD_INSTANCES, sm=sm, log=log)
    endpoints.update(PROD, config_name, deployment_config=endpoints.canary_deployment_config() if canary else None,
                     sm=sm, log=log)
    return prod_version(sm)


def training_job_of(package_arn: str, sm=None) -> str:
    """The training job that made a package: its ModelDataUrl is s3://.../<job name>/output/model.tar.gz."""
    package = sm_client(sm).describe_model_package(ModelPackageName=package_arn)
    url = package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    return url.rsplit("/output/", 1)[0].rsplit("/", 1)[1]


def sync_champion(package_arn: str, log=print, sm=None) -> str:
    """Point MLflow @champion at the version made from this package's training run (registered if needed).
    Returns the MLflow version. Raises when the run cannot be found (never a silent half sync)."""
    mlflow_layer.connect()
    job = training_job_of(package_arn, sm)
    run = mlflow_layer.latest_run("training_job", job)
    if run is None:
        raise RuntimeError(f"no MLflow run with training_job={job}: the job did not log to the MLflow App")
    version = mlflow_layer.register(run.info.run_id, {"package_version": version_of(package_arn), "training_job": job})
    before = mlflow_layer.set_alias("champion", version.version)
    log(f"@champion: version {before} -> {version.version} (package {version_of(package_arn)}, job {job})")
    return str(version.version)


def champion_package_version() -> int | None:
    """The package_version tag of the MLflow version @champion points at (None when there is no champion)."""
    mlflow_layer.connect()
    champion = mlflow_layer.get_alias("champion")
    if champion is None or "package_version" not in (champion.tags or {}):
        return None
    return int(champion.tags["package_version"])
