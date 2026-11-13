"""MLOps 14: the cd.yml / rollback.yml helpers. No AWS: botocore Stubber and source checks."""
import datetime
from pathlib import Path

import boto3
from botocore.stub import Stubber

from mlops_utils import endpoints, release

ACCOUNT = "".join(str(digit % 10) for digit in range(1, 13))
GROUP_ARN = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/ccfd-fraud-detector"
JOBS = "ccfd-mlops-jobs-a1b2c3"


def test_version_of():
    assert release.version_of(f"{GROUP_ARN}/7") == 7


def test_staging_uses_a_plain_model_never_model_package_name():
    source = Path("m14_03_deploy_staging.py").read_text()
    assert "plain=True" in source and "ModelPackageName" not in source
    plain = Path("mlops_utils/endpoints.py").read_text().split("def create_plain_model", 1)[1].split("\ndef ", 1)[0]
    assert 'PrimaryContainer={"Image": container["Image"], "ModelDataUrl": container["ModelDataUrl"]}' in plain
    assert '"ModelPackageName":' not in plain


def test_training_job_of_reads_the_model_data_url():
    sm = boto3.client("sagemaker", region_name="us-east-1")
    stub = Stubber(sm)
    stub.add_response("describe_model_package", {
        "ModelPackageName": "ccfd-fraud-detector", "ModelPackageArn": f"{GROUP_ARN}/3",
        "CreationTime": datetime.datetime(2026, 10, 21), "ModelPackageStatus": "Completed",
        "ModelPackageStatusDetails": {"ValidationStatuses": []},
        "InferenceSpecification": {"Containers": [{
            "Image": "img", "ModelDataUrl": f"s3://{JOBS}/pipeline-models/pipelines-abc-ProductionTrain-XYZ/output/model.tar.gz"}],
            "SupportedContentTypes": ["application/json"], "SupportedResponseMIMETypes": ["application/json"]}},
        {"ModelPackageName": f"{GROUP_ARN}/3"})
    with stub:
        assert release.training_job_of(f"{GROUP_ARN}/3", sm) == "pipelines-abc-ProductionTrain-XYZ"


def test_set_status_writes_the_reason():
    sm = boto3.client("sagemaker", region_name="us-east-1")
    stub = Stubber(sm)
    stub.add_response("update_model_package", {"ModelPackageArn": f"{GROUP_ARN}/3"},
                      {"ModelPackageArn": f"{GROUP_ARN}/3", "ModelApprovalStatus": "Approved",
                       "ApprovalDescription": "cd.yml: approved by kalyan"})
    with stub:
        release.set_status(f"{GROUP_ARN}/3", "Approved", "cd.yml: approved by kalyan", sm)


def test_guarded_rollout_rejects_a_rolled_back_package():
    source = Path("m14_03_guarded_rollout.py").read_text()
    assert '"Rejected"' in source and "sys.exit(1)" in source
    assert endpoints.canary_deployment_config()["AutoRollbackConfiguration"]["Alarms"]
