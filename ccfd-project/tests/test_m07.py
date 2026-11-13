"""MLOps 07: the connection check stops cleanly without a URI; the registry lister reads pages (botocore Stubber)."""
from datetime import datetime

import boto3
from botocore.stub import Stubber

import m07_02_check_connection as m07_02
import m07_04_list_sagemaker_registry as m07_04


def test_connection_check_stops_without_a_tracking_uri(monkeypatch, capsys):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert m07_02.main() == 1
    assert "MLFLOW_TRACKING_URI is not set" in capsys.readouterr().out


def test_groups_follows_next_token_and_packages_are_listed():
    client = boto3.client("sagemaker", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x")
    summary = {"ModelPackageGroupName": "ccfd-fraud-detector", "ModelPackageGroupArn": "arn:x", "CreationTime": datetime(2026, 10, 21),
               "ModelPackageGroupStatus": "Completed"}
    with Stubber(client) as stub:
        stub.add_response("list_model_package_groups", {"ModelPackageGroupSummaryList": [], "NextToken": "t"},
                          {"NameContains": "ccfd", "MaxResults": 100})
        stub.add_response("list_model_package_groups", {"ModelPackageGroupSummaryList": [summary]},
                          {"NameContains": "ccfd", "MaxResults": 100, "NextToken": "t"})
        assert [g["ModelPackageGroupName"] for g in m07_04.groups(client)] == ["ccfd-fraud-detector"]
