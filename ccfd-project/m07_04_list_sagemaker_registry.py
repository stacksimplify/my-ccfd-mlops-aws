"""
MLOps 07 Step-04: what does the SageMaker Model Registry hold after the MLflow registrations of Step-03?

    python m07_04_list_sagemaker_registry.py

It prints the App's ModelRegistrationMode and every model package group whose name starts with ccfd, with its
packages. Our App keeps registration off (AutoModelRegistrationDisabled): MLflow versions stay in MLflow, and no
group named ccfd-fraud-model appears here. MLOps 09 builds our own deployable group, ccfd-fraud-detector.
Read only; boto3 with your AWS profile.
"""

import os
import sys

import boto3

from mlops_utils import config


def groups(sagemaker) -> list[dict]:
    found, token = [], None
    while True:
        page = sagemaker.list_model_package_groups(NameContains="ccfd", MaxResults=100, **({"NextToken": token} if token else {}))
        found += page["ModelPackageGroupSummaryList"]
        token = page.get("NextToken")
        if not token:
            return found


def packages(sagemaker, group: str) -> list[dict]:
    return sagemaker.list_model_packages(ModelPackageGroupName=group, MaxResults=100)["ModelPackageSummaryList"]


def main() -> int:
    sagemaker = boto3.client("sagemaker", region_name=config.REGION)
    arn = os.environ.get("MLFLOW_TRACKING_URI", "")
    if arn.startswith("arn:"):
        mode = sagemaker.describe_mlflow_app(Arn=arn)["ModelRegistrationMode"]
        print(f"MLflow App registration mode: {mode}")
    found = groups(sagemaker)
    if not found:
        print("SageMaker model package groups named ccfd: none")
    for group in found:
        name = group["ModelPackageGroupName"]
        rows = packages(sagemaker, name)
        print(f"Group {name}: {len(rows)} package(s)")
        for row in rows:
            print(f"  version {row.get('ModelPackageVersion')}  {row.get('ModelApprovalStatus')}  "
                  f"{row['CreationTime']:%Y-%m-%d %H:%M}")
    auto = [g for g in found if g["ModelPackageGroupName"] == "ccfd-fraud-model"]
    print("No ccfd-fraud-model group: MLflow registrations stayed in MLflow." if not auto
          else "A ccfd-fraud-model group exists: registration was on when it was made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
