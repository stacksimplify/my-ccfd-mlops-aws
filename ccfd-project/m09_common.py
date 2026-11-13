"""
MLOps 09: the shared steps of the m09 scripts.

  tar_files(s3_uri)          the 3 model files of a model.tar.gz in S3: (model, evaluation dict)
  gate_holdout()             the fixed gate data (2,000 rows, seed 301), made on the laptop if it is missing
  package(version)           describe_model_package of ccfd-fraud-detector/<version>
"""

import io
import json
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path

import boto3

from mlops_utils import config, endpoints, gate

GROUP = endpoints.GROUP                     # ccfd-fraud-detector
HOLDOUT = Path("data/gate_holdout.csv")
HOLDOUT_COMMAND = [sys.executable, "m03_03_simulate_new_data.py", "--mode", "refresh", "--rows", "2000",
                   "--seed", "301", "--output", str(HOLDOUT)]


def sm():
    return boto3.client("sagemaker", region_name=config.REGION)


def tar_files(s3_uri: str):
    bucket, key = s3_uri.removeprefix("s3://").split("/", 1)
    body = boto3.client("s3", region_name=config.REGION).get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        model = pickle.load(tar.extractfile("model_latest.pkl"))
        evaluation = json.load(tar.extractfile("evaluation.json"))
    return model, evaluation


def gate_holdout():
    if not HOLDOUT.is_file():
        subprocess.run(HOLDOUT_COMMAND, check=True, stdout=subprocess.DEVNULL)
    return gate.load_gate_data(HOLDOUT)


def package(version) -> dict:
    return sm().describe_model_package(ModelPackageName=endpoints.package_arn(version))


def say(text: str) -> None:
    print(config.mask(str(text)), flush=True)
