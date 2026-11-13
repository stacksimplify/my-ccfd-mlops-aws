"""
MLOps 08 Step-04: the SageMaker model next to the model trained on your EC2 VM, on the base project's 2,000-row test split.

    python m08_04_compare_with_vm.py --model-tar s3://ccfd-mlops-jobs-<random6>/m08/data-v1/training/<job>/output/model.tar.gz

EC2 VM model: models/p5_03_production/model_latest.pkl (the base project's p5_03 on your EC2 VM, MLOps 08 Step-04-01).
Both are scored with the same rule (threshold 0.5, the base project's money rule). It also lists the files at the tar root.
"""

import argparse
import io
import pickle
import tarfile

import boto3

import m08_common as m08
from mlops_utils import config, gate

parser = argparse.ArgumentParser(description="Compare the SageMaker model with the model trained on your EC2 VM.")
parser.add_argument("--model-tar", required=True)
parser.add_argument("--vm-model", default="models/p5_03_production/model_latest.pkl")
args = parser.parse_args()

bucket, key = args.model_tar.removeprefix("s3://").split("/", 1)
body = boto3.client("s3", region_name=config.REGION).get_object(Bucket=bucket, Key=key)["Body"].read()
with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
    m08.say(f"model.tar.gz holds: {', '.join(sorted(tar.getnames()))}")
    cloud = pickle.load(tar.extractfile("model_latest.pkl"))
with open(args.vm_model, "rb") as handle:
    vm = pickle.load(handle)

data = gate.base_test_set()
rows = {"EC2 VM": gate.score(vm, data), "SageMaker": gate.score(cloud, data)}
features = data.drop(columns=[gate.TARGET])
differ = int((vm.predict(features) != cloud.predict(features)).sum())
m08.say(f"{'':<14}{'EC2 VM':>12}{'SageMaker':>12}")
for label, field in (("Net Benefit", "net_benefit"), ("caught", "fraud_caught"), ("missed", "fraud_missed"),
                     ("false alarms", "false_alarms")):
    m08.say(f"{label:<14}{rows['EC2 VM'][field]:>12,}{rows['SageMaker'][field]:>12,}")
m08.say(f"Predictions that differ: {differ} of {len(data):,}")
