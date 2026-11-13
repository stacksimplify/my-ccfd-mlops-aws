"""
MLOps 08 Step-02: the processing job. Data checks (3 hard rules) and amount_ratio, on SageMaker.

    python m08_02_run_processing_job.py [--dvc-repo ~/mlops-lab/ccfd-project] [--dvc-rev data-v1]

Input: the DVC object of that data version in S3 (MLOps 06 pushed it; the md5 comes from the lab repo's pointer).
The job runs our training image with the entrypoint m08_02_data_checks.py on one ml.t3.medium: it fails on a broken hard
rule (H1 a label not 0 or 1, H2 an empty cell, H3 a value the serving API would refuse), without changing a row; then
it runs the base project's p4_01 unchanged. Outputs: fe/ (the FE1 CSV) and report/ in the jobs bucket.
"""

import argparse
import json
from pathlib import Path

import boto3

import m08_common as m08
from mlops_utils import config, jobs

parser = argparse.ArgumentParser(description="Run the data checks + amount_ratio as a SageMaker processing job.")
parser.add_argument("--dvc-repo", type=Path, default=config.LAB_REPO)
parser.add_argument("--dvc-rev", default="data-v1")
args = parser.parse_args()

env = m08.settings("MLOPS_SUFFIX", "MLOPS_JOBS_BUCKET", "MLOPS_EXECUTION_ROLE_ARN", "MLOPS_TRAINING_IMAGE_URI")
md5, raw = m08.dvc_object(args.dvc_repo.expanduser(), args.dvc_rev, env.get("MLOPS_SUFFIX"))
bucket, base = env.get("MLOPS_JOBS_BUCKET"), f"m08/{args.dvc_rev}"
name = f"ccfd-fe-{args.dvc_rev.replace('data-', '')}-{m08.stamp()}"
m08.say(f"Input: {args.dvc_rev} (md5 {md5}) = {raw}")
jobs.start_processing_job(
    name, env.get("MLOPS_TRAINING_IMAGE_URI"), env.get("MLOPS_EXECUTION_ROLE_ARN"),
    ["python3", "/opt/program/m08_02_data_checks.py"],
    [jobs.processing_input("raw", raw, "/opt/ml/processing/input/raw")],
    [jobs.processing_output("fe", f"s3://{bucket}/{base}/fe/", "/opt/ml/processing/output/fe"),
     jobs.processing_output("report", f"s3://{bucket}/{base}/report/", "/opt/ml/processing/output/report")],
    tags=env.aws_tags() + [{"Key": "dvc_md5", "Value": md5}], log=m08.say)
jobs.wait_for_processing_job(name, log=m08.say)

s3 = boto3.client("s3", region_name=config.REGION)
report = json.loads(s3.get_object(Bucket=bucket, Key=f"{base}/report/data_checks_report.json")["Body"].read())
m08.say(f"Data checks: {report.get('status')}, {report.get('rows'):,} rows")
rules = report.get("hard_rules", {})
m08.say("Hard rules: " + ", ".join(f"{rule} {value['rows']:,}" for rule, value in rules.items()))
m08.say(f"Outputs: s3://{bucket}/{base}/fe/ and s3://{bucket}/{base}/report/")
