"""
MLOps 11 Step-03: real-time vs serverless vs batch, from what this course measured and dated prices.

    python m11_03_compare_options.py

Reads results/latency/realtime.json (MLOps 10) and serverless.json (Step-01) from the jobs bucket, the newest
ccfd-batch-month-* transform job, and pricing_us_east_1.json. A missing measurement prints "not measured".
The serverless price per 1,000 predictions uses the round trip measured from the caller, network included, so it is an
upper bound ("up to"): serverless bills only the time the container works.
"""

import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from mlops_utils import config

price = json.loads(Path("pricing_us_east_1.json").read_text())
s3, sm = boto3.client("s3", region_name=config.REGION), boto3.client("sagemaker", region_name=config.REGION)
bucket = config.load().get("MLOPS_JOBS_BUCKET")


def measured(name: str) -> dict:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=f"results/latency/{name}.json")["Body"].read())
    except ClientError:
        return {}


rt, sl = measured("realtime"), measured("serverless")
jobs = sm.list_transform_jobs(NameContains="ccfd-batch-month", SortBy="CreationTime", SortOrder="Descending",
                              MaxResults=1)["TransformJobSummaries"]
batch_seconds = None
if jobs and jobs[0]["TransformJobStatus"] == "Completed":
    described = sm.describe_transform_job(TransformJobName=jobs[0]["TransformJobName"])
    batch_seconds = (described["TransformEndTime"] - described["TransformStartTime"]).total_seconds()

hour = price["ml_m5_large_usd_per_hour"]
lat = lambda m: f"{m['p50_ms']} / {m['p95_ms']} ms" if m else "not measured"
sl_per_1000 = 1000 * (sl["p50_ms"] / 1000) * price["serverless_memory_gb"] * price["serverless_usd_per_gb_second"] if sl else None
print(f"Prices: {price['region']}, read {price['read_on']}; latency measured from this machine (network included)")
print(f"{'option':<11}{'p50 / p95':<22}{'cold start':<13}{'idle per day':<15}{'per 1,000 predictions':<24}best for")
print(f"{'real-time':<11}{lat(rt):<22}{'none':<13}{f'{24 * hour:.2f} USD':<15}{'(the hour price)':<24}card checks at payment time")
print(f"{'serverless':<11}{lat(sl):<22}{(str(sl.get('cold_s')) + ' s') if sl else 'not measured':<13}{'0.00 USD':<15}"
      f"{(f'up to {sl_per_1000:.4f} USD') if sl_per_1000 is not None else 'not measured':<24}rare calls (internal tools)")
batch = f"{batch_seconds:.0f} s for 5,000 rows" if batch_seconds else "not measured"
batch_cost = f"{batch_seconds / 3600 * hour / 5:.5f} USD" if batch_seconds else "not measured"
print(f"{'batch':<11}{batch:<22}{'(job start)':<13}{'0.00 USD':<15}{batch_cost:<24}back-tests once the labels are in")
