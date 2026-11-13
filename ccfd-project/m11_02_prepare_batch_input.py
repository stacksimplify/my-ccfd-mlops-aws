"""
MLOps 11 Step-02: month_<N> as a JSON Lines file in S3, with the 8 raw fields only (the labels stay here).

    python m11_02_generate_months.py                 first: data/production_month_1.csv ... _4.csv (5,000 rows each)
    python m11_02_prepare_batch_input.py --month 1
"""

import argparse
import json

import boto3
import pandas as pd

from mlops_utils import config

parser = argparse.ArgumentParser(description="Upload one month as JSON Lines for Batch Transform.")
parser.add_argument("--month", type=int, default=1)
month = parser.parse_args().month

data = pd.read_csv(f"data/production_month_{month}.csv")
fields = [column for column in data.columns if column != "is_fraud"]
lines = "\n".join(json.dumps(record) for record in data[fields].to_dict(orient="records")) + "\n"
bucket, key = config.load().get("MLOPS_JOBS_BUCKET"), f"batch/month_{month}/input/month_{month}.jsonl"
boto3.client("s3", region_name=config.REGION).put_object(Bucket=bucket, Key=key, Body=lines.encode())
print(f"month_{month}: {len(data):,} rows ({int(data['is_fraud'].sum())} frauds; the labels are not uploaded), "
      f"{len(fields)} fields -> s3://{bucket}/{key}")
