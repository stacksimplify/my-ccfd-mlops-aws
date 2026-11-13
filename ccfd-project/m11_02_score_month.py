"""
MLOps 11 Step-02: join the batch predictions with the month's labels and print the base project's business numbers.

    python m11_02_score_month.py --month 1
"""

import argparse
import json

import boto3
import pandas as pd

from mlops_utils import config
from utils.metrics_utils import calculate_business_metrics

parser = argparse.ArgumentParser(description="Net Benefit of one batch-scored month.")
parser.add_argument("--month", type=int, default=1)
month = parser.parse_args().month

bucket = config.load().get("MLOPS_JOBS_BUCKET")
text = boto3.client("s3", region_name=config.REGION).get_object(
    Bucket=bucket, Key=f"batch/month_{month}/output/month_{month}.jsonl.out")["Body"].read().decode()
rows = [json.loads(line) for line in text.splitlines() if line.strip()]
rows = [row[0] if isinstance(row, list) else row for row in rows]
labels = pd.read_csv(f"data/production_month_{month}.csv")["is_fraud"].tolist()
predicted = [1 if row["prediction"] == "FRAUD" else 0 for row in rows]
business = calculate_business_metrics(labels, predicted)
print(f"month_{month}: {len(rows):,} predictions for {len(labels):,} rows, {sum(labels)} frauds")
print(f"caught {business['fraud_caught']}, missed {business['fraud_missed']}, false alarms "
      f"{business['false_alarms']:,}, Net Benefit ${int(business['net_benefit']):,}")
