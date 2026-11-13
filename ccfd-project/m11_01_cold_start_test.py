"""
MLOps 11 Step-01: the cold start and the warm latency of ccfd-serverless.

    python m11_01_cold_start_test.py [--calls 20]

The first call after deploy (or after a long idle time) waits for compute: that is the cold start. Then 2 warm-up and
--calls timed calls. p50 and p95 go to s3://<jobs bucket>/results/latency/serverless.json.
"""

import argparse
import json
import time
from pathlib import Path

import boto3

from mlops_utils import config, endpoints

parser = argparse.ArgumentParser(description="Time cold and warm calls on ccfd-serverless.")
parser.add_argument("--calls", type=int, default=20)
args = parser.parse_args()

say = lambda text: print(config.mask(str(text)), flush=True)
transactions = json.loads(Path("tests/payloads/sample_transactions.json").read_text())
start = time.perf_counter()
answer = endpoints.invoke("ccfd-serverless", transactions, log=say)
cold = time.perf_counter() - start
first = json.loads(answer["body"])[0]
say(f"Cold call: {cold:.2f} s ({answer['attempts']} attempt(s)); first answer {first['prediction']} "
    f"{100 * first['probability']:.2f}% {first['risk_score']}")
for _ in range(2):
    endpoints.invoke("ccfd-serverless", transactions, log=say)
times = []
for _ in range(args.calls):
    start = time.perf_counter()
    endpoints.invoke("ccfd-serverless", [transactions[0]], log=say)
    times.append(1000 * (time.perf_counter() - start))
times.sort()
p50, p95 = times[len(times) // 2], times[int(len(times) * 0.95) - 1]
say(f"Warm calls from this machine ({args.calls}): p50 {p50:.0f} ms, p95 {p95:.0f} ms")
bucket = config.load().get("MLOPS_JOBS_BUCKET")
boto3.client("s3", region_name=config.REGION).put_object(
    Bucket=bucket, Key="results/latency/serverless.json",
    Body=json.dumps({"endpoint": "ccfd-serverless", "cold_s": round(cold, 2), "p50_ms": round(p50),
                     "p95_ms": round(p95)}).encode())
say(f"Saved: s3://{bucket}/results/latency/serverless.json")
