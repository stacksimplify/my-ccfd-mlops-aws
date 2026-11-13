"""
MLOps 10 Step-03: call ccfd-rt with the base project's 3 test transactions, then measure the latency.

    python m10_03_invoke_realtime.py [--calls 50]

Each transaction of tests/payloads/sample_transactions.json is sent with invoke_endpoint (signed with SigV4 by
boto3: there is no URL). Then 2 warm-up and --calls timed calls from this machine (your EC2 VM, in us-east-1 like
the endpoint); p50 and p95 go to s3://<jobs bucket>/results/latency/realtime.json (MLOps 11 compares them with
serverless).
"""

import argparse
import json
import time
from pathlib import Path

import boto3

from mlops_utils import config, endpoints

SECTION_00 = ["FRAUD 60.60% HIGH", "FRAUD 99.84% CRITICAL", "FRAUD 99.60% CRITICAL"]   # Section 00-A Step-06-13, p5_05

parser = argparse.ArgumentParser(description="Invoke ccfd-rt and measure latency.")
parser.add_argument("--calls", type=int, default=50)
args = parser.parse_args()

say = lambda text: print(config.mask(str(text)), flush=True)
transactions = json.loads(Path("tests/payloads/sample_transactions.json").read_text())
for i, transaction in enumerate(transactions):
    answer = json.loads(endpoints.invoke("ccfd-rt", [transaction], log=say)["body"])[0]
    say(f"{i + 1}. {answer['prediction']} {100 * answer['probability']:.2f}% {answer['risk_score']:<8} "
        f"(Section 00-A Step-06-13: {SECTION_00[i]})")

for _ in range(2):
    endpoints.invoke("ccfd-rt", transactions, log=say)
times = []
for _ in range(args.calls):
    start = time.perf_counter()
    endpoints.invoke("ccfd-rt", [transactions[0]], log=say)
    times.append(1000 * (time.perf_counter() - start))
times.sort()
p50, p95 = times[len(times) // 2], times[int(len(times) * 0.95) - 1]
say(f"Latency over {args.calls} calls from this machine: p50 {p50:.0f} ms, p95 {p95:.0f} ms")
bucket = config.load().get("MLOPS_JOBS_BUCKET")
if bucket:
    boto3.client("s3", region_name=config.REGION).put_object(
        Bucket=bucket, Key="results/latency/realtime.json",
        Body=json.dumps({"endpoint": "ccfd-rt", "calls": args.calls, "p50_ms": round(p50), "p95_ms": round(p95)}).encode())
    say(f"Saved: s3://{bucket}/results/latency/realtime.json")
