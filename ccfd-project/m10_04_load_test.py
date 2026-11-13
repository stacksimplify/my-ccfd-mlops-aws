"""
MLOps 10 Step-04 and Step-05: steady load on ccfd-rt, with the instance count each minute.

    python m10_04_load_test.py --rps 20 --minutes 12

Random rows of the base project's 2,000-row test split (seed 42), up to 32 calls at a time. Each minute it prints the calls sent, the errors, the
p95 latency and the endpoint's instance count (current/desired).
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from mlops_utils import config, endpoints, gate

parser = argparse.ArgumentParser(description="Load test ccfd-rt.")
parser.add_argument("--rps", type=float, default=20)
parser.add_argument("--minutes", type=float, default=12)
parser.add_argument("--endpoint", default="ccfd-rt")
args = parser.parse_args()

if not Path(gate.BASE_FE_FILE).is_file():
    raise SystemExit(f"STOP: no base project test data here ({gate.BASE_FE_FILE}). Copy it from the jobs bucket "
                     "first (MLOps 10 Step-03-03).")
rows = gate.base_test_set().drop(columns=[gate.TARGET, "amount_ratio"], errors="ignore").to_dict("records")
runtime = endpoints.runtime_client()
sm = boto3.client("sagemaker", region_name=config.REGION)
lock, minute = threading.Lock(), {"sent": 0, "errors": 0, "ms": []}


def call() -> None:
    start = time.perf_counter()
    try:
        runtime.invoke_endpoint(EndpointName=args.endpoint, ContentType="application/json",
                                Body=json.dumps([random.choice(rows)]))
        ok = True
    except (ClientError, BotoCoreError):   # a load test counts every AWS failure
        ok = False
    with lock:
        minute["sent"] += 1
        minute["errors"] += 0 if ok else 1
        minute["ms"].append(1000 * (time.perf_counter() - start))


total_sent = total_errors = 0
end = time.time() + 60 * args.minutes
with ThreadPoolExecutor(max_workers=32) as pool:
    for number in range(1, int(args.minutes) + 1):
        stop = min(time.time() + 60, end)
        while time.time() < stop:
            pool.submit(call)
            time.sleep(1 / args.rps)
        with lock:
            sent, errors, ms = minute["sent"], minute["errors"], sorted(minute["ms"]) or [0]
            minute.update({"sent": 0, "errors": 0, "ms": []})
        variant = sm.describe_endpoint(EndpointName=args.endpoint)["ProductionVariants"][0]
        total_sent, total_errors = total_sent + sent, total_errors + errors
        print(f"min {number:>2}: sent {sent:>5} err {errors} p95 {ms[int(len(ms) * 0.95) - 1]:.0f} ms "
              f"inst {variant.get('CurrentInstanceCount')}/{variant.get('DesiredInstanceCount')}", flush=True)
print(f"Total: sent {total_sent:,}, errors {total_errors}")
